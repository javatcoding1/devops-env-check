"""
DevOps Incident Response – Inference Script (v3)
=================================================
Token-optimised LLM agent with:
  • Compact structured prompt (Intent→Dependencies→Orchestrate)
  • Truncated logs (last 200 chars) & recent actions only (last 5)
  • Hard stop if system is healthy
  • Anti-loop safeguard (no 3× same action, no repeats)
  • Rule-based fallback for guaranteed progress
  • max_tokens=50 to minimise cost

Compatible with any OpenAI-compatible endpoint.
"""

from __future__ import annotations

import json
import os
import sys
import re

from dotenv import load_dotenv
from openai import OpenAI

from env.environment import DevOpsEnv, VALID_ACTIONS
from env.models import DevOpsAction
from env.graders import grade
from env.tasks import get_task_ids

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")

_MISSING = not all([API_BASE_URL, MODEL_NAME, HF_TOKEN])

client: OpenAI | None = None
if not _MISSING:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

SYSTEM_PROMPT = (
    "You are a DevOps SRE. Diagnose hidden root causes from logs/metrics, "
    "then fix using multi-step chains. Wrong actions worsen the system. "
    "If status is healthy → do_nothing. Never repeat a prior action. "
    "Reply with ONLY the action string."
)

SYSTEM_PROMPT_REASONING = (
    "You are a DevOps SRE. Diagnose hidden root causes from logs/metrics, "
    "then fix using multi-step chains. Wrong actions worsen the system. "
    "If status is healthy → do_nothing. Never repeat a prior action. "
    'Reply ONLY with JSON: {"action": "<action>", "reasoning": "<1-2 sentence explanation>"}'
)

# ---------------------------------------------------------------------------
# Token-optimised prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(state: dict, actions_taken: list[str]) -> str:
    """
    Compact prompt with truncated logs and recent history only.
    Designed to stay under ~800 tokens total.
    """
    # Truncate logs to last 200 chars to save tokens
    logs = state.get("logs", "")
    if len(logs) > 200:
        logs = "..." + logs[-200:]

    # Only send last 5 actions (not full history) to save tokens
    recent = actions_taken[-5:] if len(actions_taken) > 5 else actions_taken

    return f"""Resolve this DevOps incident in fewest steps.

STATUS: {state.get('status','unknown')}
CPU: {state.get('cpu_usage',0)}% | MEM: {state.get('memory_usage',0)}% | DB: {state.get('db_latency','low')}
SERVICES: {', '.join(state.get('services',[]))}
LOGS: {logs}
ACTIONS TAKEN: {json.dumps(recent)}

RULES:
- If status=healthy → do_nothing
- check_logs first if root cause unclear (status=down or multiple issues)
- Multi-step chains: memory_leak→clear_cache→restart_service:api | database_issue→optimize_database→restart_service:database
- CPU overload→scale_up:cpu | Cache failure→clear_cache
- NEVER repeat an action from ACTIONS TAKEN
- Wrong actions WORSEN system (+CPU, +MEM, +latency)

EXAMPLES:
cpu=92,mem=45,db=low,status=degraded,actions=[] → check_logs
cpu=94,logs="CPU overload confirmed",actions=["check_logs"] → scale_up:cpu
mem=85,logs="Memory leak confirmed",actions=["check_logs"] → clear_cache
actions=["check_logs","clear_cache"] → restart_service:api
db=high,logs="Missing indexes",actions=["check_logs"] → optimize_database
actions=["check_logs","optimize_database"] → restart_service:database
cpu=88,mem=82,db=high,status=down,actions=[] → check_logs
status=healthy → do_nothing
cpu=95,actions=["restart_service:database"] → check_logs (previous was wrong)
actions=["check_logs","scale_up:cpu","clear_cache","restart_service:api","optimize_database"] → restart_service:database

ACTIONS: restart_service:api | restart_service:database | scale_up:cpu | optimize_database | clear_cache | check_logs | do_nothing

Reply with ONLY one action. No explanation."""


def _build_prompt_reasoning(state: dict, actions_taken: list[str]) -> str:
    """
    Prompt that requests JSON {action, reasoning} output.
    Same token budget as the standard prompt.
    """
    logs = state.get("logs", "")
    if len(logs) > 200:
        logs = "..." + logs[-200:]

    recent = actions_taken[-5:] if len(actions_taken) > 5 else actions_taken

    return f"""Resolve this DevOps incident. Diagnose the root cause FIRST, then act.

STATUS: {state.get('status','unknown')}
CPU: {state.get('cpu_usage',0)}% | MEM: {state.get('memory_usage',0)}% | DB: {state.get('db_latency','low')}
SERVICES: {', '.join(state.get('services',[]))}
LOGS: {logs}
ACTIONS TAKEN: {json.dumps(recent)}

RULES:
- Analyze metrics+logs to identify root cause before acting
- If status=healthy → do_nothing
- check_logs first if root cause unclear
- Multi-step chains: memory_leak→clear_cache→restart_service:api | database_issue→optimize_database→restart_service:database
- CPU overload→scale_up:cpu | Cache failure→clear_cache
- NEVER repeat an action from ACTIONS TAKEN
- Wrong actions WORSEN system

ACTIONS: restart_service:api | restart_service:database | scale_up:cpu | optimize_database | clear_cache | check_logs | do_nothing

Reply with JSON ONLY: {{"action": "<action>", "reasoning": "<1-2 sentence diagnosis and why this action>"}}"""


# ---------------------------------------------------------------------------
# Core agent logic
# ---------------------------------------------------------------------------

def get_action(state: dict, actions_taken: list[str] | None = None) -> str:
    """
    Get next action with hard stop, anti-loop, and LLM fallback.
    """
    if actions_taken is None:
        actions_taken = []

    # ── HARD STOP: system is healthy ─────────────────────────────
    if state.get("status") == "healthy":
        return "do_nothing"

    # ── ANTI-LOOP: 3× same action in a row ───────────────────────
    if len(actions_taken) >= 3 and len(set(actions_taken[-3:])) == 1:
        action = actions_taken[-1]
        print(f"  🛑 Anti-loop: '{action}' repeated 3×. Forcing alternative.")
        return _select_fallback(state, actions_taken)

    # ── ANTI-LOOP: check_logs 3+ times total ─────────────────────
    if actions_taken.count("check_logs") >= 3:
        if _select_fallback(state, actions_taken) != "do_nothing":
            # Force a non-check_logs action
            fb = _select_fallback(state, actions_taken, exclude={"check_logs"})
            if fb != "do_nothing":
                print(f"  🔄 check_logs cap reached. Forcing: '{fb}'")
                return fb

    # ── LLM call ─────────────────────────────────────────────────
    if client is None:
        print("  ⚠ No LLM client. Using fallback.")
        return _select_fallback(state, actions_taken)

    prompt = _build_prompt(state, actions_taken)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=50,
        )
        raw = response.choices[0].message.content.strip()
        parsed = _parse_action(raw)
    except Exception as e:
        print(f"  ⚠ LLM error: {e}. Using fallback.")
        return _select_fallback(state, actions_taken)

    # ── Guard-rail: no repeats (except check_logs up to 3×) ──────
    if parsed in actions_taken and parsed != "do_nothing":
        if parsed == "check_logs" and actions_taken.count("check_logs") < 3:
            return parsed
        print(f"  🔄 Loop guard: '{parsed}' already taken. Selecting alternative.")
        parsed = _select_fallback(state, actions_taken)

    return parsed


def _parse_action(raw: str) -> str:
    """Extract valid action from LLM output."""
    cleaned = raw.strip().strip("`").strip("'\"").strip().lower()

    # Direct match
    if cleaned in VALID_ACTIONS:
        return cleaned

    # Substring match
    for action in VALID_ACTIONS:
        if action in cleaned:
            return action

    # Regex fallback
    for action in VALID_ACTIONS:
        if re.search(re.escape(action), cleaned, re.IGNORECASE):
            return action

    print(f"  ⚠ Could not parse: '{raw}'. Defaulting to do_nothing.")
    return "do_nothing"


def _select_fallback(
    state: dict,
    actions_taken: list[str],
    exclude: set[str] | None = None,
) -> str:
    """Rule-based fallback. Picks highest-priority untried action."""
    s = state
    exclude = exclude or set()
    candidates: list[str] = []

    log_count = actions_taken.count("check_logs")
    if log_count < 3 and "check_logs" not in exclude:
        candidates.append("check_logs")

    if s.get("cpu_usage", 0) > 70 and "scale_up:cpu" not in actions_taken:
        candidates.append("scale_up:cpu")

    if (
        "cache" not in s.get("services", []) or s.get("memory_usage", 0) > 70
    ) and "clear_cache" not in actions_taken:
        candidates.append("clear_cache")

    if "clear_cache" in actions_taken and "restart_service:api" not in actions_taken:
        candidates.append("restart_service:api")

    if s.get("db_latency", "low") in ("medium", "high") and "optimize_database" not in actions_taken:
        candidates.append("optimize_database")

    if "optimize_database" in actions_taken and "restart_service:database" not in actions_taken:
        candidates.append("restart_service:database")

    if "api" not in s.get("services", []) and "restart_service:api" not in actions_taken:
        candidates.append("restart_service:api")

    if "database" not in s.get("services", []) and "restart_service:database" not in actions_taken:
        candidates.append("restart_service:database")

    # Filter excluded
    candidates = [c for c in candidates if c not in exclude]

    if candidates:
        chosen = candidates[0]
        print(f"  🎯 Fallback: '{chosen}'")
        return chosen

    return "do_nothing"


# ---------------------------------------------------------------------------
# Reasoning-based fallback explanations
# ---------------------------------------------------------------------------

_REASONING_MAP = {
    "check_logs": "Investigating system logs to identify the hidden root cause before taking corrective action.",
    "scale_up:cpu": "CPU usage is critically high, indicating CPU overload. Scaling up to relieve pressure.",
    "clear_cache": "Memory usage is elevated or cache is missing, suggesting cache corruption or memory leak. Clearing cache.",
    "restart_service:api": "Cache was cleared; restarting the API service to apply the fix and recover from memory leak.",
    "optimize_database": "Database latency is elevated, indicating slow queries or missing indexes. Optimizing database.",
    "restart_service:database": "Database was optimized; restarting the database service to apply configuration changes.",
    "do_nothing": "System appears healthy. No further action required.",
}


def _generate_fallback_reasoning(action: str, state: dict) -> str:
    """Generate human-readable reasoning for a fallback action."""
    # Try the static map first
    if action in _REASONING_MAP:
        base = _REASONING_MAP[action]
    else:
        base = f"Executing '{action}' based on current system metrics."

    # Enrich with actual metrics
    cpu = state.get("cpu_usage", 0)
    mem = state.get("memory_usage", 0)
    db = state.get("db_latency", "low")
    status = state.get("status", "unknown")

    if action == "check_logs" and status == "down":
        return f"System is DOWN (CPU={cpu}%, MEM={mem}%). Checking logs to understand the root cause before any fix."
    if action == "scale_up:cpu" and cpu > 80:
        return f"CPU at {cpu}% indicates severe overload. Scaling up CPU resources to stabilize the system."
    if action == "clear_cache" and mem > 70:
        return f"Memory at {mem}% is critically high. Clearing cache to free memory and address potential leak."
    if action == "optimize_database" and db in ("medium", "high"):
        return f"Database latency is {db}. Optimizing queries and indexes to reduce response times."

    return base


def get_action_with_reasoning(
    state: dict,
    actions_taken: list[str] | None = None,
) -> dict:
    """
    Get next action WITH structured reasoning.
    Returns {"action": str, "reasoning": str}.
    Falls back to rule-based reasoning when LLM is unavailable.
    Does NOT change any core logic — wraps get_action internally.
    """
    if actions_taken is None:
        actions_taken = []

    # ── HARD STOP: system is healthy ─────────────────────────────
    if state.get("status") == "healthy":
        return {"action": "do_nothing", "reasoning": "System is healthy. No action needed."}

    # ── ANTI-LOOP: 3× same action in a row ───────────────────────
    if len(actions_taken) >= 3 and len(set(actions_taken[-3:])) == 1:
        action = _select_fallback(state, actions_taken)
        return {"action": action, "reasoning": f"Anti-loop: previous action repeated 3×. {_generate_fallback_reasoning(action, state)}"}

    # ── ANTI-LOOP: check_logs 3+ times total ─────────────────────
    if actions_taken.count("check_logs") >= 3:
        fb = _select_fallback(state, actions_taken, exclude={"check_logs"})
        if fb != "do_nothing":
            return {"action": fb, "reasoning": f"Logs checked 3× already. {_generate_fallback_reasoning(fb, state)}"}

    # ── LLM call with reasoning ──────────────────────────────────
    if client is not None:
        prompt = _build_prompt_reasoning(state, actions_taken)
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_REASONING},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=100,
            )
            raw = response.choices[0].message.content.strip()
            parsed = _parse_reasoning_response(raw)
            if parsed:
                action = parsed["action"]
                reasoning = parsed["reasoning"]
                # Guard-rail: no repeats
                if action in actions_taken and action != "do_nothing":
                    if action == "check_logs" and actions_taken.count("check_logs") < 3:
                        return {"action": action, "reasoning": reasoning}
                    action = _select_fallback(state, actions_taken)
                    reasoning = f"LLM suggested repeat. {_generate_fallback_reasoning(action, state)}"
                return {"action": action, "reasoning": reasoning}
        except Exception as e:
            print(f"  ⚠ LLM reasoning error: {e}")

    # ── Fallback with reasoning ──────────────────────────────────
    action = get_action(state, actions_taken)
    reasoning = _generate_fallback_reasoning(action, state)
    return {"action": action, "reasoning": reasoning}


def _parse_reasoning_response(raw: str) -> dict | None:
    """Parse JSON {action, reasoning} from LLM output."""
    try:
        # Strip markdown code fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        data = json.loads(cleaned)
        if "action" in data:
            action = data["action"].strip().lower()
            if action in VALID_ACTIONS:
                return {
                    "action": action,
                    "reasoning": data.get("reasoning", "").strip(),
                }
            # Try substring match
            for valid in VALID_ACTIONS:
                if valid in action:
                    return {"action": valid, "reasoning": data.get("reasoning", "").strip()}
    except (json.JSONDecodeError, KeyError, IndexError):
        pass

    # Try to at least extract an action from the raw text
    action = _parse_action(raw)
    if action != "do_nothing":
        return {"action": action, "reasoning": ""}
    return None


# ---------------------------------------------------------------------------
# Chat helper (used by /chat endpoint)
# ---------------------------------------------------------------------------

def chat_diagnose(state: dict, actions_taken: list[str], user_message: str) -> dict:
    """
    LLM-powered chat for diagnosis and debugging.
    Returns dict with diagnosis, reasoning, and suggested_action.
    """
    if client is None:
        return {
            "diagnosis": "LLM not configured.",
            "reasoning": "Set API_BASE_URL, MODEL_NAME, HF_TOKEN in .env",
            "suggested_action": "do_nothing",
        }

    logs = state.get("logs", "")
    if len(logs) > 300:
        logs = "..." + logs[-300:]

    recent = actions_taken[-5:]

    prompt = f"""You are a DevOps expert. A user asks about a system incident.

System: status={state.get('status')}, cpu={state.get('cpu_usage')}%, mem={state.get('memory_usage')}%, db={state.get('db_latency')}
Services: {', '.join(state.get('services', []))}
Logs: {logs}
Recent actions: {json.dumps(recent)}

User question: {user_message}

Reply in JSON: {{"diagnosis": "...", "reasoning": "...", "suggested_action": "one_valid_action"}}"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=200,
        )
        raw = response.choices[0].message.content.strip()
        # Try to parse JSON
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except Exception:
        # Fallback: extract what we can
        suggested = _select_fallback(state, actions_taken)
        return {
            "diagnosis": "Unable to fully diagnose. Check system metrics.",
            "reasoning": f"CPU={state.get('cpu_usage')}%, MEM={state.get('memory_usage')}%, DB={state.get('db_latency')}",
            "suggested_action": suggested,
        }


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def run_inference() -> dict:
    """Run all tasks and return {task_id: score}."""
    if _MISSING:
        print(
            "ERROR: Missing env vars: API_BASE_URL, MODEL_NAME, HF_TOKEN.\n"
            "Copy .env.example to .env and fill in the values.",
            file=sys.stderr,
        )
        sys.exit(1)

    env = DevOpsEnv()
    results: dict = {}

    for task_id in get_task_ids():
        print(f"\n{'='*60}")
        print(f"  TASK: {task_id}")
        print(f"{'='*60}")

        obs = env.reset(task_id)
        done = False
        step = 0
        actions_history: list[str] = []

        while not done:
            step += 1
            # Build state dict for get_action compatibility
            state_dict = {
                "status": obs.status,
                "cpu_usage": obs.cpu_usage,
                "memory_usage": obs.memory_usage,
                "db_latency": obs.db_latency,
                "services": obs.services,
                "logs": obs.logs
            }

            action_str = get_action(state_dict, actions_history)

            # Hard stop on do_nothing when healthy
            if action_str == "do_nothing" and obs.status == "healthy":
                print(f"  Step {step:>2}: ✅ System healthy. Stopping.")
                break

            print(f"  Step {step:>2}: action = {action_str}")

            obs = env.step(DevOpsAction(action_str=action_str))
            
            reward = obs.reward or 0.0
            done = obs.done or False
            
            actions_history.append(action_str)
            print(
                f"           reward={reward:+.4f} | status={obs.status}"
                f" | cpu={obs.cpu_usage}% mem={obs.memory_usage}%"
                f" db={obs.db_latency} | {obs.message}"
            )

        # Grade uses actions_taken and final obs
        final_obs_dict = {
            "status": obs.status,
            "cpu_usage": obs.cpu_usage,
            "memory_usage": obs.memory_usage,
            "db_latency": obs.db_latency,
            "services": obs.services,
            "logs": obs.logs
        }
        score = grade(task_id, env.actions_taken, final_obs_dict)
        results[task_id] = score
        print(f"\n  ✅ Score: {score:.4f}")

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for tid, sc in results.items():
        bar = "█" * int(sc * 20)
        print(f"  {tid:<10}  {sc:.4f}  {bar}")
    avg = sum(results.values()) / len(results) if results else 0.0
    print(f"\n  Average: {avg:.4f}")
    print(f"{'='*60}\n")

    return results


if __name__ == "__main__":
    run_inference()
