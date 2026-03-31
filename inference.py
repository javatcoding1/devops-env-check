"""
DevOps Incident Response – Inference Script (v7)
=================================================
Fixes over v6:
  • Dynamic prompt updating: tracks per-task score history and injects
    targeted guidance into prompts based on past performance weaknesses.
  • update_score_history() API for server/app.py to feed back scores.
  • _build_dynamic_guidance() analyses weak areas and crafts prompt addons.
  • All previous v6 fixes preserved.
"""

from __future__ import annotations

import json
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

from env.environment import DevOpsEnv, VALID_ACTIONS
from env.models import DevOpsAction, DevOpsObservation
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

# ---------------------------------------------------------------------------
# Global Configuration & Generic Dependencies
# ---------------------------------------------------------------------------

CONFIG = {
    "THRESHOLD_CPU": 65,
    "THRESHOLD_MEM": 75,
    "THRESHOLD_DB": ["medium", "high"],
    "PIVOT_REWARD_THRESHOLD": 0.20,
    "PIVOT_CONSECUTIVE_FAILURES": 2,
}

# Generic mapping: Refresh/Scale actions -> Root Cause fixes required first
GENERIC_DEPENDENCIES = {
    "restart_service:api": ["clear_cache"],
    "restart_service:database": ["optimize_database"],
}

_score_history: dict[str, list[dict]] = {}
# Structure: { "easy": [{"score": 1.0, "steps": 2}, ...], "hard": [...] }

INTENT_CHAINS = {
    "RESOLVE_CPU_OVERLOAD": ["check_logs", "scale_up:cpu"],
    "RESOLVE_MEMORY_LEAK": ["check_logs", "clear_cache", "restart_service:api"],
    "RESOLVE_DATABASE_ISSUE": ["check_logs", "optimize_database", "restart_service:database"],
    "RESOLVE_CACHE_FAILURE": ["check_logs", "clear_cache"],
    "REFRESH_API_SERVICE": ["check_logs", "restart_service:api"],
    "REFRESH_DATABASE_SERVICE": ["check_logs", "restart_service:database"],
    "MAINTAIN_SYSTEM_HEALTH": ["no_action"],
}


def update_score_history(task_id: str, score: float, steps: int) -> None:
    """Called by app.py after each auto-run to record performance."""
    if task_id not in _score_history:
        _score_history[task_id] = []
    _score_history[task_id].append({"score": score, "steps": steps})
    # Keep only the last 10 entries per task to avoid unbounded growth
    _score_history[task_id] = _score_history[task_id][-10:]


def _build_dynamic_guidance(actions_taken: list[str], rewards_history: list[float]) -> str:
    """
    Analyse recent score history and return targeted prompt guidance.
    Uses the rewards_history from the CURRENT episode to provide real-time learning.
    """
    if not actions_taken or not rewards_history:
        return ""
    
    last_reward = rewards_history[-1]
    last_action = actions_taken[-1]
    
    guidance = "\n\n=== SRE DYNAMIC PERFORMANCE REVIEW (LEARNING FROM PREVIOUS STEP) ==="
    if last_reward >= 0.85:
        guidance += f"\n- SUCCESS: Your last action '{last_action}' was highly effective (Reward: {last_reward})."
        guidance += "\n- GUIDANCE: Logic confirms your current Intent is correct. Proceed to the NEXT step in the dependency chain."
    elif last_reward < 0.20:
        guidance += f"\n- FAILURE: Your last action '{last_action}' was ineffective (Low Reward: {last_reward})."
        guidance += "\n- GUIDANCE: Re-evaluate the logs carefully. You may have chosen the wrong INTENT. Pivot if necessary."
    else:
        guidance += f"\n- NEUTRAL: Your last action '{last_action}' had partial impact (Reward: {last_reward})."
        guidance += "\n- GUIDANCE: Continue your plan but monitor metrics closely."
    
    return guidance

# ---------------------------------------------------------------------------
# System Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert Senior SRE AI. Your ONLY objective: restore system health (status=healthy)\n"
    "using the minimum number of correct, high-impact actions.\n\n"
    "VALID ACTIONS (use EXACTLY these strings):\n"
    "  restart_service:api\n"
    "  restart_service:database\n"
    "  scale_up:cpu\n"
    "  optimize_database\n"
    "  clear_cache\n"
    "  check_logs\n"
    "  no_action\n\n"
    "DIAGNOSIS RULES:\n"
    "1. If status=healthy -> ALWAYS return no_action. This is the ONLY time no_action is valid.\n"
    "2. NEVER return no_action when status is degraded or down.\n"
    "3. If no log phrase is visible yet AND actions=[] -> use check_logs FIRST before any fix action.\n"
    "   Never restart a service or scale resources before reading the logs at least once.\n"
    "4. Log phrase action chains:\n"
    "   'Memory leak'                        -> clear_cache -> restart_service:api\n"
    "   'cache miss' / 'cache failure'        -> clear_cache\n"
    "   'CPU overload' / 'compute resources'  -> scale_up:cpu\n"
    "   'latency spikes' / 'missing indexes'  -> optimize_database -> restart_service:database\n"
    "   'Connection pool' / 'db connection'   -> restart_service:database\n"
    "   'api service unresponsive'            -> restart_service:api\n"
    "5. After clear_cache -> restart_service:api (complete the memory-leak chain).\n"
    "6. After optimize_database -> restart_service:database (complete the db chain).\n"
    "7. Do NOT repeat past actions EXCEPT:\n"
    "   - scale_up:cpu may repeat if CPU is STILL above 65% after first use.\n"
    "   - clear_cache may repeat if memory is STILL above 75% after first use.\n"
    "   - restart_service:database may repeat if db latency is STILL medium/high after optimize_database.\n"
    "8. After completing all known fix chains, inspect UNRESOLVED METRICS:\n"
    "   cpu >65% -> scale_up:cpu | mem >75% -> clear_cache | db medium/high -> restart_service:database\n"
    "9. STUCK STATE: If all fix actions have been used and status is still degraded or down,\n"
    "   repeat service restarts: restart_service:api or restart_service:database.\n\n"
    "FEW-SHOT EXAMPLES:\n"
    "  cpu=37%,mem=42%,db=low,status=degraded,logs='',actions=[] -> check_logs\n"
    "  cpu=92%,mem=45%,db=low,status=degraded,actions=[] -> check_logs\n"
    "  cpu=94%,logs='CPU overload: compute resources exhausted',actions=[check_logs] -> scale_up:cpu\n"
    "  mem=87%,logs='Memory leak detected',actions=[check_logs] -> clear_cache\n"
    "  mem=87%,logs='Memory leak detected',actions=[check_logs,clear_cache] -> restart_service:api\n"
    "  db=high,logs='latency spikes due to missing indexes',actions=[check_logs] -> optimize_database\n"
    "  db=high,actions=[check_logs,optimize_database] -> restart_service:database\n"
    "  status=healthy -> no_action\n"
    "  cpu=70%,mem=35%,db=medium,status=degraded,\n"
    "  actions=[check_logs,optimize_database,restart_service:database,scale_up:cpu,clear_cache,restart_service:api]\n"
    "  UNRESOLVED: cpu=70%(>65%), db=medium -> scale_up:cpu  (repeat allowed, cpu still high)\n\n"
    "  # Scenario G — Stuck state: metrics normal but status=degraded\n"
    "  cpu=38%,mem=35%,db=low,status=degraded,\n"
    "  actions=[check_logs,clear_cache,scale_up:cpu,restart_service:api,optimize_database,restart_service:database]\n"
    "  UNRESOLVED METRICS: none above threshold, but status=degraded\n"
    "  -> restart_service:api  (stuck state: all metrics normal, repeat service restart to recover)\n\n"
    "Reply with ONLY one action string. No explanation. No punctuation."
)


SYSTEM_PROMPT_REASONING = (
    "You are an elite Senior SRE Agent. Your objective: restore system health with 'Minimal Steps, Maximal Impact'.\n"
    "You operate under a strict 'Causal Priority' protocol to ensure root causes are fixed before symptoms.\n\n"
    "VALID ACTIONS:\n"
    "  restart_service:api | restart_service:database | scale_up:cpu | optimize_database | clear_cache | check_logs | no_action\n\n"
    "DIAGNOSTIC PROTOCOL (Causal Hierarchy):\n"
    "1. CAUSAL PRIORITY: If multiple issues exist, select the root cause based on this priority list:\n"
    "   1. Memory (Upstream: Leaks cause cache/DB/API pressure)\n"
    "   2. Cache (Middle: Failures cause DB/API overload)\n"
    "   3. Database (Middle: Latency causes API timeouts)\n"
    "   4. API (Service Layer: Down/Slow)\n"
    "   5. CPU (Downstream Symptom: Often caused by GC or DB wait)\n"
    "2. CRITICAL OVERRIDE: You may bypass the hierarchy ONLY if a downstream metric (e.g. CPU) is in a CRITICAL state (>95%) AND there is ZERO log/metric evidence for an upstream cause.\n"
    "3. NO UNKNOWN INTENT: You MUST name the hypothesized root cause. Generic terms are forbidden.\n"
    "4. FIX BEFORE VERIFY: You must APPLY a fix before you can re-check logs (VERIFY mode).\n\n"
    "FEW-SHOT EXAMPLES:\n"
    '  # Phase: Upstream Priority (Memory 88% + CPU 92%)\n'
    '  metrics: {mem=88%, cpu=92%}, logs: "GC pressure high", history: [check_logs]\n'
    '  -> {"root_cause": "Upstream Memory Leak", "evidence_logs": "GC pressure high", "evidence_metrics": "MEM=88%", "intent": "RESOLVE_UPSTREAM_MEMORY", "confidence": 95, "plan": "Memory is priority 1, CPU is priority 5. Fixing upstream cause to resolve downstream CPU pressure.", "action": "clear_cache"}\n\n'
    '  # Phase: Critical Override (CPU 97% + No Upstream Evidence)\n'
    '  metrics: {cpu=97%, mem=40%, db_lat=low}, logs: "None", history: [check_logs]\n'
    '  -> {"root_cause": "Critical CPU Spike", "evidence_logs": "None", "evidence_metrics": "CPU=97%", "intent": "CRITICAL_CPU_OVERRIDE", "confidence": 98, "plan": "CPU is critical (>95%) and no upstream issues found. Overriding hierarchy to stabilize system.", "action": "scale_up:cpu"}\n\n'
    "Reply with JSON ONLY:\n"
    '{"root_cause": "<string>", "evidence_logs": "<string>", "evidence_metrics": "<string>", "intent": "<string>", "confidence": <int>, "plan": "<string>", "action": "<string>"}'
)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_prompt(state: dict, actions_taken: list[str], feedback: str = "") -> str:
    """Compact prompt. Truncates logs to last 300 chars. Shows FULL action history."""
    logs = state.get("logs", "")
    if len(logs) > 300:
        logs = "..." + logs[-300:]

    prompt = f"""=== CURRENT SYSTEM STATE ===
STATUS  : {state.get('status', 'unknown')}
CPU     : {state.get('cpu_usage', 0)}%
MEMORY  : {state.get('memory_usage', 0)}%
DB LAT  : {state.get('db_latency', 'low')}
SERVICES: {', '.join(state.get('services', []))}
LOGS    : {logs}

=== ALL ACTIONS TAKEN THIS EPISODE (do NOT repeat unless explicitly allowed) ===
{json.dumps(actions_taken)}

=== UNRESOLVED METRICS (still above threshold) ===
{_unresolved_metrics_summary(state)}
"""
    if feedback:
        prompt += f"""
=== ⚠ SELF-CORRECTION REQUIRED ===
{feedback}
Your next response MUST address this feedback and choose a DIFFERENT valid action.
"""
    # Dynamic prompt guidance based on score history
    dynamic_guidance = _build_dynamic_guidance()
    if dynamic_guidance:
        prompt += dynamic_guidance

    prompt += "\nNext action (one string only):"
    return prompt


def _build_prompt_reasoning(state: dict, actions_taken: list[str], rewards_history: list[float] = None, feedback: str = "") -> str:
    if rewards_history is None:
        rewards_history = []
        
    logs = state.get("logs", "")
    if len(logs) > 300:
        logs = "..." + logs[-300:]

    prompt = f"""=== CURRENT SYSTEM STATE ===
STATUS  : {state.get('status', 'unknown')}
CPU     : {state.get('cpu_usage', 0)}%
MEMORY  : {state.get('memory_usage', 0)}%
DB LAT  : {state.get('db_latency', 'low')}
SERVICES: {', '.join(state.get('services', []))}
LOGS    : {logs}

=== ALL ACTIONS TAKEN THIS EPISODE (do NOT repeat unless explicitly allowed) ===
{json.dumps(actions_taken)}

=== UNRESOLVED METRICS (still above threshold) ===
{_unresolved_metrics_summary(state)}
"""
    if feedback:
        prompt += f"""
=== ⚠ SELF-CORRECTION REQUIRED ===
{feedback}
Your next JSON response MUST choose a DIFFERENT valid action that directly addresses the logs.
"""
    # Dynamic prompt guidance based on current episode performance
    dynamic_guidance = _build_dynamic_guidance(actions_taken, rewards_history)
    if dynamic_guidance:
        prompt += dynamic_guidance

    prompt += '\nRespond with JSON only: {"root_cause": "<string>", "evidence_logs": "<string>", "evidence_metrics": "<string>", "intent": "<string>", "confidence": <int>, "plan": "<string>", "action": "<valid_action>"}'
    return prompt


# ---------------------------------------------------------------------------
# Unresolved metrics helper
# ---------------------------------------------------------------------------

def _unresolved_metrics_summary(state: dict) -> str:
    cpu = state.get("cpu_usage", 0)
    mem = state.get("memory_usage", 0)
    db = state.get("db_latency", "low")
    status = state.get("status", "unknown")
    issues = []
    if cpu > 65:
        issues.append(f"CPU={cpu}% (>65% threshold) -> scale_up:cpu may be needed")
    if mem > 75:
        issues.append(f"MEM={mem}% (>75% threshold) -> clear_cache may be needed")
    if db in ("medium", "high"):
        issues.append(f"DB latency={db} -> optimize_database or restart_service:database may be needed")
    if status in ("degraded", "down"):
        issues.append(f"STATUS={status} -> system is NOT yet healthy, no_action is WRONG here")
    return "\n".join(issues) if issues else "No unresolved metrics — system may be healthy."


# ---------------------------------------------------------------------------
# Core agent
# ---------------------------------------------------------------------------

def get_action(state: dict, actions_taken: list[str] | None = None) -> str:
    if actions_taken is None:
        actions_taken = []

    if state.get("status") == "healthy":
        return "no_action"

    if client is None:
        print("  ⚠ No LLM client configured. Activating safety backup.")
        return _select_fallback(state, actions_taken)

    feedback = ""
    for attempt in range(3):
        prompt = _build_prompt(state, actions_taken, feedback=feedback)

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=60,
            )
            raw = response.choices[0].message.content.strip()
            parsed = _parse_action(raw)
        except Exception as exc:
            print(f"  ⚠ LLM API error (attempt {attempt + 1}): {exc}")
            print("  ⚠ API unavailable. Activating safety backup.")
            return _select_fallback(state, actions_taken)

        valid, feedback = _validate_action(parsed, actions_taken, state)
        if valid:
            if attempt > 0:
                print(f"  ✅ LLM self-corrected to '{parsed}' after {attempt} retry(s).")
            return parsed

        print(f"  🔄 LLM retry {attempt + 1}/3 — feedback: {feedback}")

    print("  ⚠ LLM self-correction exhausted (3 attempts). Activating safety backup.")
    
    exclude_actions = set()
    if len(actions_taken) >= 2 and actions_taken[-1] == actions_taken[-2]:
        exclude_actions.add(actions_taken[-1])
        
    return _select_fallback(state, actions_taken, exclude=exclude_actions)


def get_action_with_reasoning(
    state: dict,
    actions_taken: list[str] | None = None,
    rewards_history: list[float] | None = None,
) -> dict:
    if actions_taken is None:
        actions_taken = []
    if rewards_history is None:
        rewards_history = []

    if state.get("status") == "healthy":
        return {"action": "no_action", "reasoning": "System is healthy. No action needed."}

    if client is None:
        exclude_actions = set()
        if len(actions_taken) >= 2 and actions_taken[-1] == actions_taken[-2]:
            exclude_actions.add(actions_taken[-1])
        action = _select_fallback(state, actions_taken, exclude=exclude_actions)
        return {"action": action, "reasoning": _generate_fallback_reasoning(action, state)}

    feedback = ""
    for attempt in range(3):
        prompt = _build_prompt_reasoning(state, actions_taken, rewards_history=rewards_history, feedback=feedback)

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_REASONING},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=150,
            )
            raw = response.choices[0].message.content.strip()
            parsed = _parse_reasoning_response(raw)
        except Exception as exc:
            print(f"  ⚠ LLM reasoning API error (attempt {attempt + 1}): {exc}")
            action = _select_fallback(state, actions_taken)
            return {
                "action": action,
                "reasoning": f"[Safety Backup — API Error] {_generate_fallback_reasoning(action, state)}",
            }

        if parsed is None:
            feedback = (
                "Your previous response could not be parsed as valid JSON. "
                'You MUST reply with exactly: {"action": "<valid_action>", "intent": "<string>", "reasoning": "<text>"}'
            )
            print(f"  🔄 LLM reasoning retry {attempt + 1}/3 — parse failure.")
            continue

        action = parsed["action"]
        reasoning = parsed.get("reasoning", parsed.get("plan", ""))

        # Pass the full parsed response to validator for Evidence Verification
        valid, fb = _validate_action(parsed, actions_taken, state)
        if valid:
            if attempt > 0:
                reasoning = f"[Self-Corrected after {attempt} retry(s)] {reasoning}"
            return {"action": action, "reasoning": reasoning}

        feedback = fb
        print(f"  🔄 LLM reasoning retry {attempt + 1}/3 — feedback: {feedback}")

    exclude_actions = set()
    if len(actions_taken) >= 2 and actions_taken[-1] == actions_taken[-2]:
        exclude_actions.add(actions_taken[-1])

    action = _select_fallback(state, actions_taken, exclude=exclude_actions)
    return {
        "action": action,
        "reasoning": f"[Safety Backup — LLM exhausted] {_generate_fallback_reasoning(action, state)}",
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _verify_evidence(evidence: str, actual_logs: str) -> bool:
    """Hallucination Shield: Verifies if the AI's cited evidence exists in logs."""
    if not evidence or evidence.lower() in ("none", "n/a", "no evidence"):
        return True
    # Clean up quotes if LLM added them
    clean_ev = evidence.strip('"\'').lower()
    return clean_ev in actual_logs.lower()


def _validate_action(parsed: dict | str, actions_taken: list[str], state: dict) -> tuple[bool, str]:
    """
    Generalized SRE Guardian: Enforces Intent Lock, Generic Sequencing, and 
    Evidence Verification (Hallucination Shield).
    """
    if isinstance(parsed, dict):
        action = parsed.get("action", "check_logs")
        ev_logs = parsed.get("evidence_logs", "")
        ev_metrics = parsed.get("evidence_metrics", "")
    else:
        action = parsed
        ev_logs = ""
        ev_metrics = ""

    status = state.get("status", "unknown")
    cpu = state.get("cpu_usage", 0)
    mem = state.get("memory_usage", 0)
    db = state.get("db_latency", "low")
    actual_logs = (state.get("logs", "") or "").lower()

    if action not in VALID_ACTIONS:
        return False, f"Action '{action}' is not in the list of valid actions."

    if action == "no_action":
        if status == "healthy":
            return True, ""
        return False, f"GENERAL PROTOCOL: 'no_action' is invalid while status='{status}'."

    # ── 1. Hallucination Shield (Evidence Verification) ───────────
    if ev_logs and not _verify_evidence(ev_logs, actual_logs):
        return (
            False,
            f"HALLUCINATION ALERT: Your cited log evidence '{ev_logs}' does not exist in actual logs. "
            "You MUST ONLY use verbatim snippets from the provided LOGS section."
        )

    if action == "check_logs":
        if not actions_taken:
            return True, ""  # Always allowed at Step 1 (DIAGNOSE)
        if actions_taken[-1] == "check_logs":
            return (
                False,
                "REDUNDANCY ERROR: You just checked logs. You MUST apply a fix action "
                "before checking them again (VERIFY mode) to see if you improved the system."
            )
        # Ensure at least one fix or scale action happened since the LAST log check
        # (This is a simplified check: we just ensure the last action wasn't check_logs, 
        # which we already checked above. But more strictly, we want a 'fix-like' action).
        fix_actions = ["clear_cache", "optimize_database", "scale_up:cpu", "restart_service:api", "restart_service:database"]
        if not any(a in fix_actions for a in actions_taken):
            return False, "PROTOCOL ERROR: You must perform at least one fix action before re-checking logs."

    # ── 2. Metric Verification (Using CONFIG) ─────────────────────
    cache_keywords = ["gc", "cache", "checksum", "eviction", "rss"]
    if action == "clear_cache" and mem <= CONFIG["THRESHOLD_MEM"] and not any(k in actual_logs for k in cache_keywords):
        return (
            False,
            f"EVIDENCE MISMATCH: Attempted 'clear_cache' but MEM is {mem}% (Threshold: {CONFIG['THRESHOLD_MEM']}%) "
            "and logs show no cache or GC pressure. Re-examine metrics."
        )

    if action == "scale_up:cpu" and cpu <= CONFIG["THRESHOLD_CPU"]:
        return (
            False,
            f"EVIDENCE MISMATCH: Attempted 'scale_up:cpu' but CPU is {cpu}% (Threshold: {CONFIG['THRESHOLD_CPU']}%)."
        )

    if action == "optimize_database" and db not in CONFIG["THRESHOLD_DB"]:
        return (
            False,
            f"EVIDENCE MISMATCH: Attempted 'optimize_database' but DB latency is '{db}'."
        )

    # ── 3. Generic Dependency Sequencing ──────────────────────────
    if action in GENERIC_DEPENDENCIES:
        required_fixes = GENERIC_DEPENDENCIES[action]
        # If any of the required fixes are missing and the system is still stressed
        if not any(fix in actions_taken for fix in required_fixes):
            # Only block if there's evidence that a fix was actually needed
            if action == "restart_service:api" and (mem > CONFIG["THRESHOLD_MEM"] or "gc" in actual_logs):
                return False, f"SEQUENCING ERROR: You must apply a FIX (e.g. {required_fixes}) before refreshing the service."
            if action == "restart_service:database" and (db in CONFIG["THRESHOLD_DB"] or "sequential" in actual_logs):
                return False, f"SEQUENCING ERROR: You must apply a FIX (e.g. {required_fixes}) before refreshing the service."

    # ── 4. Protocol Rule 1: Step 1 MUST be check_logs ────────────
    if not actions_taken and action != "check_logs" and status != "healthy":
        return False, "SRE PROTOCOL: Initial action must be 'check_logs' for evidence collection."

    # ── 5. Circuit Breaker ────────────────────────────────────────
    # Block immediate repeats (A -> A -> A)
    if len(actions_taken) >= 2 and action == actions_taken[-1] == actions_taken[-2]:
        return False, f"CIRCUIT BREAKER: '{action}' has failed 3 times. Your analysis is flawed. Re-read logs."
        
    # Block alternating loops (A -> B -> A -> B)
    if len(actions_taken) >= 3 and action == actions_taken[-2] and actions_taken[-1] == actions_taken[-3]:
        return False, f"CIRCUIT BREAKER: Alternating loop detected ('{action}' and '{actions_taken[-1]}'). Break the cycle and re-read logs."

    return True, ""


def _parse_reasoning_response(raw: str) -> dict | None:
    """Parses Intent/Action JSON from LLM."""
    if not raw:
        return None
    
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
            
    try:
        data = json.loads(cleaned)
        # Structural normalization
        if "action" not in data:
            return None
            
        # Ensure intent exists or derive it
        if "intent" not in data:
            data["intent"] = "UNKNOWN_INTENT"
            
        return data
    except Exception:
        # Final attempt: extract action via regex if JSON fails
        import re
        match = re.search(r'"action":\s*"([^"]+)"', cleaned)
        if match:
            action = match.group(1)
            return {"action": action, "intent": "PARSE_FAILURE_RECOVERY", "root_cause": "JSON decode failed", "plan": "N/A"}
        return None


def _build_hint_from_state(state: dict, actions_taken: list[str]) -> str:
    logs = (state.get("logs", "") or "").lower()
    cpu = state.get("cpu_usage", 0)
    mem = state.get("memory_usage", 0)
    db = state.get("db_latency", "low")
    status = state.get("status", "unknown")
    hints = []

    # ── Noisy Log Signals ────────────────────────────────────────
    if any(x in logs for x in ["gc pressure", "heap rss", "oom"]):
        if "clear_cache" not in actions_taken:
            hints.append("Logs show GC pressure/RSS growth -> consider clear_cache to free memory.")
        elif "restart_service:api" not in actions_taken:
            hints.append("Memory was cleared -> consider restart_service:api to refresh the process.")

    if any(x in logs for x in ["context switching", "scheduler latency", "throttling"]):
        hints.append("Logs confirm scheduler saturation/throttling -> consider scale_up:cpu.")

    if any(x in logs for x in ["sequential scan", "iowait spiking", "index bloat"]):
        if "optimize_database" not in actions_taken:
            hints.append("Logs show iowait/sequential scans -> consider optimize_database.")
        elif "restart_service:database" not in actions_taken:
            hints.append("Database was optimized -> consider restart_service:database to apply changes.")

    if "buffer cache" in logs and "restart_service:database" not in actions_taken:
        hints.append("Logs mention buffer cache issues -> consider restart_service:database.")

    if any(x in logs for x in ["eviction", "checksum mismatch", "segment corruption"]):
        hints.append("Logs show cache corruption/eviction -> consider clear_cache.")

    # ── Residual metric pressure ──────────────────────────────────
    if not hints:
        if cpu > 65:
            hints.append(f"CPU is still {cpu}% (>65%) — scale_up:cpu is allowed to repeat.")
        if mem > 75:
            hints.append(f"Memory is still {mem}% (>75%) — clear_cache is allowed to repeat.")
        if db in ("medium", "high"):
            if "optimize_database" not in actions_taken:
                hints.append(f"DB latency is {db} — consider optimize_database.")
            elif "restart_service:database" not in actions_taken:
                hints.append(f"DB latency is {db} and DB was optimized — consider restart_service:database.")
            else:
                # Both optimize and restart have been tried — db still elevated.
                # This is the db-chain-exhausted stuck case.
                hints.append(
                    f"DB latency is still {db} after optimize_database + restart_service:database. "
                    f"Repeating restart_service:database is explicitly allowed when db remains elevated."
                )

    # ── Stuck-state: all metrics normal but status still broken ───
    if not hints and cpu <= 65 and mem <= 75 and db not in ("medium", "high") and status in ("degraded", "down"):
        hints.append(
            f"All metrics are within normal range but status='{status}'. "
            f"This is a stuck state — repeat a service restart: restart_service:api or restart_service:database."
        )

    return " ".join(hints) if hints else (
        "Re-read the logs and UNRESOLVED METRICS section to determine the correct next action."
    )


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _parse_action(raw: str) -> str:
    if not raw:
        return "check_logs"

    cleaned = raw.strip().strip("`").strip("'\"").lower()

    if "{" in cleaned and "action" in cleaned:
        try:
            data = json.loads(cleaned)
            cleaned = data.get("action", "").lower().strip()
        except Exception:
            pass

    analysis_words = ["analyze", "diagnose", "monitor", "investigate", "inspect", "examine"]
    if any(w in cleaned for w in analysis_words) and "log" not in cleaned:
        return "check_logs"

    if cleaned in VALID_ACTIONS:
        return cleaned

    for action in VALID_ACTIONS:
        if action in cleaned:
            return action

    print(f"  ⚠ Unparseable LLM output: '{raw}'. Defaulting to check_logs.")
    return "check_logs"


def _parse_reasoning_response(raw: str) -> dict | None:
    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            parts = cleaned.split("```")
            cleaned = parts[1] if len(parts) > 1 else cleaned
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        data = json.loads(cleaned.strip())
        if "action" not in data:
            return None
        action = data["action"].strip().lower()
        if action in VALID_ACTIONS:
            rc = data.get("root_cause", "Unknown issue")
            ev_l = data.get("evidence_logs", "No log evidence listed.")
            ev_m = data.get("evidence_metrics", "No metric evidence listed.")
            conf = data.get("confidence", 0)
            plan = data.get("plan", "No plan specified.")
            reasoning_str = f"[Confidence: {conf}%] Root Cause: {rc} | Logs: {ev_l} | Metrics: {ev_m} | Plan: {plan}"
            return {"action": action, "reasoning": reasoning_str}
        for valid in VALID_ACTIONS:
            if valid in action:
                rc = data.get("root_cause", "Unknown issue")
                ev_l = data.get("evidence_logs", "No log evidence listed.")
                ev_m = data.get("evidence_metrics", "No metric evidence listed.")
                conf = data.get("confidence", 0)
                plan = data.get("plan", "No plan specified.")
                reasoning_str = f"[Confidence: {conf}%] Root Cause: {rc} | Logs: {ev_l} | Metrics: {ev_m} | Plan: {plan}"
                return {"action": valid, "reasoning": reasoning_str}
    except (json.JSONDecodeError, KeyError, IndexError):
        pass

    action = _parse_action(raw)
    if action and action != "no_action":
        return {"action": action, "reasoning": ""}
    return None


# ---------------------------------------------------------------------------
# Safety backup
# ---------------------------------------------------------------------------

def _select_fallback(
    state: dict,
    actions_taken: list[str],
    exclude: set[str] | None = None,
) -> str:
    exclude = exclude or set()
    status = state.get("status", "unknown")
    cpu = state.get("cpu_usage", 0)
    mem = state.get("memory_usage", 0)
    db = state.get("db_latency", "low")
    logs = (state.get("logs", "") or "").lower()
    log_count = actions_taken.count("check_logs")

    if status == "healthy":
        return "no_action"

    candidates: list[str] = []

    # 1. Investigate first if logs not yet checked
    if log_count < 2 and "check_logs" not in exclude:
        candidates.append("check_logs")

    # 2. Noisy Log-driven first-use actions
    if any(x in logs for x in ["gc pressure", "heap rss", "oom"]):
        if "clear_cache" not in actions_taken:
            candidates.append("clear_cache")
        if "restart_service:api" not in actions_taken:
            candidates.append("restart_service:api")

    if any(x in logs for x in ["sequential scan", "iowait spiking", "index bloat"]):
        if "optimize_database" not in actions_taken:
            candidates.append("optimize_database")
        if "restart_service:database" not in actions_taken:
            candidates.append("restart_service:database")

    if any(x in logs for x in ["context switching", "scheduler latency", "throttling"]):
        if "scale_up:cpu" not in actions_taken:
            candidates.append("scale_up:cpu")

    if "buffer cache" in logs:
        if "restart_service:database" not in actions_taken:
            candidates.append("restart_service:database")

    if any(x in logs for x in ["eviction", "checksum mismatch", "segment corruption"]):
        if "clear_cache" not in actions_taken:
            candidates.append("clear_cache")

    # 3. Metric-driven first-use actions
    if not [c for c in candidates if c not in ("check_logs",)]:
        if cpu > 65 and "scale_up:cpu" not in actions_taken:
            candidates.append("scale_up:cpu")
        if mem > 70:
            if "clear_cache" not in actions_taken:
                candidates.append("clear_cache")
            if "restart_service:api" not in actions_taken:
                candidates.append("restart_service:api")
        if db in ("medium", "high"):
            if "optimize_database" not in actions_taken:
                candidates.append("optimize_database")
            if "restart_service:database" not in actions_taken:
                candidates.append("restart_service:database")

    # 4. Chain completion
    if "clear_cache" in actions_taken and "restart_service:api" not in actions_taken:
        candidates.append("restart_service:api")
    if "optimize_database" in actions_taken and "restart_service:database" not in actions_taken:
        candidates.append("restart_service:database")

    # 5. Service-down restarts
    if "api" not in state.get("services", []) and "restart_service:api" not in actions_taken:
        candidates.append("restart_service:api")
    if "database" not in state.get("services", []) and "restart_service:database" not in actions_taken:
        candidates.append("restart_service:database")

    # 6. Repeatable metric actions
    if cpu > 65:
        candidates.append("scale_up:cpu")
    if mem > 70:
        candidates.append("clear_cache")
    # DB still elevated after optimize — repeat database restart
    if db in ("medium", "high") and "optimize_database" in actions_taken:
        candidates.append("restart_service:database")

    # Deduplicate preserving order, filter excluded
    seen: set[str] = set()
    filtered: list[str] = []
    for c in candidates:
        if c not in exclude and c not in seen:
            seen.add(c)
            filtered.append(c)
    candidates = filtered

    if candidates:
        chosen = candidates[0]
        print(f"  🛡 Safety backup selected: '{chosen}'")
        return chosen

    # Last resort — repeat service restarts unconditionally
    last_resort_order = [
        "restart_service:api",
        "restart_service:database",
        "scale_up:cpu",
        "clear_cache",
        "optimize_database",
    ]
    for last_resort in last_resort_order:
        if last_resort not in exclude:
            print(f"  🛡 Safety backup last resort (repeat allowed): '{last_resort}'")
            return last_resort

    print("  ⚠ Safety backup has no options left. System may need manual intervention.")
    return "restart_service:api"


# ---------------------------------------------------------------------------
# Reasoning helpers
# ---------------------------------------------------------------------------

_REASONING_MAP = {
    "check_logs": "Investigating system logs to identify the root cause before taking corrective action.",
    "scale_up:cpu": "CPU usage is critically high. Scaling up to relieve pressure.",
    "clear_cache": "Memory usage is elevated or cache is failing. Clearing cache.",
    "restart_service:api": "Completing memory-leak recovery chain: cache cleared, now restarting the API service.",
    "optimize_database": "Database latency is elevated. Optimizing queries and indexes.",
    "restart_service:database": "Completing database recovery chain: DB optimised, now restarting database service.",
    "no_action": "System appears healthy. No further action required.",
}


def _generate_fallback_reasoning(action: str, state: dict) -> str:
    base = _REASONING_MAP.get(action, f"Executing '{action}' based on current system metrics.")
    cpu = state.get("cpu_usage", 0)
    mem = state.get("memory_usage", 0)
    db = state.get("db_latency", "low")
    status = state.get("status", "unknown")

    msg = base

    if action == "no_action" and status != "healthy":
        msg = f"Safety backup exhausted all options while system is {status.upper()}. Manual investigation required."
    elif action == "scale_up:cpu" and cpu > 80:
        msg = f"CPU at {cpu}% — severe overload. Scaling up CPU resources."
    elif action == "clear_cache" and mem > 70:
        msg = f"Memory at {mem}% — critically high. Clearing cache to free memory."
    elif action == "optimize_database" and db in ("medium", "high"):
        msg = f"Database latency is {db}. Optimising queries and indexes."
        
    return f'[Confidence: 10%] Intent: UNKNOWN | Root Cause: Safety backup triggered. | Plan: {msg}'


# ---------------------------------------------------------------------------
# Chat helper
# ---------------------------------------------------------------------------

def chat_diagnose(state: dict, actions_taken: list[str], user_message: str) -> dict:
    if client is None:
        return {
            "diagnosis": "LLM not configured.",
            "reasoning": "Set API_BASE_URL, MODEL_NAME, HF_TOKEN in .env",
            "suggested_action": "no_action",
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
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except Exception:
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

        res = env.reset(task_id)
        if isinstance(res, tuple):
            _, info = res[:2]
        else:
            info = getattr(res, "info", {})

        obs = DevOpsObservation(**env._sys_state.to_dict())
        done = False
        step = 0
        actions_history: list[str] = []

        while not done:
            step += 1
            state_dict = {
                "status": obs.status,
                "cpu_usage": obs.cpu_usage,
                "memory_usage": obs.memory_usage,
                "db_latency": obs.db_latency,
                "services": obs.services,
                "logs": obs.logs,
            }

            # Check healthy before LLM call — never waste a token
            if obs.status == "healthy":
                print(f"  Step {step:>2}: ✅ System healthy. Stopping.")
                break

            action_str = get_action(state_dict, actions_history)

            if action_str == "no_action" and obs.status != "healthy":
                print(f"  Step {step:>2}: ⚠ no_action on {obs.status} system — backup exhausted all options.")
            else:
                print(f"  Step {step:>2}: action = {action_str}")

            res = env.step(DevOpsAction(action_str=action_str))
            if isinstance(res, tuple):
                _, reward, term, trunc, info = res[:5]
                done = term or trunc
            else:
                reward = getattr(res, "reward", 0.0)
                done = getattr(res, "done", False)
                info = getattr(res, "info", {})

            obs = DevOpsObservation(**env._sys_state.to_dict())
            obs.message = getattr(res, "message", "") if not isinstance(res, tuple) else ""
            actions_history.append(action_str)
            print(
                f"           reward={reward:+.4f} | status={obs.status}"
                f" | cpu={obs.cpu_usage}% mem={obs.memory_usage}%"
                f" db={obs.db_latency} | {obs.message}"
            )

        final_obs_dict = {
            "status": obs.status,
            "cpu_usage": obs.cpu_usage,
            "memory_usage": obs.memory_usage,
            "db_latency": obs.db_latency,
            "services": obs.services,
            "logs": obs.logs,
        }
        score = grade(task_id, env.actions_taken, final_obs_dict)
        results[task_id] = score
        print(f"\n  ✅ Score: {score:.4f}")

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