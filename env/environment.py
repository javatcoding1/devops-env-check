"""
DevOps Incident Response Environment (v2 — Production-Grade)
=============================================================
A highly realistic RL-compatible environment simulating DevOps system
failures with:

  • Hidden root causes (not directly visible to agent)
  • Dynamic, evolving log messages
  • Failure propagation (wrong actions worsen the system)
  • Multi-step action dependencies
  • Continuous state evolution every step

Supports reset(), step(), and state() for OpenEnv compatibility.
"""

from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_ACTIONS: List[str] = [
    "restart_service:api",
    "restart_service:database",
    "scale_up:cpu",
    "optimize_database",
    "clear_cache",
    "check_logs",
    "do_nothing",
]

MAX_STEPS = 15
STEP_PENALTY = -0.05

# Root cause categories (hidden from agent)
ROOT_CAUSES = [
    "cpu_overload",
    "memory_leak",
    "database_issue",
    "cache_failure",
]


# ---------------------------------------------------------------------------
# Observable state (what the agent sees)
# ---------------------------------------------------------------------------

@dataclass
class SystemState:
    logs: str = ""
    cpu_usage: int = 30
    memory_usage: int = 40
    db_latency: str = "low"
    services: List[str] = field(default_factory=lambda: ["api", "database", "cache"])
    status: str = "healthy"
    step_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SystemState":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Hidden internal state (NOT exposed to agent)
# ---------------------------------------------------------------------------

@dataclass
class HiddenState:
    """Tracks the true root causes and internal dynamics."""
    root_causes: List[str] = field(default_factory=list)
    resolved_causes: List[str] = field(default_factory=list)
    log_depth: int = 0          # how many check_logs have been done
    degradation_counter: int = 0  # accumulates from wrong actions
    wrong_action_streak: int = 0  # consecutive wrong actions
    partially_fixed: Dict[str, bool] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Dynamic log generator
# ---------------------------------------------------------------------------

class LogGenerator:
    """Generates evolving log messages based on root causes and investigation depth."""

    # Initial (vague) logs per root cause
    INITIAL_LOGS: Dict[str, str] = {
        "cpu_overload": (
            "WARN {ts} monitor: System performance degraded. "
            "High resource consumption detected across multiple processes."
        ),
        "memory_leak": (
            "WARN {ts} monitor: System slowdown observed. "
            "Unexpected latency spikes in application responses."
        ),
        "database_issue": (
            "WARN {ts} db-proxy: Elevated query response times. "
            "Connection pool utilization above normal thresholds."
        ),
        "cache_failure": (
            "WARN {ts} cache-layer: Intermittent cache misses detected. "
            "Application fallback to direct database queries increasing."
        ),
    }

    # After first check_logs — more specific hints
    DEPTH_1_LOGS: Dict[str, str] = {
        "cpu_overload": (
            "INFO {ts} profiler: CPU utilization breakdown — "
            "worker processes consuming 85%+ of available cores. "
            "Context switches elevated. Possible CPU saturation."
        ),
        "memory_leak": (
            "INFO {ts} profiler: Memory usage trending upward — "
            "heap allocations not being freed. RSS grew 200MB in last hour. "
            "Suspect memory leak in application layer."
        ),
        "database_issue": (
            "INFO {ts} db-proxy: Slow query log analysis — "
            "full table scans on 'orders' and 'sessions' tables. "
            "Missing indexes suspected. Lock contention observed."
        ),
        "cache_failure": (
            "INFO {ts} cache-layer: Cache health check — "
            "checksum mismatches on 12% of cached entries. "
            "Eviction policy failing. Cache data may be corrupt."
        ),
    }

    # After second check_logs — reveals root cause clearly
    DEPTH_2_LOGS: Dict[str, str] = {
        "cpu_overload": (
            "CRITICAL {ts} diagnostics: ROOT CAUSE IDENTIFIED — "
            "CPU overload confirmed. Runaway worker threads exhausting "
            "all available CPU cores. Immediate scale-up required."
        ),
        "memory_leak": (
            "CRITICAL {ts} diagnostics: ROOT CAUSE IDENTIFIED — "
            "Memory leak confirmed in request handler. Object finalizers "
            "not releasing connections. Clear cache and restart service required."
        ),
        "database_issue": (
            "CRITICAL {ts} diagnostics: ROOT CAUSE IDENTIFIED — "
            "Database performance degradation from unoptimized queries "
            "and stale connection pool. Optimize database and restart required."
        ),
        "cache_failure": (
            "CRITICAL {ts} diagnostics: ROOT CAUSE IDENTIFIED — "
            "Cache corruption from concurrent write race condition. "
            "Full cache clear and service restart required."
        ),
    }

    @classmethod
    def generate(cls, root_causes: List[str], depth: int, resolved: List[str]) -> str:
        ts = "2024-01-15 05:{:02d}:{:02d}".format(
            min(depth * 3, 59), min(depth * 7 % 60, 59)
        )
        active = [rc for rc in root_causes if rc not in resolved]

        if not active:
            return f"INFO {ts} monitor: All systems nominal. No active issues detected."

        parts = []
        for rc in active:
            if depth == 0:
                template = cls.INITIAL_LOGS.get(rc, "")
            elif depth == 1:
                template = cls.DEPTH_1_LOGS.get(rc, cls.INITIAL_LOGS.get(rc, ""))
            else:
                template = cls.DEPTH_2_LOGS.get(rc, cls.DEPTH_1_LOGS.get(rc, ""))
            parts.append(template.format(ts=ts))

        return " | ".join(parts)


# ---------------------------------------------------------------------------
# Multi-step solution definitions
# ---------------------------------------------------------------------------

# What sequence of actions resolves each root cause
SOLUTION_CHAINS: Dict[str, List[str]] = {
    "cpu_overload": ["scale_up:cpu"],
    "memory_leak": ["clear_cache", "restart_service:api"],
    "database_issue": ["optimize_database", "restart_service:database"],
    "cache_failure": ["clear_cache"],
}

# What actions are harmful given a root cause (will make things worse)
HARMFUL_ACTIONS: Dict[str, List[str]] = {
    "cpu_overload": ["restart_service:database", "optimize_database"],
    "memory_leak": ["scale_up:cpu", "restart_service:database"],
    "database_issue": ["restart_service:api", "clear_cache"],
    "cache_failure": ["restart_service:database", "scale_up:cpu"],
}


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class DevOpsEnv:
    """OpenEnv-compatible DevOps Incident Response Environment (v2)."""

    def __init__(self) -> None:
        self._state: SystemState = SystemState()
        self._hidden: HiddenState = HiddenState()
        self._task_id: Optional[str] = None
        self._done: bool = False
        self._actions_taken: List[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, task_id: Optional[str] = None) -> Dict[str, Any]:
        """Reset the environment, optionally loading a specific task."""
        from env.tasks import TASKS

        self._task_id = task_id or "easy"
        if self._task_id not in TASKS:
            raise ValueError(
                f"Unknown task_id '{self._task_id}'. "
                f"Available: {list(TASKS.keys())}"
            )

        task = TASKS[self._task_id]
        initial = task["initial_state"]

        self._state = SystemState.from_dict(copy.deepcopy(initial))
        self._hidden = HiddenState(
            root_causes=copy.deepcopy(task["root_causes"]),
            resolved_causes=[],
            log_depth=0,
            degradation_counter=0,
            wrong_action_streak=0,
            partially_fixed=copy.deepcopy(task.get("partially_fixed", {})),
        )
        self._done = False
        self._actions_taken = []

        return self._state.to_dict()

    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Execute *action* and return (observation, reward, done, info)."""
        if self._done:
            return (
                self._state.to_dict(),
                0.0,
                True,
                {"message": "Episode already finished. Call reset()."},
            )

        action = action.strip().lower()
        if action not in VALID_ACTIONS:
            return (
                self._state.to_dict(),
                -0.2,
                False,
                {"message": f"Invalid action '{action}'."},
            )

        self._state.step_count += 1
        self._actions_taken.append(action)

        reward = self._apply_action(action)
        self._evolve_state()             # dynamic state evolution each step
        reward += STEP_PENALTY            # per-step penalty

        # Check termination
        if self._state.status == "healthy":
            self._done = True
            info = {"message": "System restored to healthy. Episode complete."}
        elif self._state.step_count >= MAX_STEPS:
            self._done = True
            info = {"message": "Max steps reached. Episode terminated."}
        else:
            info = {"message": f"Action '{action}' applied."}

        return self._state.to_dict(), round(reward, 4), self._done, info

    def state(self) -> Dict[str, Any]:
        """Return the current observation (hidden state is NOT included)."""
        return self._state.to_dict()

    @property
    def done(self) -> bool:
        return self._done

    @property
    def actions_taken(self) -> List[str]:
        return list(self._actions_taken)

    # ------------------------------------------------------------------
    # Core action logic
    # ------------------------------------------------------------------

    def _apply_action(self, action: str) -> float:
        """Mutate state and return the immediate reward (before step penalty)."""
        s = self._state
        h = self._hidden
        reward = 0.0

        # ── CHECK LOGS ─────────────────────────────────────────────
        if action == "check_logs":
            h.log_depth += 1
            s.logs = LogGenerator.generate(
                h.root_causes, h.log_depth, h.resolved_causes
            )
            h.wrong_action_streak = 0
            # Reward for investigating — more reward on first check
            if h.log_depth == 1:
                reward = 0.5
            elif h.log_depth == 2:
                reward = 0.3
            else:
                reward = 0.1  # diminishing returns

        # ── SCALE UP CPU ───────────────────────────────────────────
        elif action == "scale_up:cpu":
            if "cpu_overload" in h.root_causes and "cpu_overload" not in h.resolved_causes:
                h.resolved_causes.append("cpu_overload")
                s.cpu_usage = max(20, s.cpu_usage - 65)
                s.logs = (
                    f"INFO monitor: CPU scaled up successfully. "
                    f"Usage dropped to {s.cpu_usage}%. Workers rebalanced."
                )
                h.wrong_action_streak = 0
                reward = 1.0
            elif s.cpu_usage > 70:
                s.cpu_usage = max(30, s.cpu_usage - 25)
                s.logs = f"INFO monitor: Partial CPU relief. Usage now {s.cpu_usage}%."
                reward = 0.3
            else:
                reward = self._apply_wrong_action(
                    "CPU usage is normal. Scaling up wastes resources and causes brief instability."
                )

        # ── OPTIMIZE DATABASE ──────────────────────────────────────
        elif action == "optimize_database":
            if "database_issue" in h.root_causes and "database_issue" not in h.resolved_causes:
                # Multi-step: mark as partially fixed (still needs restart)
                h.partially_fixed["database_issue"] = True
                s.db_latency = "low"
                s.logs = (
                    "INFO db-proxy: Query optimizer rebuilt indexes. "
                    "Latency reduced. Connection pool still needs refresh — "
                    "consider restart_service:database to complete the fix."
                )
                h.wrong_action_streak = 0
                reward = 0.7  # partial reward, full fix needs restart
            elif s.db_latency in ("medium", "high"):
                s.db_latency = "low"
                s.logs = "INFO db-proxy: Minor optimization applied."
                reward = 0.3
            else:
                reward = self._apply_wrong_action(
                    "Database latency is already low. Unnecessary optimization causes brief lock contention."
                )

        # ── RESTART SERVICE: DATABASE ──────────────────────────────
        elif action == "restart_service:database":
            if (
                "database_issue" in h.root_causes
                and "database_issue" not in h.resolved_causes
                and h.partially_fixed.get("database_issue")
            ):
                # Completing the multi-step fix
                h.resolved_causes.append("database_issue")
                h.partially_fixed["database_issue"] = False
                s.logs = (
                    "INFO db-proxy: Database restarted. Connection pool refreshed. "
                    "All queries executing within normal latency bounds."
                )
                if "database" not in s.services:
                    s.services.append("database")
                h.wrong_action_streak = 0
                reward = 1.0
            elif "database" not in s.services:
                s.services.append("database")
                s.logs = "INFO monitor: Database service restarted from DOWN state."
                reward = 0.5
            elif (
                "database_issue" in h.root_causes
                and not h.partially_fixed.get("database_issue")
            ):
                # Restart without optimizing first — partial help at best
                s.logs = (
                    "WARN db-proxy: Database restarted but underlying query "
                    "performance issues persist. Consider optimize_database first."
                )
                s.db_latency = "medium"  # slightly improved but not fixed
                reward = 0.1
            else:
                reward = self._apply_wrong_action(
                    "Database service is running fine. Unnecessary restart causes brief downtime."
                )

        # ── RESTART SERVICE: API ───────────────────────────────────
        elif action == "restart_service:api":
            if (
                "memory_leak" in h.root_causes
                and "memory_leak" not in h.resolved_causes
                and h.partially_fixed.get("memory_leak")
            ):
                # Completing the multi-step fix for memory leak
                h.resolved_causes.append("memory_leak")
                h.partially_fixed["memory_leak"] = False
                s.memory_usage = 35
                s.logs = (
                    "INFO monitor: API service restarted with clean memory. "
                    "Memory leak eliminated. Heap usage stable at 35%."
                )
                if "api" not in s.services:
                    s.services.append("api")
                h.wrong_action_streak = 0
                reward = 1.0
            elif "api" not in s.services:
                s.services.append("api")
                s.logs = "INFO monitor: API service restarted from DOWN state."
                if "memory_leak" in h.root_causes and "memory_leak" not in h.resolved_causes:
                    s.logs += " Warning: memory leak may recur without cache clear."
                    s.memory_usage = max(55, s.memory_usage - 15)
                    reward = 0.3
                else:
                    reward = 0.5
                h.wrong_action_streak = 0
            else:
                reward = self._apply_wrong_action(
                    "API service is already running. Unnecessary restart causes connection drops."
                )

        # ── CLEAR CACHE ────────────────────────────────────────────
        elif action == "clear_cache":
            if "cache_failure" in h.root_causes and "cache_failure" not in h.resolved_causes:
                h.resolved_causes.append("cache_failure")
                log_parts = [
                    "INFO cache-layer: Cache fully cleared and rebuilt. "
                    "Checksums validated. Hit rate recovering."
                ]
                if "cache" not in s.services:
                    s.services.append("cache")
                h.wrong_action_streak = 0
                reward = 1.0
                # Also handles step 1 of memory_leak fix if both are active
                if (
                    "memory_leak" in h.root_causes
                    and "memory_leak" not in h.resolved_causes
                ):
                    h.partially_fixed["memory_leak"] = True
                    s.memory_usage = max(40, s.memory_usage - 20)
                    log_parts.append(
                        "Memory pressure also reduced. However, leaked allocations "
                        "remain — restart_service:api recommended to fully resolve."
                    )
                s.logs = " ".join(log_parts)
            elif (
                "memory_leak" in h.root_causes
                and "memory_leak" not in h.resolved_causes
            ):
                # Multi-step: clear_cache is step 1 of memory_leak fix
                h.partially_fixed["memory_leak"] = True
                s.memory_usage = max(40, s.memory_usage - 20)
                s.logs = (
                    "INFO cache-layer: Cache cleared. Memory pressure reduced. "
                    "However, leaked allocations remain — restart_service:api "
                    "recommended to fully resolve memory issue."
                )
                if "cache" not in s.services:
                    s.services.append("cache")
                h.wrong_action_streak = 0
                reward = 0.7
            else:
                reward = self._apply_wrong_action(
                    "Cache is healthy. Clearing it unnecessarily hurts hit rates."
                )

        # ── DO NOTHING ─────────────────────────────────────────────
        elif action == "do_nothing":
            # Inaction lets the system degrade further
            active_count = len(
                [rc for rc in h.root_causes if rc not in h.resolved_causes]
            )
            if active_count > 0:
                s.cpu_usage = min(100, s.cpu_usage + 3)
                s.memory_usage = min(100, s.memory_usage + 2)
                s.logs = (
                    "WARN monitor: No action taken. System continuing to degrade. "
                    f"CPU now at {s.cpu_usage}%, memory at {s.memory_usage}%."
                )
                reward = -0.1
            else:
                s.logs = "INFO monitor: System stable. No action needed."
                reward = 0.0

        # ── Update aggregate status ────────────────────────────────
        self._recompute_status()

        return reward

    def _apply_wrong_action(self, message: str) -> float:
        """Handle an action that doesn't address any active issue — causes degradation."""
        s = self._state
        h = self._hidden

        h.wrong_action_streak += 1
        h.degradation_counter += 1

        # Failure propagation: wrong actions make things worse
        cpu_penalty = min(5, 2 * h.wrong_action_streak)
        mem_penalty = min(3, h.wrong_action_streak)

        s.cpu_usage = min(100, s.cpu_usage + cpu_penalty)
        s.memory_usage = min(100, s.memory_usage + mem_penalty)

        if h.wrong_action_streak >= 3 and s.db_latency == "low":
            s.db_latency = "medium"

        s.logs = (
            f"WARN monitor: {message} "
            f"System stress increased — CPU {s.cpu_usage}%, memory {s.memory_usage}%."
        )

        # Harsher penalty for consecutive wrong actions
        base_penalty = -0.2
        streak_penalty = -0.05 * h.wrong_action_streak
        return base_penalty + streak_penalty

    # ------------------------------------------------------------------
    # State evolution (runs every step, simulates real drift)
    # ------------------------------------------------------------------

    def _evolve_state(self) -> None:
        """
        Simulate natural system evolution each step:
        - Unresolved issues slowly worsen metrics
        - Partially fixed issues are semi-stable
        """
        s = self._state
        h = self._hidden
        active = [rc for rc in h.root_causes if rc not in h.resolved_causes]

        for rc in active:
            if rc == "cpu_overload" and not h.partially_fixed.get("cpu_overload"):
                s.cpu_usage = min(100, s.cpu_usage + 2)
            elif rc == "memory_leak" and not h.partially_fixed.get("memory_leak"):
                s.memory_usage = min(100, s.memory_usage + 3)
                if s.memory_usage > 85:
                    s.cpu_usage = min(100, s.cpu_usage + 1)  # memory pressure → CPU
            elif rc == "database_issue" and not h.partially_fixed.get("database_issue"):
                if s.db_latency == "low":
                    s.db_latency = "medium"
                elif s.db_latency == "medium":
                    s.db_latency = "high"
            elif rc == "cache_failure":
                if s.db_latency == "low":
                    s.db_latency = "medium"  # cache miss → DB hit

    def _recompute_status(self) -> None:
        """Derive overall system status from unresolved root causes and metrics."""
        h = self._hidden
        s = self._state
        active = [rc for rc in h.root_causes if rc not in h.resolved_causes]
        partially = sum(1 for v in h.partially_fixed.values() if v)

        # Count effective unresolved (partially fixed count as 0.5)
        effective_active = len(active) - (partially * 0.5)

        if effective_active <= 0 and s.cpu_usage < 70 and s.memory_usage < 80:
            s.status = "healthy"
        elif effective_active <= 1 or (s.cpu_usage < 85 and s.memory_usage < 90):
            s.status = "degraded"
        else:
            s.status = "down"

        # Override: if any critical metrics are extreme, force status
        if s.cpu_usage >= 95 or s.memory_usage >= 95:
            s.status = "down" if len(active) > 1 else "degraded"
