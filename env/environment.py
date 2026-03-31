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
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

try:
    import gymnasium as gym
    from gymnasium import spaces
    import numpy as np
    HAS_GYM = True
except ImportError:
    HAS_GYM = False

from openenv.core.env_server import Environment
from env.models import DevOpsAction, DevOpsObservation, DevOpsState

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
    "no_action",
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
    total_correct_actions: int = 0
    total_wrong_actions: int = 0


# ---------------------------------------------------------------------------
# Dynamic log generator
# ---------------------------------------------------------------------------

class LogGenerator:
    """Generates evolving log messages based on root causes and investigation depth."""

    # Initial (vague) logs per root cause
    INITIAL_LOGS: Dict[str, str] = {
        "cpu_overload": (
            "[{ts}] [WARN] [kernel] high context switching detected. "
            "System performance degraded due to task scheduler overhead."
        ),
        "memory_leak": (
            "[{ts}] [WARN] [runtime] GC pressure increasing. "
            "Minor GC frequency above 5s threshold. Transient slowdown observed."
        ),
        "database_issue": (
            "[{ts}] [WARN] [db-proxy] Elevated iowait on database volume. "
            "Disk read latency spiking in application responses."
        ),
        "cache_failure": (
            "[{ts}] [WARN] [cache-layer] cache_miss_rate > 40%. "
            "High eviction count detected in cache cluster metrics."
        ),
    }

    # After first check_logs — more specific hints
    DEPTH_1_LOGS: Dict[str, str] = {
        "cpu_overload": (
            "[{ts}] [INFO] [profiler] CPU profile analysis — "
            "kernel kworker threads consuming 85%+ of available cycles. "
            "Interrupt frequency spiking. Possible CPU saturation."
        ),
        "memory_leak": (
            "[{ts}] [INFO] [profiler] Memory telemetry — "
            "heap allocations trending upward. RSS grew 200MB in last hour. "
            "Resident set size approaching container limits."
        ),
        "database_issue": (
            "[{ts}] [INFO] [db-proxy] Slow query log — "
            "sequential scans detected on 'orders' and 'sessions' tables. "
            "Lock contention on buffer cache suspect."
        ),
        "cache_failure": (
            "[{ts}] [INFO] [cache-layer] Cluster heartbeats — "
            "checksum mismatches on 12% of cached keys. "
            "Network timeout during SET operation. Cache layer unstable."
        ),
    }

    # After second check_logs — reveals root cause clearly
    DEPTH_2_LOGS: Dict[str, str] = {
        "cpu_overload": (
            "[{ts}] [CRITICAL] [diagnostics] CPU throttling active. "
            "cgroup CPU quota limits reached. Runaway worker threads "
            "overflowing scheduler queue. Immediate resource expansion required."
        ),
        "memory_leak": (
            "[{ts}] [CRITICAL] [diagnostics] Memory saturation reached. "
            "OOM killer invoked on nearby processes. Heap object finalizers "
            "not releasing handles. Full rebuild of process layer required."
        ),
        "database_issue": (
            "[{ts}] [CRITICAL] [diagnostics] DB index bloat confirmed. "
            "Shared buffer cache hit rate < 60%. High I/O wait times. "
            "Index maintenance and pool refresh required."
        ),
        "cache_failure": (
            "[{ts}] [CRITICAL] [diagnostics] Cache segment corruption. "
            "Concurrent write race condition detected. Multiple LRU list "
            "segment faults. Flush and restart recommended."
        ),
    }

    @classmethod
    def generate(cls, root_causes: List[str], depth: int, resolved: List[str]) -> str:
        ts = "2024-01-15 05:{:02d}:{:02d}".format(
            min(depth * 3, 59), min(depth * 7 % 60, 59)
        )
        active = [rc for rc in root_causes if rc not in resolved]

        if not active:
            return f"[{ts}] [INFO] [monitor] All systems nominal. No active issues detected."

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

class DevOpsEnv(Environment, gym.Env if HAS_GYM else object):
    """OpenEnv-compatible DevOps Incident Response Environment (v2)."""

    def __init__(self) -> None:
        super().__init__()
        self._sys_state: SystemState = SystemState()
        self._hidden: HiddenState = HiddenState()
        self._task_id: Optional[str] = None
        self._done: bool = False
        self._actions_taken: List[str] = []
        self._episode_id = str(uuid.uuid4())
        self._score: float = 0.0
        self._done_reward_given = False

        if HAS_GYM:
            self.action_space = spaces.Discrete(len(VALID_ACTIONS))
            self.observation_space = spaces.Box(
                low=0.0, high=200.0, shape=(4,), dtype=np.float32
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, task_id: Optional[str] = None, seed: int | None = None, options: dict | None = None) -> Any:
        """Reset the environment, optionally loading a specific task."""
        if HAS_GYM:
            super().reset(seed=seed)
        from env.tasks import TASKS, get_task, get_task_ids

        self._task_id = task_id or "easy"
        if self._task_id not in TASKS:
            raise ValueError(
                f"Unknown task_id '{self._task_id}'. "
                f"Available: {list(TASKS.keys())}"
            )

        task = get_task(task_id)
        initial = task["initial_state"]

        self._state = SystemState.from_dict(copy.deepcopy(initial))
        self._sys_state = self._state
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
        self._episode_id = str(uuid.uuid4())
        self._score = 0.0

        obs = DevOpsObservation(**self._sys_state.to_dict(), message="Environment reset")
        obs.done = False
        obs.reward = 0.0
        obs.info = {
            "total_correct_actions": self._hidden.total_correct_actions,
            "total_wrong_actions": self._hidden.total_wrong_actions,
            "total_actions": self._sys_state.step_count,
        }
        
        if HAS_GYM:
            return self._obs_to_numpy(obs), {"task_id": self._task_id}
        return obs

    def _obs_to_numpy(self, obs: DevOpsObservation) -> Any:
        if not HAS_GYM: return None
        latency_map = {"low": 0.0, "medium": 1.0, "high": 2.0}
        return np.array([
            float(obs.cpu_usage),
            float(obs.memory_usage),
            latency_map.get(obs.db_latency, 0.0),
            float(obs.step_count)
        ], dtype=np.float32)

    def step(self, action: DevOpsAction | int) -> Any:
        """Execute a step. Accepts either Typed DevOpsAction or Discrete ID from RL."""
        if HAS_GYM and isinstance(action, (int, np.integer)):
            action_str = VALID_ACTIONS[int(action)]
            action_obj = DevOpsAction(action_str=action_str)
        else:
            action_obj = action

        obs = self._step_internal(action_obj)

        if HAS_GYM:
            reward = obs.reward or 0.0
            return self._obs_to_numpy(obs), reward, self._done, False, {"status": obs.status}
        
        return obs

    def _step_internal(self, action: DevOpsAction) -> DevOpsObservation:
        """Internal logic for step execution."""
        if self._done:
            msg = "Episode already finished. Call reset()."
            obs = DevOpsObservation(**self._sys_state.to_dict(), message=msg)
            return obs

        action_str = action.action_str.strip().lower()
        if action_str not in VALID_ACTIONS:
            msg = f"Invalid action '{action_str}'."
            obs = DevOpsObservation(**self._sys_state.to_dict(), message=msg)
            return obs

        self._sys_state.step_count += 1
        self._actions_taken.append(action_str)

        reward = self._apply_action(action_str)
        self._evolve_state()             # dynamic state evolution each step
        reward += STEP_PENALTY            # per-step penalty

        self._score += round(reward, 4)

        # Check termination
        if self._sys_state.status == "healthy":
            self._done = True
            msg = "System restored to healthy. Episode complete."
        elif self._sys_state.step_count >= MAX_STEPS:
            self._done = True
            msg = "Max steps reached. Episode terminated."
        else:
            msg = f"Action '{action_str}' applied."

        obs = DevOpsObservation(**self._sys_state.to_dict(), message=msg)
        obs.reward = round(reward, 4)
        obs.done = self._done
        
        # Inject tracking metrics
        obs.info = {
            "total_correct_actions": self._hidden.total_correct_actions,
            "total_wrong_actions": self._hidden.total_wrong_actions,
            "total_actions": self._sys_state.step_count,
        }
        
        # Inject episode summary at the end
        if self._done:
            obs.info["episode_summary"] = {
                "total_steps": self._sys_state.step_count,
                "total_reward": self._score,
                "final_status": self._sys_state.status
            }
            
        return obs

    @property
    def state(self) -> DevOpsState:
        """Return the episode state metadata."""
        return DevOpsState(episode_id=self._episode_id, step_count=self._sys_state.step_count)

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
        elif action == "no_action":
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
                h.total_wrong_actions += 1
            else:
                s.logs = "INFO monitor: System stable. No action needed."
                reward = 0.0
                h.total_correct_actions += 1

        # ── Update aggregate status ────────────────────────────────
        self._recompute_status()

        if reward > 0.0:
            h.total_correct_actions += 1

        return reward

    def _apply_wrong_action(self, message: str) -> float:
        """Handle an action that doesn't address any active issue — causes degradation."""
        s = self._state
        h = self._hidden

        h.wrong_action_streak += 1
        h.degradation_counter += 1
        h.total_wrong_actions += 1

        # Failure propagation: wrong actions make things worse
        cpu_penalty = min(5, 2 * h.wrong_action_streak)
        mem_penalty = min(3, h.wrong_action_streak)

        s.cpu_usage = min(100, s.cpu_usage + cpu_penalty)
        s.memory_usage = min(100, s.memory_usage + mem_penalty)

        if h.wrong_action_streak >= 3 and s.db_latency == "low":
            s.db_latency = "medium"

        s.logs = (
            f"[{s.step_count}] [WARN] [monitor] {message} "
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

        # A system is ONLY healthy if:
        # 1. No root causes remain unresolved
        # 2. All 3 core services (api, database, cache) are UP
        # 3. CPU usage is below 70% and Memory usage is below 80%
        is_fully_resolved = (len(active) == 0)
        has_all_services = (len(s.services) == 3)
        metrics_nominal = (s.cpu_usage < 70 and s.memory_usage < 80)

        if is_fully_resolved and has_all_services and metrics_nominal:
            s.status = "healthy"
        elif len(s.services) < 2 or s.cpu_usage >= 95 or s.memory_usage >= 95:
            s.status = "down"
        else:
            s.status = "degraded"

        # Override: if any critical metrics are extreme, force status
        if s.cpu_usage >= 95 or s.memory_usage >= 95:
            s.status = "down" if len(active) > 1 else "degraded"
