"""
DevOps Incident Response – Task Definitions (v3)
==================================================
Provides both static preset tasks AND dynamic random task generation
with optional seed for reproducibility.

Static tasks: easy, medium, hard, expert
Dynamic generation: generate_task(difficulty, seed=None)
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Root cause pool
# ---------------------------------------------------------------------------

ALL_ROOT_CAUSES = ["cpu_overload", "memory_leak", "database_issue", "cache_failure"]

# Solution chains per root cause (used to build ground_truth)
SOLUTION_CHAINS: Dict[str, List[str]] = {
    "cpu_overload":   ["scale_up:cpu"],
    "memory_leak":    ["clear_cache", "restart_service:api"],
    "database_issue": ["optimize_database", "restart_service:database"],
    "cache_failure":  ["clear_cache"],
}

# Log snippets for dynamic generation
_VAGUE_LOGS = {
    "cpu_overload":   "System performance degraded. High resource consumption detected.",
    "memory_leak":    "System slowdown observed. Latency spikes in responses.",
    "database_issue": "Elevated query response times. Connection pool stressed.",
    "cache_failure":  "Intermittent cache misses. Fallback to direct DB queries.",
}

# ---------------------------------------------------------------------------
# Static task registry (preserved from v2)
# ---------------------------------------------------------------------------

TASKS: Dict[str, Dict[str, Any]] = {
    "easy": {
        "description": "CPU overload — investigate then scale up.",
        "initial_state": {
            "logs": (
                "ERROR 2024-01-15 03:12:44 api-gateway: "
                "Connection refused on port 8080. "
                "Service 'api' is not responding to health checks."
            ),
            "cpu_usage": 35,
            "memory_usage": 42,
            "db_latency": "low",
            "services": ["database", "cache"],
            "status": "degraded",
            "step_count": 0,
        },
        "root_causes": ["cpu_overload"],
        "partially_fixed": {},
        "ground_truth_solution_steps": ["check_logs", "scale_up:cpu"],
    },
    "medium": {
        "description": "CPU overload + database issue — multi-step fix chain.",
        "initial_state": {
            "logs": (
                "WARN 2024-01-15 04:30:01 monitor: System performance degraded. "
                "High resource consumption detected across multiple processes. "
                "WARN 2024-01-15 04:30:05 db-proxy: Elevated query response times."
            ),
            "cpu_usage": 92,
            "memory_usage": 55,
            "db_latency": "high",
            "services": ["api", "database", "cache"],
            "status": "degraded",
            "step_count": 0,
        },
        "root_causes": ["cpu_overload", "database_issue"],
        "partially_fixed": {},
        "ground_truth_solution_steps": [
            "check_logs", "scale_up:cpu",
            "optimize_database", "restart_service:database",
        ],
    },
    "hard": {
        "description": "Memory leak + database issue + cache failure — cascading failure.",
        "initial_state": {
            "logs": (
                "CRITICAL 2024-01-15 05:00:00 monitor: System slowdown observed. "
                "Unexpected latency spikes in application responses. "
                "ERROR 2024-01-15 05:00:02 db-proxy: Connection pool utilization "
                "above normal thresholds. "
                "ERROR 2024-01-15 05:00:03 cache-layer: Intermittent cache misses "
                "detected. Application fallback to direct database queries increasing."
            ),
            "cpu_usage": 88,
            "memory_usage": 82,
            "db_latency": "high",
            "services": ["api", "database"],
            "status": "down",
            "step_count": 0,
        },
        "root_causes": ["memory_leak", "database_issue", "cache_failure"],
        "partially_fixed": {},
        "ground_truth_solution_steps": [
            "check_logs", "clear_cache", "restart_service:api",
            "optimize_database", "restart_service:database",
        ],
    },
    "expert": {
        "description": "All 4 root causes active — total system meltdown.",
        "initial_state": {
            "logs": (
                "CRITICAL 2024-01-15 06:00:00 monitor: MULTIPLE SYSTEM ALERTS. "
                "Performance severely degraded across all subsystems. "
                "Investigate immediately."
            ),
            "cpu_usage": 97,
            "memory_usage": 91,
            "db_latency": "high",
            "services": ["database"],
            "status": "down",
            "step_count": 0,
        },
        "root_causes": ["cpu_overload", "memory_leak", "database_issue", "cache_failure"],
        "partially_fixed": {},
        "ground_truth_solution_steps": [
            "check_logs", "scale_up:cpu", "clear_cache",
            "restart_service:api", "optimize_database",
            "restart_service:database",
        ],
    },
}


# ---------------------------------------------------------------------------
# Dynamic task generation
# ---------------------------------------------------------------------------

def generate_task(
    difficulty: str = "medium",
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Generate a randomised task with the given difficulty.

    Parameters
    ----------
    difficulty : str
        One of 'easy', 'medium', 'hard', 'expert', or 'random'.
    seed : int, optional
        If provided, ensures deterministic generation for reproducibility.

    Returns
    -------
    dict
        A task definition compatible with TASKS format.
    """
    rng = random.Random(seed)

    # Determine number of root causes by difficulty
    cause_counts = {
        "easy": 1, "medium": 2, "hard": 3, "expert": 4, "random": None,
    }
    n = cause_counts.get(difficulty)
    if n is None:
        n = rng.randint(1, 4)

    # Select root causes
    causes = rng.sample(ALL_ROOT_CAUSES, min(n, len(ALL_ROOT_CAUSES)))

    # Generate initial metrics based on which root causes are active
    cpu = rng.randint(25, 45)
    mem = rng.randint(35, 50)
    db_lat = "low"
    services = ["api", "database", "cache"]

    if "cpu_overload" in causes:
        cpu = rng.randint(82, 99)
    if "memory_leak" in causes:
        mem = rng.randint(72, 95)
    if "database_issue" in causes:
        db_lat = rng.choice(["medium", "high"])
    if "cache_failure" in causes:
        if "cache" in services:
            services.remove("cache")

    # Determine overall status
    if len(causes) >= 3 or cpu >= 95 or mem >= 90:
        status = "down"
    elif len(causes) >= 2 or cpu >= 80 or mem >= 75:
        status = "degraded"
    else:
        status = "degraded"

    # Build vague initial logs
    ts = "2024-01-15 {:02d}:{:02d}:{:02d}".format(
        rng.randint(0, 23), rng.randint(0, 59), rng.randint(0, 59)
    )
    log_parts = [f"WARN {ts} monitor: {_VAGUE_LOGS[c]}" for c in causes]
    logs = " | ".join(log_parts)

    # Build ground truth solution (check_logs first, then fix chains)
    gt_steps = ["check_logs"]
    seen_actions: set = set()
    for cause in causes:
        for action in SOLUTION_CHAINS[cause]:
            if action not in seen_actions:
                gt_steps.append(action)
                seen_actions.add(action)

    task = {
        "description": f"Dynamic {difficulty} task with {len(causes)} root cause(s): {', '.join(causes)}.",
        "initial_state": {
            "logs": logs,
            "cpu_usage": cpu,
            "memory_usage": mem,
            "db_latency": db_lat,
            "services": services,
            "status": status,
            "step_count": 0,
        },
        "root_causes": causes,
        "partially_fixed": {},
        "ground_truth_solution_steps": gt_steps,
    }

    return task


def register_dynamic_task(
    task_id: str,
    difficulty: str = "medium",
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Generate a dynamic task and register it in the TASKS registry."""
    task = generate_task(difficulty, seed)
    TASKS[task_id] = task
    return task


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_task_ids() -> List[str]:
    """Return sorted list of available task identifiers."""
    return sorted(TASKS.keys())


def get_task(task_id: str) -> Dict[str, Any]:
    """Return the full task definition or raise ValueError."""
    if task_id not in TASKS:
        raise ValueError(
            f"Unknown task_id '{task_id}'. Available: {get_task_ids()}"
        )
    return TASKS[task_id]
