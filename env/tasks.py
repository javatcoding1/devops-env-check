"""
DevOps Incident Response – Task Definitions (v4)
==================================================
v4 changes:
  • Static tasks (easy/medium/hard/expert) now generate dynamic initial states
    on every call to get_task(). CPU, memory, db_latency, and services vary
    within difficulty-appropriate ranges while keeping the same root causes and
    ground-truth solution chains. No two runs see identical numbers.
  • Added get_task_initial_state(task_id, seed=None) for reproducible resets.
  • Dynamic generation (generate_task / register_dynamic_task) unchanged.
  • Backwards-compatible: TASKS still holds static base definitions; callers
    that read TASKS directly see the canonical base state, not a dynamic one.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Root cause pool
# ---------------------------------------------------------------------------

ALL_ROOT_CAUSES = ["cpu_overload", "memory_leak", "database_issue", "cache_failure"]

SOLUTION_CHAINS: Dict[str, List[str]] = {
    "cpu_overload":   ["scale_up:cpu"],
    "memory_leak":    ["clear_cache", "restart_service:api"],
    "database_issue": ["optimize_database", "restart_service:database"],
    "cache_failure":  ["clear_cache"],
}

_VAGUE_LOGS = {
    "cpu_overload":   "System performance degraded. High resource consumption detected.",
    "memory_leak":    "System slowdown observed. Latency spikes in responses.",
    "database_issue": "Elevated query response times. Connection pool stressed.",
    "cache_failure":  "Intermittent cache misses. Fallback to direct DB queries.",
}

# ---------------------------------------------------------------------------
# Log template pools — picked randomly so logs feel different each run
# ---------------------------------------------------------------------------

_LOG_TEMPLATES: Dict[str, List[str]] = {
    "cpu_overload": [
        "[{ts}] [ERROR] [api-gateway] Connection refused on port 8080. Service 'api' is not responding to health checks.",
        "[{ts}] [WARN] [monitor] High resource consumption detected across multiple processes.",
        "[{ts}] [ERROR] [scheduler] CPU throttling active. Task queue backed up.",
        "[{ts}] [WARN] [monitor] System performance degraded. CPU saturation approaching critical threshold.",
    ],
    "memory_leak": [
        "[{ts}] [CRITICAL] [monitor] System slowdown observed. Unexpected latency spikes in application responses.",
        "[{ts}] [ERROR] [api-service] Memory allocation failed. Heap exhausted.",
        "[{ts}] [WARN] [runtime] GC pressure high. Memory usage climbing steadily.",
        "[{ts}] [ERROR] [monitor] Memory leak suspected. Resident set size growing unbounded.",
    ],
    "database_issue": [
        "[{ts}] [ERROR] [db-proxy] Connection pool utilization above normal thresholds.",
        "[{ts}] [WARN] [db-proxy] Elevated query response times. Missing indexes suspected.",
        "[{ts}] [ERROR] [db-layer] Latency spikes detected. Query planner selecting full table scans.",
        "[{ts}] [WARN] [monitor] Database throughput degraded. Slow query log filling rapidly.",
    ],
    "cache_failure": [
        "[{ts}] [ERROR] [cache-layer] Intermittent cache misses detected. Application fallback to direct database queries increasing.",
        "[{ts}] [WARN] [cache] Cache hit rate dropped below 20%. High miss rate causing DB overload.",
        "[{ts}] [ERROR] [cache-service] Cache eviction storm. All hot keys expired simultaneously.",
        "[{ts}] [WARN] [monitor] Cache failure detected. Service degrading to uncached path.",
    ],
}

# ---------------------------------------------------------------------------
# Metric ranges per root cause (used for dynamic initial state generation)
# ---------------------------------------------------------------------------

# Each cause pushes certain metrics into elevated ranges
_METRIC_RANGES: Dict[str, Dict[str, Any]] = {
    "cpu_overload": {
        "cpu_min": 80, "cpu_max": 99,
        "mem_boost": 0,           # no extra memory pressure from CPU alone
    },
    "memory_leak": {
        "mem_min": 72, "mem_max": 97,
        "cpu_boost": 8,           # memory pressure causes some CPU overhead
    },
    "database_issue": {
        "db_latency_choices": ["medium", "high"],
    },
    "cache_failure": {
        "removes_service": "cache",
        "mem_boost": 6,           # cache miss → more DB calls → slight mem rise
    },
}

# ---------------------------------------------------------------------------
# Static task registry — defines root causes and solution chains only.
# Initial state is generated dynamically by get_task() on every call.
# ---------------------------------------------------------------------------

TASKS: Dict[str, Dict[str, Any]] = {
    "easy": {
        "description": "CPU overload — investigate then scale up.",
        "root_causes": ["cpu_overload"],
        "partially_fixed": {},
        "ground_truth_solution_steps": ["check_logs", "scale_up:cpu"],
        # _base_state used as fallback only; real initial state is always dynamic
        "_base_state": {
            "cpu_usage": 35, "memory_usage": 42, "db_latency": "low",
            "services": ["database", "cache"], "status": "degraded",
        },
    },
    "medium": {
        "description": "CPU overload + database issue — multi-step fix chain.",
        "root_causes": ["cpu_overload", "database_issue"],
        "partially_fixed": {},
        "ground_truth_solution_steps": [
            "check_logs", "scale_up:cpu",
            "optimize_database", "restart_service:database",
        ],
        "_base_state": {
            "cpu_usage": 92, "memory_usage": 55, "db_latency": "high",
            "services": ["api", "database", "cache"], "status": "degraded",
        },
    },
    "hard": {
        "description": "Memory leak + database issue + cache failure — cascading failure.",
        "root_causes": ["memory_leak", "database_issue", "cache_failure"],
        "partially_fixed": {},
        "ground_truth_solution_steps": [
            "check_logs", "clear_cache", "restart_service:api",
            "optimize_database", "restart_service:database",
        ],
        "_base_state": {
            "cpu_usage": 88, "memory_usage": 82, "db_latency": "high",
            "services": ["api", "database"], "status": "degraded",
        },
    },
    "expert": {
        "description": "All 4 root causes active — total system meltdown.",
        "root_causes": ["cpu_overload", "memory_leak", "database_issue", "cache_failure"],
        "partially_fixed": {},
        "ground_truth_solution_steps": [
            "check_logs", "scale_up:cpu", "clear_cache",
            "restart_service:api", "optimize_database",
            "restart_service:database",
        ],
        "_base_state": {
            "cpu_usage": 97, "memory_usage": 91, "db_latency": "high",
            "services": ["database"], "status": "down",
        },
    },
}


# ---------------------------------------------------------------------------
# Dynamic initial state builder — used for ALL tasks on every reset
# ---------------------------------------------------------------------------

def _build_initial_state(
    root_causes: List[str],
    rng: random.Random,
) -> Dict[str, Any]:
    """
    Generate a randomised but difficulty-consistent initial state for a task
    given its list of root causes.

    Rules:
    - cpu_overload  → cpu in [80, 99], random noise on mem
    - memory_leak   → mem in [72, 97], small cpu boost
    - database_issue → db_latency random in [medium, high]
    - cache_failure  → removes 'cache' from services, small mem boost
    - Baseline cpu/mem (no relevant cause) drawn from [25, 55]
    - Status: 'down' if ≥3 causes or cpu≥95 or mem≥90; else 'degraded'
    """
    # Start with baseline noise
    cpu = rng.randint(25, 55)
    mem = rng.randint(30, 55)
    db_latency = "low"
    services = ["api", "database", "cache"]

    if "cpu_overload" in root_causes:
        cpu = rng.randint(
            _METRIC_RANGES["cpu_overload"]["cpu_min"],
            _METRIC_RANGES["cpu_overload"]["cpu_max"],
        )

    if "memory_leak" in root_causes:
        mem = rng.randint(
            _METRIC_RANGES["memory_leak"]["mem_min"],
            _METRIC_RANGES["memory_leak"]["mem_max"],
        )
        cpu = min(99, cpu + rng.randint(0, _METRIC_RANGES["memory_leak"]["cpu_boost"]))

    if "database_issue" in root_causes:
        db_latency = rng.choice(_METRIC_RANGES["database_issue"]["db_latency_choices"])

    if "cache_failure" in root_causes:
        svc = _METRIC_RANGES["cache_failure"]["removes_service"]
        if svc in services:
            services.remove(svc)
        mem = min(99, mem + rng.randint(0, _METRIC_RANGES["cache_failure"]["mem_boost"]))

    # When cpu_overload is NOT present but multiple other causes raise pressure,
    # still add some CPU noise so metrics don't look artificially clean
    if "cpu_overload" not in root_causes and len(root_causes) >= 2:
        cpu = min(75, cpu + rng.randint(5, 20))

    # Status determination
    if len(root_causes) >= 3 or cpu >= 95 or mem >= 90:
        status = "down"
    else:
        status = "degraded"

    # Build log string — pick one template per cause and randomise timestamp
    log_parts = []
    for cause in root_causes:
        ts = "{:04d}-{:02d}-{:02d} {:02d}:{:02d}:{:02d}".format(
            2024,
            rng.randint(1, 12),
            rng.randint(1, 28),
            rng.randint(0, 23),
            rng.randint(0, 59),
            rng.randint(0, 59),
        )
        template = rng.choice(_LOG_TEMPLATES[cause])
        log_parts.append(template.format(ts=ts))

    logs = " ".join(log_parts)

    return {
        "logs": logs,
        "cpu_usage": cpu,
        "memory_usage": mem,
        "db_latency": db_latency,
        "services": services,
        "status": status,
        "step_count": 0,
    }


# ---------------------------------------------------------------------------
# Dynamic task generation (for /generate endpoint — unchanged from v3)
# ---------------------------------------------------------------------------

def generate_task(
    difficulty: str = "medium",
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    rng = random.Random(seed)

    cause_counts = {
        "easy": 1, "medium": 2, "hard": 3, "expert": 4, "random": None,
    }
    n = cause_counts.get(difficulty)
    if n is None:
        n = rng.randint(1, 4)

    causes = rng.sample(ALL_ROOT_CAUSES, min(n, len(ALL_ROOT_CAUSES)))
    initial_state = _build_initial_state(causes, rng)

    gt_steps = ["check_logs"]
    seen_actions: set = set()
    for cause in causes:
        for action in SOLUTION_CHAINS[cause]:
            if action not in seen_actions:
                gt_steps.append(action)
                seen_actions.add(action)

    return {
        "description": f"Dynamic {difficulty} task with {len(causes)} root cause(s): {', '.join(causes)}.",
        "initial_state": initial_state,
        "root_causes": causes,
        "partially_fixed": {},
        "ground_truth_solution_steps": gt_steps,
    }


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


def get_tasks_detailed() -> List[Dict[str, str]]:
    return [
        {
            "task_id": tid,
            "difficulty": tid if tid in ["easy", "medium", "hard", "expert"] else "custom",
        }
        for tid in sorted(TASKS.keys())
    ]


def get_task(task_id: str) -> Dict[str, Any]:
    """
    Return the full task definition.
    For static tasks, initial_state is generated fresh on every call
    so no two resets see identical metric numbers.
    """
    if task_id not in TASKS:
        raise ValueError(
            f"Unknown task_id '{task_id}'. Available: {get_task_ids()}"
        )
    task = TASKS[task_id]

    # Dynamic tasks already have a pre-built initial_state from generate_task()
    if "initial_state" in task:
        return task

    # Static tasks: build a fresh dynamic initial state each call
    rng = random.Random()   # unseeded → different every call
    initial_state = _build_initial_state(task["root_causes"], rng)

    return {
        "description": task["description"],
        "root_causes": task["root_causes"],
        "partially_fixed": task.get("partially_fixed", {}),
        "ground_truth_solution_steps": task["ground_truth_solution_steps"],
        "initial_state": initial_state,
    }


def get_task_initial_state(task_id: str, seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Return only the initial state for a task, with optional seed for
    reproducible resets (useful for evaluation / benchmarking).
    """
    if task_id not in TASKS:
        raise ValueError(f"Unknown task_id '{task_id}'. Available: {get_task_ids()}")
    task = TASKS[task_id]
    if "initial_state" in task:
        # Already a dynamic task with a fixed initial state
        return task["initial_state"]
    rng = random.Random(seed)
    return _build_initial_state(task["root_causes"], rng)