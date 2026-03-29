"""
DevOps Incident Response – Graders (v2 — Production-Grade)
============================================================
Deterministic, reproducible scoring that accounts for:
  • System outcome (healthy / degraded / down)
  • Root-cause coverage (did the agent address all causes?)
  • Action-match against ground truth
  • Efficiency (extra steps penalised)
  • Diagnostic bonus (did the agent investigate first?)
"""

from __future__ import annotations

from typing import Any, Dict, List

from env.tasks import TASKS


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def grade(task_id: str, actions_taken: List[str], final_state: Dict[str, Any]) -> float:
    """
    Score an episode.

    Parameters
    ----------
    task_id : str
        The task that was attempted.
    actions_taken : list[str]
        Ordered list of actions the agent executed.
    final_state : dict
        The observation dict returned after the final step.

    Returns
    -------
    float
        A score in [0.0, 1.0].
    """
    if task_id not in TASKS:
        raise ValueError(f"Unknown task_id '{task_id}'.")

    task = TASKS[task_id]
    gt_steps: List[str] = task["ground_truth_solution_steps"]

    # ── 1. Outcome score (35 %) ──────────────────────────────────────
    outcome = _outcome_score(final_state)

    # ── 2. Action-match score (25 %) ─────────────────────────────────
    action_match = _action_overlap_score(gt_steps, actions_taken)

    # ── 3. Efficiency score (15 %) ───────────────────────────────────
    efficiency = _efficiency_score(gt_steps, actions_taken)

    # ── 4. System health score (15 %) ────────────────────────────────
    health = _health_metrics_score(final_state)

    # ── 5. Diagnostic bonus (10 %) ───────────────────────────────────
    diagnostic = _diagnostic_score(actions_taken, task)

    raw = (
        0.35 * outcome
        + 0.25 * action_match
        + 0.15 * efficiency
        + 0.15 * health
        + 0.10 * diagnostic
    )
    return round(min(max(raw, 0.0), 1.0), 4)


# ---------------------------------------------------------------------------
# Sub-scores
# ---------------------------------------------------------------------------

def _outcome_score(final_state: Dict[str, Any]) -> float:
    """1.0 if healthy, 0.5 if degraded, 0.0 if down."""
    status = final_state.get("status", "down")
    if status == "healthy":
        return 1.0
    if status == "degraded":
        return 0.5
    return 0.0


def _action_overlap_score(ground_truth: List[str], taken: List[str]) -> float:
    """Fraction of ground-truth actions present in taken list (order-independent)."""
    if not ground_truth:
        return 1.0
    gt_set = set(ground_truth)
    taken_set = set(taken)
    matched = gt_set & taken_set
    return len(matched) / len(gt_set)


def _efficiency_score(ground_truth: List[str], taken: List[str]) -> float:
    """
    Penalise excess actions relative to the optimal solution length.
    Perfect efficiency → 1.0; degrades linearly with extra steps.
    """
    optimal = len(ground_truth)
    actual = len(taken)
    if actual == 0:
        return 0.0
    if actual <= optimal:
        return 1.0
    # Allow up to 2× optimal before score hits 0
    ratio = (actual - optimal) / max(optimal, 1)
    return max(1.0 - ratio, 0.0)


def _health_metrics_score(final_state: Dict[str, Any]) -> float:
    """
    Score based on final CPU, memory, and DB latency values.
    Lower resource usage and lower latency → higher score.
    """
    cpu = final_state.get("cpu_usage", 100)
    mem = final_state.get("memory_usage", 100)
    latency = final_state.get("db_latency", "high")

    # CPU score: 0-40 → 1.0, 40-70 → 0.7, 70-90 → 0.3, 90+ → 0.0
    if cpu <= 40:
        cpu_score = 1.0
    elif cpu <= 70:
        cpu_score = 0.7
    elif cpu <= 90:
        cpu_score = 0.3
    else:
        cpu_score = 0.0

    # Memory score: same brackets
    if mem <= 50:
        mem_score = 1.0
    elif mem <= 70:
        mem_score = 0.7
    elif mem <= 85:
        mem_score = 0.3
    else:
        mem_score = 0.0

    # Latency score
    latency_score = {"low": 1.0, "medium": 0.5, "high": 0.0}.get(latency, 0.0)

    return (cpu_score + mem_score + latency_score) / 3.0


def _diagnostic_score(actions_taken: List[str], task: dict) -> float:
    """
    Bonus for investigating before acting, especially on complex tasks.
    """
    root_causes = task.get("root_causes", [])
    complexity = len(root_causes)

    if complexity <= 1:
        # Simple task — investigation optional but good practice
        if "check_logs" in actions_taken:
            return 1.0
        return 0.5

    # Complex task — investigation is critical
    if "check_logs" not in actions_taken:
        return 0.0

    # Bonus if check_logs was done early (within first 2 actions)
    try:
        idx = actions_taken.index("check_logs")
        if idx == 0:
            return 1.0
        elif idx == 1:
            return 0.8
        else:
            return 0.5
    except ValueError:
        return 0.0
