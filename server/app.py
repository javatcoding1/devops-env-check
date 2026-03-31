"""
DevOps Incident Response – FastAPI Server (v5 / OpenEnv Core)
=============================================================
Synced with inference.py v5:
  • auto_run: removed manual do_nothing break (inference.py now owns that logic)
  • auto_run: strict 15-step cap matching environment MAX_STEPS
  • auto_run: obs updated correctly each step (logs, services, reward all captured)
  • Consolidated single __main__ entry point (removed duplicate run_server / main clash)
  • app.version / app.title bumped to v5
"""

from __future__ import annotations
import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from openenv.core.env_server import create_fastapi_app
from env.environment import DevOpsEnv, VALID_ACTIONS
from env.models import DevOpsAction, DevOpsObservation
from env.tasks import get_task_ids, get_tasks_detailed, register_dynamic_task

# ---------------------------------------------------------------------------
# Setup OpenEnv Server
# ---------------------------------------------------------------------------

MAX_STEPS = 15  # mirrors environment's hard limit

# Global environment instance
env_instance = DevOpsEnv()

# Factory returns our singleton so custom endpoints stay in sync.
app = create_fastapi_app(lambda: env_instance, DevOpsAction, DevOpsObservation)

app.title = "DevOps Incident Response Environment (v5 / OpenEnv)"
app.version = "5.0.0"
app.description = """
## 🤖 DevOps Incident Response (SRE) Simulation
A production-grade, OpenEnv-compliant reinforcement learning environment for evaluating AI agents.

### 📐 Technical Architecture
This platform is built on the **openenv-core** SDK and utilizes a **Dual-State Model**:
1.  **Observable State**: CPU (%), Memory (%), DB Latency, and Service Status (API/DB/Cache).
2.  **Hidden Root Causes**: Memory leaks, database locks, and CPU scaling bottlenecks.

### 🔍 Simulation Engine
- **Failure Propagation**: Metrics evolve dynamically based on hidden failures. A memory leak (`memory > 85%`) increases the objective chance of a service crash.
- **Multi-Step Chains**: Resolving 'Expert' tasks requires specific action sequences (e.g., investigating logs → clearing cache → restarting the API).
- **Log Generator**: A 3-depth log system that only reveals the true 'Root Cause' message after the agent executes `check_logs`.

### 📊 Scoring & Rewards
Episodes are graded out of **1.0** via a deterministic 5-component scoring model:
- **Outcome (35%)**: Reaching a 'healthy' status.
- **Logic Match (25%)**: Using the correct corrective action for the specific hidden cause.
- **Efficiency (15%)**: Minimizing unnecessary steps.
- **Health Bonus (15%)**: Maximizing resource optimization.
- **Diagnostic (10%)**: Rewarding log investigation before fixing.

### 🎮 Available Actions
- `restart_service:[api|database|cache]`
- `scale_up:cpu`
- `optimize_database`
- `clear_cache`
- `check_logs`
- `do_nothing`
"""

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Context tracking for chat endpoint
_actions_history: List[str] = []

# ---------------------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    task_id: str
    difficulty: str = "medium"
    seed: Optional[int] = None

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    diagnosis: str
    reasoning: str
    suggested_action: str

class GenerateResponse(BaseModel):
    task_id: str
    description: str
    difficulty: str
    root_causes: List[str]

class AutoRunRequest(BaseModel):
    task_id: Optional[str] = None

class ResetRequest(BaseModel):
    task_id: Optional[str] = None

class StepRequest(BaseModel):
    action: str

class AutoRunStepResult(BaseModel):
    step: int
    action: str
    reasoning: str
    reward: float
    status: str
    cpu_usage: int
    memory_usage: int
    db_latency: str
    message: str
    logs: str
    services: List[str]

class AutoRunResponse(BaseModel):
    steps: List[AutoRunStepResult]
    final_status: str
    total_steps: int
    total_reward: float

class HealthResponse(BaseModel):
    status: str = "ok"
    environment: str = "DevOps Incident Response Environment"
    version: str = "5.0.0"

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["Meta"])
async def root():
    return {
        "message": "DevOps AI Simulator v5 is running",
        "endpoints": ["/reset", "/step", "/state", "/auto-run", "/chat"],
    }

@app.get("/health", response_model=HealthResponse, tags=["Meta"])
async def health():
    return HealthResponse()

@app.get("/tasks", response_model=List[Dict[str, str]], tags=["Meta"])
async def list_tasks():
    return get_tasks_detailed()

@app.post("/reset", response_model=DevOpsObservation, tags=["Core"])
async def reset(body: ResetRequest = ResetRequest(), debug: bool = False):
    env_instance.reset(task_id=body.task_id)
    obs = DevOpsObservation(**env_instance._sys_state.to_dict())
    if debug:
        obs.debug = {"hidden_state": env_instance._hidden.__dict__}
    return obs

@app.post("/step", response_model=DevOpsObservation, tags=["Core"])
async def step(body: StepRequest, debug: bool = False):
    res = env_instance.step(DevOpsAction(action_str=body.action))
    if isinstance(res, tuple):
        _, reward, term, trunc, info = res[:5]
        obs = DevOpsObservation(**env_instance._sys_state.to_dict())
        obs.reward = reward
        obs.done = term or trunc
    else:
        obs = res
    if debug:
        obs.debug = {"hidden_state": env_instance._hidden.__dict__}
    return obs

@app.get("/actions", response_model=List[str], tags=["Meta"])
async def list_actions():
    return VALID_ACTIONS

@app.post("/chat", response_model=ChatResponse, tags=["AI Agent"])
async def chat_endpoint(body: ChatRequest):
    from inference import chat_diagnose
    state = env_instance.state.model_dump()
    result = chat_diagnose(state, _actions_history, body.message)
    return ChatResponse(**result)

@app.post("/generate", response_model=GenerateResponse, tags=["Tasks"])
async def generate_task_endpoint(body: GenerateRequest):
    task = register_dynamic_task(
        task_id=body.task_id,
        difficulty=body.difficulty,
        seed=body.seed,
    )
    return GenerateResponse(
        task_id=body.task_id,
        description=task["description"],
        difficulty=body.difficulty,
        root_causes=task["root_causes"],
    )

@app.post("/auto-run", response_model=AutoRunResponse, tags=["AI Agent"])
async def auto_run(body: AutoRunRequest = AutoRunRequest()):
    """
    Runs the inference.py v5 agent against the current environment state.

    Key behaviours (v5):
    - Always resets the env so the episode starts clean (mirrors run_inference).
    - task_id resolved from env_instance after reset — never a hardcoded fallback.
    - Healthy check fires BEFORE the LLM call, matching run_inference ordering.
    - grade() called at the end for a proper episodic score (not summed step rewards).
    - Strict MAX_STEPS=15 cap to prevent infinite loops.
    """
    from inference import get_action_with_reasoning
    from env.graders import grade
    global _actions_history

    # Fresh history for each auto-run session
    _actions_history = []

    # Always reset — mirrors run_inference() which calls env.reset(task_id) per task.
    env_instance.reset(task_id=body.task_id)

    # Resolve the actual task_id the environment is now running.
    # env_instance.current_task_id is set by reset(); fall back to the first known task
    # only if the attribute doesn't exist (older env builds).
    resolved_task_id: str = (
        getattr(env_instance, "current_task_id", None)
        or getattr(env_instance, "task_id", None)
        or (get_task_ids()[0] if get_task_ids() else None)
    )
    if resolved_task_id is None:
        raise HTTPException(status_code=500, detail="Could not resolve a valid task_id from the environment.")

    obs = DevOpsObservation(**env_instance._sys_state.to_dict())
    obs.message = "Auto-run v5 started"

    steps: List[AutoRunStepResult] = []
    total_reward = 0.0
    rewards_history: List[float] = []
    done = False
    step_count = 0

    while not done and step_count < MAX_STEPS:
        # Mirror run_inference: check healthy BEFORE the LLM call so we never
        # waste a token when the system is already recovered.
        if obs.status == "healthy":
            break

        step_count += 1

        current_state = {
            "status": obs.status,
            "cpu_usage": obs.cpu_usage,
            "memory_usage": obs.memory_usage,
            "db_latency": obs.db_latency,
            "services": obs.services,
            "logs": obs.logs,
        }

        # inference.py v5 is fully in control; do_nothing only returned when healthy.
        result = get_action_with_reasoning(current_state, _actions_history, rewards_history)
        action: str = result["action"]
        reasoning: str = result["reasoning"]

        # Execute the action in the environment
        res = env_instance.step(DevOpsAction(action_str=action))

        if isinstance(res, tuple):
            _, reward, term, trunc, info = res[:5]
            done = term or trunc
            obs = DevOpsObservation(**env_instance._sys_state.to_dict())
            # Read message from info dict when env returns a gym tuple
            obs.message = (info or {}).get("message", "") if isinstance(info, dict) else ""
        else:
            obs = res
            reward = getattr(res, "reward", 0.0)
            done = getattr(res, "done", False)

        _actions_history.append(action)
        rewards_history.append(reward)
        total_reward += reward

        steps.append(AutoRunStepResult(
            step=step_count,
            action=action,
            reasoning=reasoning,
            reward=reward,
            status=obs.status,
            cpu_usage=obs.cpu_usage,
            memory_usage=obs.memory_usage,
            db_latency=obs.db_latency,
            message=obs.message or "",
            logs=obs.logs or "",
            services=obs.services or [],
        ))

    if step_count >= MAX_STEPS and obs.status != "healthy":
        print(f"  ⚠ auto-run hit MAX_STEPS ({MAX_STEPS}) without reaching healthy status.")

    # Compute the proper episodic grade — same call as run_inference().
    # This gives a 0–1.0 score across all 5 components, not summed step rewards.
    final_obs_dict = {
        "status": obs.status,
        "cpu_usage": obs.cpu_usage,
        "memory_usage": obs.memory_usage,
        "db_latency": obs.db_latency,
        "services": obs.services,
        "logs": obs.logs,
    }
    try:
        score = grade(resolved_task_id, env_instance.actions_taken, final_obs_dict)
    except ValueError as exc:
        # grade() raised — log it and fall back to summed step rewards so the
        # endpoint doesn't 500. This should only happen if the env uses a dynamic
        # task_id that graders.py doesn't know about.
        print(f"  ⚠ grade() failed for task_id='{resolved_task_id}': {exc}. Using raw reward sum.")
        score = total_reward

    return AutoRunResponse(
        steps=steps,
        final_status=obs.status,
        total_steps=len(steps),
        total_reward=score,
    )

# ---------------------------------------------------------------------------
# Single consolidated entry point
# ---------------------------------------------------------------------------

def main():
    """CLI entry point for `uv run server` / `openenv` compatibility."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()