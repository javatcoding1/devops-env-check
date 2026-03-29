"""
DevOps Incident Response – FastAPI Server (v3 / OpenEnv Core)
=============================================================
Exposes the DevOpsEnv as a REST API for OpenEnv compatibility
using the official create_fastapi_app factory, plus LLM chat endpoints.
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
from env.tasks import get_task_ids, register_dynamic_task

# ---------------------------------------------------------------------------
# Setup OpenEnv Server
# ---------------------------------------------------------------------------

# Global environment instance
env_instance = DevOpsEnv()

# This factory provides the required strict /reset, /step, and /state endpoints
# create_fastapi_app expects a factory/class, not an instance.
# We use a lambda to return our singleton instance to keep custom endpoints in sync.
app = create_fastapi_app(lambda: env_instance, DevOpsAction, DevOpsObservation)

app.title = "DevOps Incident Response Environment (v3 / OpenEnv)"
app.version = "3.1.1"
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

# Context tracking for chat
_actions_history: List[str] = []

# ---------------------------------------------------------------------------
# Extra Request / Response schemas for custom endpoints
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

class AutoRunResponse(BaseModel):
    steps: List[AutoRunStepResult]
    final_status: str
    total_steps: int
    total_reward: float

class HealthResponse(BaseModel):
    status: str = "ok"
    environment: str = "DevOps Incident Response Environment"
    version: str = "3.1.0"

# ---------------------------------------------------------------------------
# Custom Endpoints (Preserved for UI and backwards compat)
# ---------------------------------------------------------------------------

@app.get("/health-check", response_model=HealthResponse, tags=["Meta"])
async def health():
    return HealthResponse()

@app.get("/tasks", response_model=List[str], tags=["Meta"])
async def list_tasks():
    return get_task_ids()

@app.get("/actions", response_model=List[str], tags=["Meta"])
async def list_actions():
    return VALID_ACTIONS

@app.post("/chat", response_model=ChatResponse, tags=["AI Agent"])
async def chat_endpoint(body: ChatRequest):
    # Dynamic import
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
    from inference import get_action_with_reasoning
    global _actions_history

    # Clear history for are fresh auto-run session
    _actions_history = []

    if body.task_id is not None:
        try:
            env_instance.reset(task_id=body.task_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        _actions_history = []

    steps: List[AutoRunStepResult] = []
    total_reward = 0.0

    while True:
        # Use openenv dict dump for legacy compatibility
        current_state = env_instance._sys_state.to_dict()

        if current_state["status"] == "healthy":
            break

        # Get AI action with reasoning
        result = get_action_with_reasoning(current_state, _actions_history)
        action = result["action"]
        reasoning = result["reasoning"]

        print(f"DEBUG [auto-run]: Step {len(steps)+1} - State: status={current_state['status']}, cpu={current_state['cpu_usage']}%, mem={current_state['memory_usage']}% - Action: {action}")

        # Stop if agent decides do_nothing (even if status is not yet healthy)
        if action == "do_nothing":
            print(f"DEBUG [auto-run]: Agent suggested 'do_nothing'. Breaking loop.")
            break

        # Execute step
        obs = env_instance.step(DevOpsAction(action_str=action))
        _actions_history.append(action)
        total_reward += obs.reward

        steps.append(AutoRunStepResult(
            step=len(steps) + 1,
            action=action,
            reasoning=reasoning,
            reward=obs.reward,
            status=obs.status,
            cpu_usage=obs.cpu_usage,
            memory_usage=obs.memory_usage,
            db_latency=obs.db_latency,
            message=obs.message,
        ))

        if obs.done:
            break

    final_state = env_instance._sys_state.to_dict()
    return AutoRunResponse(
        steps=steps,
        final_status=final_state["status"],
        total_steps=len(steps),
        total_reward=total_reward,
    )

# ---------------------------------------------------------------------------
# CLI entry point (for `uv run server` / `openenv` compatibility)
# ---------------------------------------------------------------------------

def run_server():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    run_server()
