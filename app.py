"""
DevOps Incident Response – FastAPI Server (v3)
===============================================
Exposes the DevOpsEnv as a REST API for OpenEnv compatibility,
plus LLM-powered /chat endpoint and dynamic task generation.

Endpoints:
    POST /reset       – reset(task_id)
    POST /step        – step(action)
    GET  /state       – state()
    POST /chat        – LLM diagnosis chat
    POST /generate    – generate dynamic task
    POST /auto-run    – run AI agent automatically
    GET  /health      – liveness probe
    GET  /tasks       – list available task IDs
    GET  /actions     – list valid actions
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from environment import DevOpsEnv, VALID_ACTIONS
from tasks import get_task_ids, register_dynamic_task

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="DevOps Incident Response Environment",
    description=(
        "A production-grade OpenEnv-compatible DevOps simulation with hidden "
        "root causes, dynamic logs, failure propagation, multi-step fixes, "
        "and LLM-powered diagnostic chat."
    ),
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instance
env = DevOpsEnv()
# Track actions for chat context
_actions_history: List[str] = []


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: Optional[str] = Field(
        default=None,
        description="Task ID. Static: easy/medium/hard/expert. Or any dynamically generated ID.",
    )


class StepRequest(BaseModel):
    action: str = Field(
        ..., description="Action to execute. Must be one of the valid actions.",
    )


class GenerateRequest(BaseModel):
    task_id: str = Field(
        ..., description="ID to register the generated task under.",
    )
    difficulty: str = Field(
        default="medium",
        description="Difficulty: easy, medium, hard, expert, or random.",
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducible generation.",
    )


class ChatRequest(BaseModel):
    message: str = Field(
        ..., description="User message / question about the current incident.",
    )


class StateResponse(BaseModel):
    logs: str
    cpu_usage: int
    memory_usage: int
    db_latency: str
    services: List[str]
    status: str
    step_count: int


class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]


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
    task_id: Optional[str] = Field(
        default=None,
        description="Task to auto-run. If None, uses the currently loaded task.",
    )


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
    version: str = "3.0.0"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/reset", response_model=StateResponse, tags=["Environment"])
async def reset_env(body: ResetRequest = ResetRequest()):
    """Reset the environment, optionally specifying a task."""
    global _actions_history
    _actions_history = []
    try:
        obs = env.reset(task_id=body.task_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return StateResponse(**obs)


@app.post("/step", response_model=StepResponse, tags=["Environment"])
async def step_env(body: StepRequest):
    """Execute an action and receive observation, reward, done, info."""
    action = body.action.strip().lower()
    if action not in VALID_ACTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid action '{body.action}'. Valid: {VALID_ACTIONS}",
        )
    obs, reward, done, info = env.step(action)
    _actions_history.append(action)
    return StepResponse(observation=obs, reward=reward, done=done, info=info)


@app.get("/state", response_model=StateResponse, tags=["Environment"])
async def get_state():
    """Return the current system state."""
    return StateResponse(**env.state())


@app.post("/chat", response_model=ChatResponse, tags=["AI Agent"])
async def chat_endpoint(body: ChatRequest):
    """LLM-powered diagnostic chat using current state and history."""
    from inference import chat_diagnose

    state = env.state()
    result = chat_diagnose(state, _actions_history, body.message)
    return ChatResponse(**result)


@app.post("/generate", response_model=GenerateResponse, tags=["Tasks"])
async def generate_task_endpoint(body: GenerateRequest):
    """Generate and register a dynamic task."""
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


@app.get("/health", response_model=HealthResponse, tags=["Meta"])
async def health():
    """Liveness / readiness probe."""
    return HealthResponse()


@app.get("/tasks", response_model=List[str], tags=["Meta"])
async def list_tasks():
    """Return all available task identifiers."""
    return get_task_ids()


@app.get("/actions", response_model=List[str], tags=["Meta"])
async def list_actions():
    """Return all valid action strings."""
    return VALID_ACTIONS


@app.post("/auto-run", response_model=AutoRunResponse, tags=["AI Agent"])
async def auto_run(body: AutoRunRequest = AutoRunRequest()):
    """
    Run the AI agent automatically step-by-step until the system is
    healthy or max steps are reached. Returns action + reasoning per step.
    """
    from inference import get_action_with_reasoning

    global _actions_history

    # Reset if task_id provided
    if body.task_id is not None:
        try:
            env.reset(task_id=body.task_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        _actions_history = []

    steps: List[AutoRunStepResult] = []
    total_reward = 0.0

    while True:
        current_state = env.state()

        # Stop if system is already healthy
        if current_state["status"] == "healthy":
            break

        # Get AI action with reasoning
        result = get_action_with_reasoning(current_state, _actions_history)
        action = result["action"]
        reasoning = result["reasoning"]

        # Stop if agent decides do_nothing on a healthy system
        if action == "do_nothing" and current_state["status"] == "healthy":
            break

        # Execute step
        obs, reward, done, info = env.step(action)
        _actions_history.append(action)
        total_reward += reward

        steps.append(AutoRunStepResult(
            step=len(steps) + 1,
            action=action,
            reasoning=reasoning,
            reward=round(reward, 4),
            status=obs["status"],
            cpu_usage=obs["cpu_usage"],
            memory_usage=obs["memory_usage"],
            db_latency=obs["db_latency"],
            message=info.get("message", ""),
        ))

        if done:
            break

    final_state = env.state()
    return AutoRunResponse(
        steps=steps,
        final_status=final_state["status"],
        total_steps=len(steps),
        total_reward=round(total_reward, 4),
    )
