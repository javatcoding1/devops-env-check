from pydantic import Field
from openenv.core.env_server import Action, Observation, State
from typing import List

class DevOpsAction(Action):
    """The action taken in the DevOps environment."""
    action_str: str

class DevOpsObservation(Observation):
    """The observation from the DevOps environment."""
    logs: str = ""
    cpu_usage: int = 0
    memory_usage: int = 0
    db_latency: str = ""
    services: List[str] = Field(default_factory=list)
    status: str = "healthy"
    step_count: int = 0
    message: str = ""

class DevOpsState(State):
    """Episode state tracking for DevOps environment."""
    pass
