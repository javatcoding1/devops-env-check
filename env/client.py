from openenv.core.env_client import EnvClient
from env.models import DevOpsAction, DevOpsObservation, DevOpsState

class DevOpsEnvClient(EnvClient[DevOpsAction, DevOpsObservation, DevOpsState]):
    """Client for connecting to a deployed DevOpsEnvironment via REST API."""
    
    def __init__(self, base_url: str):
        super().__init__(
            base_url=base_url,
            action_cls=DevOpsAction,
            observation_cls=DevOpsObservation,
            state_cls=DevOpsState
        )
