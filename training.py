"""
DevOps Incident Response – Training Script (Stable Baselines 3)
=============================================================
Trains a PPO agent to resolve system incidents.
Includes a custom callback to log progress for the Streamlit UI.
"""

import os
import json
import time
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from env.environment import DevOpsEnv

# Path for Streamlit to monitor
LOG_FILE = "training_log.json"

class StreamlitCallback(BaseCallback):
    """Custom callback for logging training progress to a JSON file."""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.metrics = []

    def _on_step(self) -> bool:
        # Log every 500 steps
        if self.n_calls % 500 == 0:
            reward = 0
            if len(self.model.ep_info_buffer) > 0:
                reward = sum([info['r'] for info in self.model.ep_info_buffer]) / len(self.model.ep_info_buffer)
            
            entry = {
                "step": self.num_timesteps,
                "reward": float(reward),
                "timestamp": time.time()
            }
            self.metrics.append(entry)
            
            with open(LOG_FILE, "w") as f:
                json.dump(self.metrics, f)
        return True

def train(timesteps=10000):
    """Run PPO training loop."""
    print(f"🚀 Starting RL training for {timesteps} steps...")
    
    # Initialize env
    env = make_vec_env(DevOpsEnv, n_envs=1)
    
    # Initialize PPO
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        tensorboard_log="./ppo_devops_tensorboard/",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        device="cpu" # Use CPU for local stability
    )
    
    # Clear old logs
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
    
    callback = StreamlitCallback()
    
    # Train
    model.learn(total_timesteps=timesteps, callback=callback)
    
    # Save
    model_path = "models/ppo_devops_agent.zip"
    os.makedirs("models", exist_ok=True)
    model.save(model_path)
    print(f"✅ Training complete! Model saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=10000)
    args = parser.parse_args()
    
    train(timesteps=args.timesteps)
