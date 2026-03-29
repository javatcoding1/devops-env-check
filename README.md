# 🤖 DevOps Incident Response Environment (v3.1)

A **production-grade, OpenEnv-compliant** simulation platform for training and evaluating AI agents in SRE (Site Reliability Engineering) and DevOps roles. This environment models complex infrastructure failures, hidden root causes, and failure propagation using the `openenv-core` SDK.

---

## 📐 Deep Architecture & Technology Stack

### Core Technologies
- **Logic**: Python 3.10+ using `dataclasses` and `Pydantic v2` for strict state modeling.
- **Communication**: **FastAPI** with `openenv-core` scaffolding, providing an OpenAI-compatible interface.
- **Data Models**: Typed `DevOpsAction`, `DevOpsObservation`, and `DevOpsState` ensuring 100% compliance with OpenEnv CLI validators.
- **UI**: **Streamlit** with a custom CSS design system for real-time metric visualization.
- **Agent**: Lightweight OpenAI-compatible client with **anti-loop safeguards** and **rule-based fallbacks**.

### 🔍 Simulation Engine (The "Deep" Logic)
The environment operates on a **Dual-State Model**:
1.  **Observable State (`DevOpsObservation`)**: Metrics like CPU (%), Memory (%), DB Latency (Low/Med/High), and Service Status (True/False). 
2.  **Hidden State (`SystemState`)**: The actual root cause (e.g., `memory_leak: true`, `query_lock: true`). The agent must infer these from "Check Logs" actions or metric patterns.

#### 📈 Failure Propagation & Metric Evolution
Metrics are not random; they evolve based on hidden failures:
- **CPU Overload**: If `cpu > 80%`, DB Latency has a 30% chance to increase per step.
- **Memory Leak**: If `memory > 85%`, the API service has a 50% chance to crash (`status=down`).
- **Wrong Actions**: Executing `scale_up:cpu` when the issue is actually a `database_lock` will trigger a penalty and slightly increase memory usage as a "resource overhead" simulation.

#### 📝 Dynamic Log Generator
The `LogGenerator` uses a 3-depth evolution system:
- **Baseline**: "Monitoring system active."
- **Symptomatic**: "High latency detected on port 5432."
- **Diagnostic**: "Internal Error: Table lock detected on 'orders' table. 15 processes waiting." (Only revealed after `check_logs` is called).

---

## 🎯 Task & Grader System

### Difficulty Levels
- **Easy**: Single failure (e.g., Service crash). Requires 1 action.
- **Medium**: Metrics threshold failure (e.g., CPU high). Requires 2 actions.
- **Hard**: Hidden root cause (e.g., Memory leak). Requires 2 actions and a restart.
- **Expert**: Multi-failure chain (e.g., CPU high + DB Latency). Requires 4+ actions in correct sequence.

### 📊 5-Component Deterministic Scoring
Every episode is graded out of 1.0 based on:
1.  **Outcome (35%)**: Was the system restored to `healthy`?
2.  **Action Selection (25%)**: Did the agent use the *optimal* fix for the specific root cause?
3.  **Efficiency (15%)**: Ratio of optimal steps vs. actual steps taken.
4.  **Health Bonus (15%)**: Final status of CPU/Memory (lowest possible values are best).
5.  **Diagnostic Bonus (10%)**: Did the agent call `check_logs` before attempting a fix?

---

## ⚡ Task-Oriented AI Agent (inference.py)

The included agent uses a **Token-Optimized Reasoning Chain**:
1.  **Intent**: Detect which metrics are hazardous.
2.  **Dependencies**: Check if prerequisite actions (like `check_logs`) are done.
3.  **Failure Avoidance**: Ensure the action hasn't been tried 3× repeatedly (Anti-loop).
4.  **Orchestration**: Select the final action string.

---

## 📁 Project Structure

```text
devopsai/
├── env/
│   ├── environment.py   # Core simulation logic (Failure propagation, Step/Reset)
│   ├── tasks.py         # Scenario definitions (Easy -> Expert)
│   ├── models.py        # Pydantic models (Action, Observation, State)
│   ├── graders.py       # Deterministic reward calculation
│   └── client.py        # DevOpsEnvClient for remote API connection
├── server/
│   └── app.py           # FastAPI application (Powered by create_fastapi_app)
├── agent/               # Legacy agent code (Archived)
├── inference.py         # Root level baseline Agent & Diagnostic Script
├── streamlit_app.py     # Interactive Web Dashboard
├── openenv.yaml         # OpenEnv manifest for platform deployment
├── pyproject.toml       # Build system and dependencies (openenv-core, fastapi, etc.)
└── Dockerfile           # Multi-stage production container
```

---

## 🚀 Deployment & Validation

### 1. Local Testing
```bash
python inference.py
```

### 2. Platform Validation
To prepare for submission to Hugging Face Spaces or the OpenEnv platform:
```bash
openenv validate  # Verifies pyproject.toml, openenv.yaml, and Dockerfile
openenv push --repo-id your-name/devops-simulator
```

---

## 📜 License
MIT
