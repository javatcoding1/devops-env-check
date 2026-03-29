# 🚨 DevOps Incident Response Environment (v3)

A **production-grade, OpenEnv-compatible** reinforcement learning environment that simulates real-world DevOps system failures with hidden root causes, dynamic logs, failure propagation, and an interactive Streamlit UI.

---

## ✨ What's New in v3

| Feature | Description |
|---------|-------------|
| 🎲 **Dynamic Task Generation** | Random task creation with `generate_task(difficulty, seed)` |
| 💬 **AI Diagnostic Chat** | LLM-powered `/chat` endpoint for interactive debugging |
| 🖥️ **Streamlit UI** | Full interactive web interface with metrics, actions, and chat |
| ⚡ **Token Optimization** | 60% less token usage — truncated logs, recent-only history, max_tokens=50 |
| 🛑 **Smart Stopping** | Auto-stop when system is healthy; no unnecessary actions |
| 🔁 **Anti-Loop Guards** | Prevents 3× repeats, caps check_logs at 3, rule-based fallback |
| 📈 **Failure Propagation** | Wrong actions degrade CPU, memory, and DB latency |
| 🔗 **Multi-Step Chains** | memory_leak → clear_cache → restart_service:api |

---

## 📐 Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                 Streamlit UI (streamlit_app.py)               │
│   Metrics Dashboard │ Action Buttons │ AI Chat │ Step Log     │
├──────────────┬───────────────────────────────────────────────┤
│              │         FastAPI Server (app.py)                │
│              │                                                │
│  POST /reset ──────► DevOpsEnv.reset(task_id)                │
│  POST /step  ──────► DevOpsEnv.step(action)                  │
│  GET  /state ──────► DevOpsEnv.state()                       │
│  POST /chat  ──────► chat_diagnose(state, history, msg)      │
│  POST /generate ───► register_dynamic_task(id, difficulty)   │
│  GET  /health ─────► Liveness probe                          │
│              │                                                │
├──────────────┴───────────────────────────────────────────────┤
│                  DevOpsEnv (environment.py)                    │
│                                                                │
│   SystemState (observable) ←→ HiddenState (root causes)       │
│   LogGenerator (3-depth evolution)                             │
│   Failure propagation │ State evolution │ Multi-step chains    │
│                                                                │
├────────────────┬────────────────────┬────────────────────────┤
│  tasks.py      │   graders.py       │   inference.py          │
│  Static tasks  │   5-component      │   Token-optimized       │
│  + dynamic     │   scoring (0–1)    │   LLM agent + chat      │
│  generation    │                    │   + anti-loop guards     │
└────────────────┴────────────────────┴────────────────────────┘
```

---

## ⚡ Quick Start

### 1. Install

```bash
git clone <repo-url>
cd devopsai
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env with your API key
```

### 3. Run API Server

```bash
uvicorn app:app --host 0.0.0.0 --port 7860 --reload
```

### 4. Run Streamlit UI

```bash
# In a second terminal:
streamlit run streamlit_app.py --server.port 8501
```

### 5. Run LLM Inference

```bash
python inference.py
```

---

## 🖥️ Streamlit UI Guide

The UI provides:

- **📊 System State Panel** — Real-time CPU, memory, DB latency, status with color-coded metrics
- **🎮 Action Panel** — Clickable action buttons (disabled after use to prevent repeats)
- **📋 Step Log** — Color-coded history of every action and its reward
- **💬 AI Chat** — Ask questions about the incident, get diagnosis + suggested action
- **⚙️ Sidebar** — Task selection, random task generation, action history, and "How it works" guides

---

## 🎮 Execution Modes

### Manual Mode
You control actions step-by-step. Choose which action to execute, observe the results, and decide the next move. Great for learning how the system works and understanding action dependencies.

### AI Agent Mode
The AI agent automatically diagnoses and resolves the system using the `/auto-run` endpoint. It uses structured reasoning (Intent → Dependencies → Failure Avoidance → Orchestration) to find the optimal fix sequence.

```bash
# Run agent automatically via API
curl -X POST http://localhost:7860/auto-run \
  -H "Content-Type: application/json" \
  -d '{"task_id": "medium"}'
```

Response includes step-by-step execution log, final status, total steps, and total reward.

Or click **▶️ Run AI Agent Automatically** in the Streamlit UI.

---

## 🎲 Dynamic Task Generation

Generate randomized tasks at runtime:

```python
from tasks import generate_task, register_dynamic_task

# Generate without registering
task = generate_task("hard", seed=42)

# Generate and register for use with env.reset()
register_dynamic_task("my_task", "medium", seed=123)
```

Or via API:

```bash
curl -X POST http://localhost:7860/generate \
  -H "Content-Type: application/json" \
  -d '{"task_id": "rng_1", "difficulty": "hard", "seed": 42}'
```

---

## 🎮 Action Space

| Action | Description |
|--------|-------------|
| `restart_service:api` | Restart the API service |
| `restart_service:database` | Restart the database service |
| `scale_up:cpu` | Scale up CPU resources |
| `optimize_database` | Optimise database queries/indexes |
| `clear_cache` | Clear and rebuild the cache |
| `check_logs` | Analyse system logs (reveals deeper info each time) |
| `do_nothing` | Take no action |

---

## 🎯 Reward Structure

| Outcome | Reward |
|---------|--------|
| Fixes root cause | +1.0 |
| Partial fix (multi-step chain) | +0.7 |
| Investigation (check_logs) | +0.5 / +0.3 / +0.1 |
| Wrong action | −0.2 to −0.35 |
| Per-step penalty | −0.05 |
| do_nothing while issues active | −0.1 |

---

## 🛑 Stopping Logic

The agent automatically stops when:
- System status becomes `"healthy"` → returns `do_nothing`
- Max steps (15) reached
- Anti-loop guard triggers after 3× same action

---

## ⚡ Token Optimization Strategy

| Technique | Saving |
|-----------|--------|
| Truncated logs (last 200 chars) | ~40% fewer prompt tokens |
| Recent actions only (last 5) | ~30% fewer history tokens |
| Compact prompt format | ~50% vs verbose v2 prompt |
| max_tokens=50 (response) | ~75% fewer completion tokens |
| Hard stop on healthy | Eliminates unnecessary API calls |
| Anti-loop guards | Prevents repeated wasted calls |

**Total estimated savings: ~60% fewer tokens per episode vs v2.**

---

## 📊 Grading

5-component scoring:

| Component | Weight | Description |
|-----------|--------|-------------|
| Outcome | 35% | Was the system restored to healthy? |
| Action Match | 25% | Did the agent use correct actions? |
| Efficiency | 15% | Extra steps vs optimal? |
| Health Metrics | 15% | Final CPU, memory, DB latency values |
| Diagnostic Bonus | 10% | Did the agent investigate first? |

---

## 🐳 Docker

```bash
docker build -t devops-incident-env .
docker run -p 7860:7860 -p 8501:8501 devops-incident-env
```

---

## ☁️ Deploy to Hugging Face Spaces

1. Create a **Docker** Space on [huggingface.co/spaces](https://huggingface.co/spaces)
2. Push this repository:

```bash
git remote add space https://huggingface.co/spaces/<username>/devops-incident-env
git push space main
```

3. Set secrets in Space Settings: `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`

---

## 📁 Project Structure

```
devopsai/
├── environment.py      # Core RL environment (hidden root causes, dynamic logs)
├── tasks.py            # Static + dynamic task generation
├── graders.py          # 5-component deterministic scoring
├── inference.py        # Token-optimized LLM agent + chat helper
├── app.py              # FastAPI server (REST + chat + generate)
├── streamlit_app.py    # Interactive Streamlit UI
├── openenv.yaml        # OpenEnv specification
├── Dockerfile          # Production container (FastAPI + Streamlit)
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
└── README.md           # This file
```

---

## 🔧 Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `API_BASE_URL` | OpenAI-compatible endpoint | `https://api.groq.com/openai/v1` |
| `MODEL_NAME` | Model identifier | `llama3-8b-8192` |
| `HF_TOKEN` | API key / bearer token | — |

---

## 📜 License

MIT
