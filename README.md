# 🤖 DevOps Incident Response AI (v4.0)

A **Production-Grade SRE AI Agent** platform featuring advanced **Causal Priority** and **Intent-Lock** reasoning protocols. Designed to train and evaluate AI agents in high-fidelity, multi-failure infrastructure scenarios.

---

## 🧠 SRE Reasoning Protocols (v4.0)

The core `inference.py` script has been overhauled with a **Senior SRE Decision Model**:

### 1. Causal Priority Hierarchy
To prevent the agent from fixing downstream symptoms before upstream root causes, it follows a strict causal list:
1.  **Memory** ➡ (Upstream: Leaks cause cache/DB/API pressure)
2.  **Cache** ➡ (Middle: Failures cause DB/API overload)
3.  **Database** ➡ (Middle: Latency causes API timeouts)
4.  **API** ➡ (Service Layer: Down/Slow)
5.  **CPU** ➡ (Symptom: Often caused by GC or DB wait)

### 2. Critical Override Fallback
In high-stakes scenarios, the agent can bypass the causal hierarchy if:
- A downstream metric (like **CPU**) is in a **Critical State (>95%)**.
- There is **Zero Evidence** in the logs for any higher-priority upstream cause.
This ensures the system is stabilized immediately during a meltdown.

---

## 🛠️ Technology Stack & Features

- **Logic**: Python 3.11 with Pydantic for state integrity.
- **Communication**: FastAPI with **OpenEnv SDK** compliance.
- **Diagnostics**: **Verification-Driven Log Protocol** (Diagnose ➡ Fix ➡ Verify).
- **Control**: **Intent-Lock Protocol** to ensure diagnostic stability during resolution.
- **Deployment**: Multi-stage **Dockerfile** optimized for Hugging Face Spaces.

---

## 📁 Project Structure

```text
devopsai/
├── env/
│   ├── environment.py   # Failure propagation & logic
│   ├── tasks.py         # Easy -> Expert scenarios
│   └── graders.py       # Deterministic scoring
├── server/
│   └── app.py           # OpenAI-compatible API
├── inference.py         # Advanced SRE AI Reasoner
├── streamlit_app.py     # Real-time dashboard
└── Dockerfile           # Optimized production container
```

---

## 🚀 Getting Started

### 1. Local Development
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python inference.py
```

### 2. Manual Diagnostics (Dashboard)
```bash
streamlit run streamlit_app.py
```

### 3. Containerized Deployment
```bash
docker build -t devops-sre-ai .
docker run -p 7860:7860 devops-sre-ai
```

---

## 📊 Evaluation & Grader
Every SRE session is graded based on **Outcome**, **Diagnostic Bonus**, and **Causal Efficiency**. The best agent is the one that fixes the upstream cause in the fewest possible steps.

---

## 📜 License
MIT
