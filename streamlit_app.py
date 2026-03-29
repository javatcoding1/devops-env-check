"""
DevOps AI Agent Simulator — Streamlit UI (v3)
===============================================
Interactive web interface for the DevOps Incident Response Environment.
Communicates with the FastAPI backend via HTTP.
"""

import json
import time
import requests
import streamlit as st
from dotenv import load_dotenv
import os


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


load_dotenv()

API_URL = os.getenv("API_URL") or st.secrets.get("API_URL")
st.set_page_config(
    page_title="DevOps AI Agent Simulator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .hero-section {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        color: white;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .hero-section h1 { margin: 0 0 0.3rem 0; font-size: 2rem; }
    .hero-section p { margin: 0.2rem 0; opacity: 0.85; font-size: 0.95rem; }

    .metric-card {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        color: white;
    }
    .metric-card .label { font-size: 0.75rem; opacity: 0.6; text-transform: uppercase; letter-spacing: 1px; }
    .metric-card .value { font-size: 1.8rem; font-weight: 700; margin: 0.3rem 0; }

    .status-healthy { color: #00e676; }
    .status-degraded { color: #ffab00; }
    .status-down { color: #ff1744; }
    .status-critical { color: #d50000; font-weight: 800; animation: blink 1.5s linear infinite; }
    @keyframes blink { 50% { opacity: 0.3; } }

    .log-box {
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 1rem;
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
        font-size: 0.82rem;
        color: #c9d1d9;
        white-space: pre-wrap;
        max-height: 200px;
        overflow-y: auto;
    }

    .chat-msg {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        color: #e6edf3;
    }

    .action-btn { margin: 0.2rem; }

    .step-log {
        background: #0d1117;
        border-left: 3px solid #58a6ff;
        padding: 0.6rem 1rem;
        margin: 0.3rem 0;
        border-radius: 0 6px 6px 0;
        font-size: 0.85rem;
        color: #c9d1d9;
    }
    .step-log.positive { border-left-color: #00e676; }
    .step-log.negative { border-left-color: #ff1744; }

    .auto-step {
        background: #0d1117;
        border-left: 3px solid #bb86fc;
        padding: 0.6rem 1rem;
        margin: 0.3rem 0;
        border-radius: 0 6px 6px 0;
        font-size: 0.85rem;
        color: #c9d1d9;
    }
    .auto-step.positive { border-left-color: #00e676; }
    .auto-step.negative { border-left-color: #ff1744; }

    .result-banner {
        background: linear-gradient(135deg, #1b5e20, #2e7d32);
        border: 1px solid #43a047;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .result-banner.failed {
        background: linear-gradient(135deg, #b71c1c, #c62828);
        border-color: #e53935;
    }
    .result-banner h3 { margin: 0; font-size: 1.4rem; }
    .result-banner p { margin: 0.3rem 0 0 0; opacity: 0.9; }

    .mode-card {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1.2rem;
        color: white;
    }
    .mode-card h4 { margin: 0 0 0.5rem 0; }
    .mode-card p { margin: 0; opacity: 0.8; font-size: 0.9rem; }

    .usecase-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 0.8rem;
        margin: 1rem 0;
    }
    .usecase-item {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 1rem;
        color: white;
        text-align: center;
    }
    .usecase-item .icon { font-size: 1.8rem; margin-bottom: 0.4rem; }
    .usecase-item .title { font-weight: 600; font-size: 0.9rem; }
    .usecase-item .desc { font-size: 0.78rem; opacity: 0.7; margin-top: 0.3rem; }

    .reasoning-box {
        background: #161b22;
        border: 1px solid #bb86fc;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        margin: 0.2rem 0 0.5rem 0;
        color: #e6edf3;
        font-size: 0.85rem;
    }
    .reasoning-box .label { color: #bb86fc; font-weight: 600; font-size: 0.78rem; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def api_call(method: str, endpoint: str, data: dict = None) -> dict | list | None:
    """Make an API call to the FastAPI backend."""
    url = f"{API_URL}{endpoint}"
    try:
        if method == "GET":
            r = requests.get(url, timeout=10)
        else:
            r = requests.post(url, json=data or {}, timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error(f"⚠️ Cannot connect to backend at {API_URL}. Please ensure it is running.")
        return None
    except Exception as e:
        st.error(f"API Error: {e}")
        return None


def get_status_class(status: str) -> str:
    return f"status-{status}" if status in ("healthy", "degraded", "down", "critical") else ""


def get_cpu_color(cpu: int) -> str:
    if cpu <= 40: return "#00e676"
    if cpu <= 70: return "#ffab00"
    return "#ff1744"


def get_mem_color(mem: int) -> str:
    if mem <= 50: return "#00e676"
    if mem <= 75: return "#ffab00"
    return "#ff1744"


def get_latency_color(lat: str) -> str:
    return {"low": "#00e676", "medium": "#ffab00", "high": "#ff1744"}.get(lat, "#999")


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "state" not in st.session_state:
    st.session_state.state = None
if "step_history" not in st.session_state:
    st.session_state.step_history = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "actions_taken" not in st.session_state:
    st.session_state.actions_taken = []
if "total_reward" not in st.session_state:
    st.session_state.total_reward = 0.0
if "episode_done" not in st.session_state:
    st.session_state.episode_done = False
if "auto_run_result" not in st.session_state:
    st.session_state.auto_run_result = None


# ---------------------------------------------------------------------------
# Hero Section
# ---------------------------------------------------------------------------

st.markdown("""
<div class="hero-section">
    <h1>🤖 DevOps AI Agent Simulator</h1>
    <p style="font-size: 1.1rem; margin-bottom: 0.8rem;">Interactive simulation of real-world DevOps system failures with AI-powered diagnosis</p>
    <p>🔍 Diagnose system issues from logs &nbsp;•&nbsp; ⚙️ Simulate real DevOps failures (CPU, memory, DB, cache)</p>
    <p>🤖 AI agent resolves incidents step-by-step &nbsp;•&nbsp; 📈 Handle dynamic and evolving system states</p>
    <p>💬 Chat with AI for debugging and explanations &nbsp;•&nbsp; 🔗 Perform multi-step fixes like real-world systems</p>
    <p>💥 Simulate failure propagation for wrong actions &nbsp;•&nbsp; 🎲 Generate random tasks for unlimited training</p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Real-World Use Cases (add-on)
# ---------------------------------------------------------------------------

st.markdown("""
<div style="margin-bottom: 1.5rem;">
    <h3 style="margin-bottom: 0.5rem;">🌍 Where This Can Be Used</h3>
    <p style="opacity: 0.7; font-size: 0.9rem; margin-bottom: 0.8rem;">This system simulates how AI agents can diagnose and resolve real-world infrastructure issues automatically.</p>
    <div class="usecase-grid">
        <div class="usecase-item">
            <div class="icon">🏢</div>
            <div class="title">DevOps Automation</div>
            <div class="desc">Automated incident response for cloud systems</div>
        </div>
        <div class="usecase-item">
            <div class="icon">⚙️</div>
            <div class="title">Incident Response</div>
            <div class="desc">Production-grade failure diagnosis & remediation</div>
        </div>
        <div class="usecase-item">
            <div class="icon">📊</div>
            <div class="title">Self-Healing Infra</div>
            <div class="desc">Monitoring and automatic issue resolution</div>
        </div>
        <div class="usecase-item">
            <div class="icon">🤖</div>
            <div class="title">AI-Powered SRE</div>
            <div class="desc">Site Reliability Engineering with intelligent agents</div>
        </div>
        <div class="usecase-item">
            <div class="icon">☁️</div>
            <div class="title">Cloud Integration</div>
            <div class="desc">AWS, GCP, Azure log analysis & auto-remediation</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar — Controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## ⚙️ Controls")

    def load_selected_task():
        """Callback to automatically load the task when dropdown changes."""
        task_id = st.session_state.task_choice_sb
        result = api_call("POST", "/reset", {"task_id": task_id})
        if result:
            st.session_state.state = result
            st.session_state.step_history = []
            st.session_state.actions_taken = []
            st.session_state.chat_history = []
            st.session_state.total_reward = 0.0
            st.session_state.episode_done = False
            st.session_state.auto_run_result = None

    # Task selection
    available_tasks = ["easy", "medium", "hard", "expert"]

    st.markdown("### 📋 Select Task")
    task_choice = st.selectbox(
        "Task", 
        available_tasks, 
        label_visibility="collapsed", 
        key="task_choice_sb",
        on_change=load_selected_task
    )

    col_reset, col_gen = st.columns(2)
    with col_reset:
        if st.button("🔄 Reset", use_container_width=True):
            load_selected_task()
            st.rerun()

    with col_gen:
        if st.button("🎲 Random", use_container_width=True):
            import time
            tid = f"random_scenario_{int(time.time()) % 1000}"
            gen_result = api_call("POST", "/generate", {
                "task_id": tid, "difficulty": "random",
            })
            if gen_result:
                result = api_call("POST", "/reset", {"task_id": tid})
                if result:
                    st.session_state.state = result
                    st.session_state.step_history = []
                    st.session_state.actions_taken = []
                    st.session_state.chat_history = []
                    st.session_state.total_reward = 0.0
                    st.session_state.episode_done = False
                    st.session_state.auto_run_result = None
                    st.toast(f"🎲 Generated new task: {tid}")
                    st.rerun()

    st.divider()

    # Step history
    st.markdown("### 📜 Action History")
    if st.session_state.actions_taken:
        for i, a in enumerate(st.session_state.actions_taken, 1):
            st.markdown(f"`{i}.` {a}")
        st.metric("Total Reward", f"{st.session_state.total_reward:+.2f}")
    else:
        st.caption("No actions taken yet. Reset a task to begin.")

    st.divider()

    # How AI Works
    st.markdown("### 🧠 How AI Works")
    st.markdown("""
    1. **Analyzes** system state (CPU, memory, DB, logs)
    2. **Identifies** likely root cause from patterns
    3. **Selects** the best corrective action
    4. **Executes** multi-step fix chains
    5. **Learns** from feedback (rewards/penalties)
    """)


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

if st.session_state.state is None:
    st.info("👈 Select a task and click **🔄 Reset** to start a simulation.")
    st.stop()

state = st.session_state.state

# ── System State Panel ──────────────────────────────────────────────

st.markdown("## 📊 System State")

cols = st.columns(5)

with cols[0]:
    cpu = state["cpu_usage"]
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">CPU Usage</div>
        <div class="value" style="color: {get_cpu_color(cpu)}">{cpu}%</div>
    </div>
    """, unsafe_allow_html=True)

with cols[1]:
    mem = state["memory_usage"]
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">Memory</div>
        <div class="value" style="color: {get_mem_color(mem)}">{mem}%</div>
    </div>
    """, unsafe_allow_html=True)

with cols[2]:
    lat = state["db_latency"]
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">DB Latency</div>
        <div class="value" style="color: {get_latency_color(lat)}">{lat.upper()}</div>
    </div>
    """, unsafe_allow_html=True)

with cols[3]:
    status = state["status"]
    if status == "down" and st.session_state.get("task_choice_sb") == "expert":
        status = "critical"
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">Status</div>
        <div class="value {get_status_class(status)}">{status.upper()}</div>
    </div>
    """, unsafe_allow_html=True)

with cols[4]:
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">Step</div>
        <div class="value" style="color: #64b5f6">{state['step_count']}</div>
    </div>
    """, unsafe_allow_html=True)

# Services
svc_icons = {"api": "🌐", "database": "🗄️", "cache": "⚡"}
svc_text = "  ".join(
    f"{svc_icons.get(s, '•')} **{s}** ✅" if s in state["services"]
    else f"{svc_icons.get(s, '•')} **{s}** ❌"
    for s in ["api", "database", "cache"]
)
st.markdown(f"**Services:** {svc_text}")

# Logs
st.markdown("**📝 System Logs:**")
st.markdown(f'<div class="log-box">{state["logs"]}</div>', unsafe_allow_html=True)

st.markdown("---")

# ── Action Panel ────────────────────────────────────────────────────

action_col, chat_col = st.columns([1, 1])

with action_col:
    st.markdown("## 🎮 Actions")

    if st.session_state.episode_done:
        if state["status"] == "healthy":
            st.markdown(
                '<div class="result-banner">'
                '<h3>✅ System Fully Resolved!</h3>'
                '<p>All root causes have been addressed. The system is healthy.</p>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="result-banner failed">'
                f'<h3>⚠️ Episode Ended — Status: {state["status"].upper()}</h3>'
                '<p>Max steps reached. The system was not fully restored.</p>'
                '</div>',
                unsafe_allow_html=True,
            )
        st.info("Click **🔄 Reset** in the sidebar to start a new episode.")
    else:
        actions = [
            ("🔄 restart_service:api", "restart_service:api"),
            ("🗄️ restart_service:database", "restart_service:database"),
            ("📈 scale_up:cpu", "scale_up:cpu"),
            ("⚡ optimize_database", "optimize_database"),
            ("🧹 clear_cache", "clear_cache"),
            ("🔍 check_logs", "check_logs"),
            ("⏸️ do_nothing", "do_nothing"),
        ]

        action_cols = st.columns(3)
        for i, (label, action_id) in enumerate(actions):
            with action_cols[i % 3]:
                disabled = action_id in st.session_state.actions_taken and action_id not in ("check_logs", "do_nothing")
                if st.button(label, key=f"act_{action_id}", use_container_width=True, disabled=disabled):
                    result = api_call("POST", "/step", {"action": action_id})
                    if result:
                        obs = result["observation"]
                        reward = result["reward"]
                        done = result["done"]
                        info = result["info"]

                        st.session_state.state = obs
                        st.session_state.actions_taken.append(action_id)
                        st.session_state.total_reward += reward
                        st.session_state.episode_done = done
                        st.session_state.step_history.append({
                            "step": len(st.session_state.actions_taken),
                            "action": action_id,
                            "reward": reward,
                            "status": obs["status"],
                            "message": info.get("message", ""),
                        })
                        st.rerun()

    # Step log
    if st.session_state.step_history:
        st.markdown("### 📋 Step Log")
        for entry in reversed(st.session_state.step_history):
            reward = entry["reward"]
            cls = "positive" if reward > 0 else "negative" if reward < 0 else ""
            st.markdown(
                f'<div class="step-log {cls}">'
                f'<strong>Step {entry["step"]}</strong>: {entry["action"]} → '
                f'reward={reward:+.4f} | status={entry["status"]}'
                f'<br><small>{entry["message"]}</small></div>',
                unsafe_allow_html=True,
            )


# ── Chat Panel ──────────────────────────────────────────────────────

with chat_col:
    st.markdown("## 💬 AI Diagnosis Chat")
    st.caption("Ask the AI about the current incident for diagnosis and advice.")

    # Chat input
    user_msg = st.text_input("Ask about the incident...", key="chat_input",
                             placeholder="What's wrong with the system?")
    if st.button("🔍 Diagnose", use_container_width=True) and user_msg:
        result = api_call("POST", "/chat", {"message": user_msg})
        if result:
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_msg,
            })
            st.session_state.chat_history.append({
                "role": "assistant",
                "diagnosis": result.get("diagnosis", ""),
                "reasoning": result.get("reasoning", ""),
                "suggested_action": result.get("suggested_action", ""),
            })
            st.rerun()

    # Chat history
    if st.session_state.chat_history:
        for msg in reversed(st.session_state.chat_history):
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="chat-msg"><strong>🧑 You:</strong> {msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="chat-msg">'
                    f'<strong>🤖 AI Diagnosis:</strong><br>'
                    f'<strong>Issue:</strong> {msg.get("diagnosis", "")}<br>'
                    f'<strong>Reasoning:</strong> {msg.get("reasoning", "")}<br>'
                    f'<strong>Suggested Action:</strong> <code>{msg.get("suggested_action", "")}</code>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
    else:
        st.caption("No chat messages yet. Ask a question above!")


# ---------------------------------------------------------------------------
# AI Agent Mode (add-on — appended below existing UI)
# ---------------------------------------------------------------------------

st.markdown("---")

st.markdown("## 🤖 AI Agent Mode")
st.caption("Let the AI agent automatically diagnose and resolve the incident.")

agent_col1, agent_col2 = st.columns([2, 1])

with agent_col1:
    if st.session_state.episode_done:
        st.info("Episode already finished. **🔄 Reset** to run the agent on a new task.")
    elif st.session_state.state is None:
        st.info("Select a task and **🔄 Reset** first, then run the AI agent.")
    else:
        if st.button("▶️ Run AI Agent Automatically", use_container_width=True, type="primary"):
            # Step-by-step animated execution
            progress_placeholder = st.empty()
            step_container = st.container()

            with st.spinner("🤖 AI Agent is analyzing the system..."):
                result = api_call("POST", "/auto-run")

            if result:
                # Animate step-by-step reveal
                with step_container:
                    for i, step_entry in enumerate(result.get("steps", [])):
                        progress_placeholder.progress(
                            (i + 1) / max(len(result["steps"]), 1),
                            text=f"⏳ AI is executing step {i+1}/{len(result['steps'])}..."
                        )
                        time.sleep(0.6)  # Realistic delay between steps

                    progress_placeholder.progress(1.0, text="✅ AI Agent finished!")
                    time.sleep(0.5)
                    progress_placeholder.empty()

                st.session_state.auto_run_result = result
                final_state = api_call("GET", "/state")
                if final_state:
                    st.session_state.state = final_state
                st.session_state.episode_done = True
                for step_entry in result.get("steps", []):
                    st.session_state.actions_taken.append(step_entry["action"])
                    st.session_state.total_reward += step_entry["reward"]
                    st.session_state.step_history.append({
                        "step": step_entry["step"],
                        "action": step_entry["action"],
                        "reward": step_entry["reward"],
                        "status": step_entry["status"],
                        "message": step_entry.get("message", ""),
                    })
                st.rerun()

with agent_col2:
    if st.session_state.auto_run_result:
        r = st.session_state.auto_run_result
        st.metric("Final Status", r["final_status"].upper())
        st.metric("Steps Taken", r["total_steps"])
        st.metric("Total Reward", f"{r['total_reward']:+.2f}")

# Display auto-run execution log
if st.session_state.auto_run_result:
    ar = st.session_state.auto_run_result

    if ar["final_status"] == "healthy":
        st.markdown(
            '<div class="result-banner">'
            '<h3>✅ AI Agent Successfully Resolved the Incident!</h3>'
            f'<p>Completed in {ar["total_steps"]} steps with total reward {ar["total_reward"]:+.2f}</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="result-banner failed">'
            f'<h3>⚠️ Agent Finished — Status: {ar["final_status"].upper()}</h3>'
            f'<p>{ar["total_steps"]} steps taken. System not fully restored.</p>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown("### 📋 AI Execution Log")
    for step_entry in ar.get("steps", []):
        reward = step_entry["reward"]
        reasoning = step_entry.get("reasoning", "")
        cls = "positive" if reward > 0 else "negative" if reward < 0 else ""
        st.markdown(
            f'<div class="auto-step {cls}">'
            f'<strong>Step {step_entry["step"]}</strong>: '
            f'{step_entry["action"]} → '
            f'reward={reward:+.4f} | status={step_entry["status"]} | '
            f'cpu={step_entry["cpu_usage"]}% mem={step_entry["memory_usage"]}% db={step_entry["db_latency"]}'
            f'<br><small>{step_entry["message"]}</small></div>',
            unsafe_allow_html=True,
        )
        if reasoning:
            st.markdown(
                f'<div class="reasoning-box">'
                f'<span class="label">🧠 Reasoning:</span> {reasoning}'
                f'</div>',
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Mode Explanation (add-on — appended at bottom)
# ---------------------------------------------------------------------------

st.markdown("---")

st.markdown("## 📖 Execution Modes")

mode_col1, mode_col2 = st.columns(2)

with mode_col1:
    st.markdown(
        '<div class="mode-card">'
        '<h4>🎮 Manual Mode</h4>'
        '<p>You control actions step-by-step. Choose which action to execute, '
        'observe the results, and decide the next move. Great for learning '
        'how the system works and understanding action dependencies.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

with mode_col2:
    st.markdown(
        '<div class="mode-card">'
        '<h4>🤖 AI Agent Mode</h4>'
        '<p>The AI agent automatically diagnoses and resolves the system. '
        'It uses structured reasoning (Intent → Dependencies → Failure Avoidance '
        '→ Orchestration) to find the optimal fix sequence. Watch the execution '
        'log to see how the agent reasons.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

