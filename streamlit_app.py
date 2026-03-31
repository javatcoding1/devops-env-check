"""
DevOps AI Agent Simulator — Streamlit UI (v4)
===============================================
Interactive web interface for the DevOps Incident Response Environment.
Communicates with the FastAPI backend via HTTP.
v4 fixes:
  - Passes task_id to /auto-run (fixes always-easy bug)
  - AI suggestion button in manual mode via /suggest
  - Reasoning display in manual step log
"""

import requests
import json
import time
import subprocess
import signal
try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv
import os
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


load_dotenv()

API_URL = os.getenv("API_URL")
if not API_URL:
    try:
        API_URL = st.secrets.get("API_URL")
    except Exception: # Catch StreamlitSecretNotFoundError and others
        API_URL = None

if not API_URL:
    API_URL = "http://localhost:7860"

st.set_page_config(
    page_title="DevOps AI Agent Simulator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)
try:
    import gymnasium as gym
    import numpy as np
    import stable_baselines3
    HAS_TRAINING = True
except ImportError:
    HAS_TRAINING = False

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
            # POST requests can involve long-running LLM loops (auto-run)
            r = requests.post(url, json=data or {}, timeout=300)
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

def get_delta_html(curr: Any, prev: Any, invert: bool = False) -> str:
    if prev is None or curr == prev: return ""
    try:
        diff = float(curr) - float(prev)
        if diff > 0:
            color = "#ff1744" if not invert else "#00e676"
            return f"<div style='font-size:0.75rem; color:{color}; margin-top:2px;'>▲ +{diff:.1f}</div>"
        else:
            color = "#00e676" if not invert else "#ff1744"
            return f"<div style='font-size:0.75rem; color:{color}; margin-top:2px;'>▼ {diff:.1f}</div>"
    except (ValueError, TypeError):
        return f"<div style='font-size:0.75rem; color:#ffab00; margin-top:2px;'>➤ from {str(prev).upper()}</div>"


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "state" not in st.session_state:
    st.session_state.state = None
if "prev_state" not in st.session_state:
    st.session_state.prev_state = None
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
if "training_process" not in st.session_state:
    st.session_state.training_process = None
if "training_log" not in st.session_state:
    st.session_state.training_log = []


# ---------------------------------------------------------------------------
# Training Dashboard
# ---------------------------------------------------------------------------

def show_training_dashboard():
    st.markdown("## 🤖 Reinforcement Learning Training")
    
    if not HAS_TRAINING:
        st.warning("⚠️ **Platform Not Supported for Local Training**")
        st.markdown(f"""
        RL Training requires **PyTorch** and **Stable Baselines 3**, which are currently not supported on your platform 
        (**{st.session_state.get('platform', 'Intel Mac')}** + **Python 3.13**).

        **How to Train?**
        1. **Push to Cloud**: Click the 'Push' button or run `openenv push`.
        2. **Run in Hugging Face**: All features will be fully functional on the Linux-based cloud environment.
        """)
        return

    st.info("Train a **PPO (Proximal Policy Optimization)** model using Stable Baselines 3 to resolve incidents automatically.")
    # ... rest of training logic ...

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### ⚙️ Training Settings")
        timesteps = st.number_input("Total Timesteps", min_value=1000, max_value=100000, value=10000, step=1000)
        
        if st.session_state.training_process is None:
            if st.button("🚀 Start Training", type="primary", use_container_width=True):
                try:
                    proc = subprocess.Popen(["python", "training.py", "--timesteps", str(timesteps)])
                    st.session_state.training_process = proc.pid
                    st.success(f"Training started (PID: {proc.pid})")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to start training: {e}")
        else:
            if st.button("🛑 Stop Training", type="secondary", use_container_width=True):
                try:
                    os.kill(st.session_state.training_process, signal.SIGTERM)
                    st.session_state.training_process = None
                    st.warning("Training stopped.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to stop process: {e}")
        
        st.markdown("---")
        st.markdown("**Status:** " + ("🟢 Running" if st.session_state.training_process else "⚪ Idle"))
    
    with col2:
        st.markdown("### 📈 Performance Monitor")
        
        if os.path.exists("training_log.json"):
            try:
                with open("training_log.json", "r") as f:
                    log_data = json.load(f)
                
                if log_data:
                    steps = [entry["step"] for entry in log_data]
                    rewards = [entry["reward"] for entry in log_data]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=steps, y=rewards, mode='lines+markers', name='Mean Reward'))
                    fig.update_layout(
                        title="Mean Episode Reward vs. Timesteps",
                        xaxis_title="Timesteps",
                        yaxis_title="Mean Reward",
                        template="plotly_dark",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.table(log_data[-5:][::-1])
                else:
                    st.info("Waiting for first training metrics...")
            except Exception as e:
                st.error(f"Error reading log: {e}")
        else:
            st.info("No training log found. Start training to see metrics.")


# ---------------------------------------------------------------------------
# Sidebar — Controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## ⚙️ Controls")
    view_mode = st.radio("View Mode", ["Incident Dashboard", "AI Training Hub"])
    st.divider()

    if view_mode == "Incident Dashboard":
        def load_selected_task():
            task_id = st.session_state.task_choice_sb
            st.session_state.active_task_id = task_id  # Save active task
            result = api_call("POST", "/reset", {"task_id": task_id})
            if result:
                st.session_state.state = result.get("observation", result)
                st.session_state.prev_state = None
                st.session_state.step_history = []
                st.session_state.actions_taken = []
                st.session_state.chat_history = []
                st.session_state.total_reward = 0.0
                st.session_state.episode_done = False
                st.session_state.auto_run_result = None

        available_tasks = ["easy", "medium", "hard", "expert"]
        st.markdown("### 📋 Select Task")
        task_choice = st.selectbox("Task", available_tasks, label_visibility="collapsed", key="task_choice_sb", on_change=load_selected_task)

        col_reset, col_gen = st.columns(2)
        with col_reset:
            if st.button("🔄 Reset", use_container_width=True):
                load_selected_task()
                st.rerun()
        with col_gen:
            if st.button("🎲 Random", use_container_width=True):
                tid = f"random_scenario_{int(time.time()) % 1000}"
                gen_result = api_call("POST", "/generate", {"task_id": tid, "difficulty": "random"})
                if gen_result:
                    result = api_call("POST", "/reset", {"task_id": tid})
                    if result:
                        st.session_state.active_task_id = tid  # Ensure Auto-Run uses this random task
                        st.session_state.state = result.get("observation", result)
                        st.session_state.prev_state = None
                        st.session_state.step_history = []
                        st.session_state.actions_taken = []
                        st.session_state.chat_history = []
                        st.session_state.total_reward = 0.0
                        st.session_state.episode_done = False
                        st.session_state.auto_run_result = None
                        st.toast(f"🎲 Generated new task: {tid}")
                        st.rerun()

        st.divider()
        st.markdown("### 📜 Action History")
        if st.session_state.actions_taken:
            for i, a in enumerate(st.session_state.actions_taken, 1):
                st.markdown(f"`{i}.` {a}")
            st.metric("Total Reward", f"{st.session_state.total_reward:+.2f}")
        else:
            st.caption("No actions taken yet. Reset a task to begin.")

    st.divider()
    st.markdown("### 🧠 How AI Works")
    st.markdown("""
    1. **Analyzes** system metrics and logs
    2. **Infers** hidden root causes (Leaks/Locks)
    3. **Executes** multi-step corrective chains
    """)

    with st.expander("🛠️ Technical Deep Dive"):
        st.markdown("""
        **Tech Stack:**
        - **Backend:** FastAPI + OpenEnv Core
        - **Logic:** Pydantic v2 + Dataclasses
        - **Agent:** OpenAI-compatible Reasoning Agent
        - **UI:** Streamlit + Custom CSS

        **Deep Logic:**
        - **Dual-State Engine:** Metrics are driven by *hidden* variables (Leaks, Locks).
        - **Failure Propagation:** High CPU increases DB Latency chance; High Memory triggers API crashes.
        - **Deterministic Grader:** Scientific 5-component scoring (0.0 - 1.0).
        """)


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

if view_mode == "AI Training Hub":
    show_training_dashboard()
else:
    # ── Hero Section ────────────────────────────────────────────────────
    st.markdown("""
<div class="hero-section">
    <h1>🤖 DevOps AI Agent Simulator</h1>
    <p style="font-size: 1.1rem; margin-bottom: 0.8rem;">Interactive simulation of real-world DevOps system failures with AI-powered diagnosis</p>
</div>
""", unsafe_allow_html=True)

    if st.session_state.state is None:
        st.info("👈 Select a task and click **🔄 Reset** to start a simulation.")
        st.stop()

    state = st.session_state.state
    prev_state = st.session_state.prev_state

    # ── System State Panel ──────────────────────────────────────────────
    st.markdown("## 📊 System State")
    cols = st.columns(5)

    with cols[0]:
        cpu = state["cpu_usage"]
        pcpu = prev_state.get("cpu_usage") if prev_state else None
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">CPU Usage</div>
            <div class="value" style="color: {get_cpu_color(cpu)}">{cpu}%</div>
            {get_delta_html(cpu, pcpu)}
        </div>
        """, unsafe_allow_html=True)

    with cols[1]:
        mem = state["memory_usage"]
        pmem = prev_state.get("memory_usage") if prev_state else None
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Memory</div>
            <div class="value" style="color: {get_mem_color(mem)}">{mem}%</div>
            {get_delta_html(mem, pmem)}
        </div>
        """, unsafe_allow_html=True)

    with cols[2]:
        lat = state["db_latency"]
        plat = prev_state.get("db_latency") if prev_state else None
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">DB Latency</div>
            <div class="value" style="color: {get_latency_color(lat)}">{lat.upper()}</div>
            {get_delta_html(lat, plat)}
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
    
    # Reveal Reward Progression Chart if available
    if HAS_PLOTLY and getattr(st.session_state, "step_history", []):
        cumulative = []
        cur = 0
        for e in st.session_state.step_history:
            cur += e.get("reward", 0)
            cumulative.append(cur)
        if cumulative:
            st.markdown("### 📈 Reward Progression")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[e["step"] for e in st.session_state.step_history] , 
                y=cumulative, 
                mode='lines+markers', 
                name='Total Reward', 
                line=dict(color='#00e676', width=2),
                marker=dict(size=8)
            ))
            fig.update_layout(
                xaxis_title="Step", 
                yaxis_title="Total Reward", 
                template="plotly_dark", 
                height=250, 
                margin=dict(t=10, b=10, l=10, r=10),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0.2)'
            )
            st.plotly_chart(fig, use_container_width=True)
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
                ("⏸️ no_action", "no_action"),
            ]

            # AI Suggestion Button
            suggest_col1, suggest_col2 = st.columns([3, 1])
            with suggest_col1:
                if st.button("🧠 Ask AI for Next Action", use_container_width=True, type="secondary"):
                    suggestion = api_call("POST", "/suggest")
                    if suggestion:
                        st.session_state["ai_suggestion"] = suggestion
                        st.rerun()
            with suggest_col2:
                pass
            
            if st.session_state.get("ai_suggestion"):
                sug = st.session_state["ai_suggestion"]
                st.markdown(
                    f'<div class="reasoning-box">'
                    f'<span class="label">🧠 AI Suggests:</span> <strong>{sug.get("action", "?")}</strong><br>'
                    f'{sug.get("reasoning", "")}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            action_cols = st.columns(3)
            for i, (label, action_id) in enumerate(actions):
                with action_cols[i % 3]:
                    disabled = action_id in st.session_state.actions_taken and action_id not in ("check_logs", "no_action")
                    if st.button(label, key=f"act_{action_id}", use_container_width=True, disabled=disabled):
                        # Get AI reasoning for this action
                        diag = {}
                        try:
                            suggest_result = api_call("POST", "/suggest")
                            if suggest_result:
                                diag = suggest_result
                        except Exception:
                            pass

                        result = api_call("POST", "/step", {"action": action_id})
                        if result:
                            obs = result.get("observation", result)
                            reward = result.get("reward", 0.0)
                            done = result.get("done", False)
                            step_diag = result.get("diagnosis", diag)

                            st.session_state.prev_state = dict(st.session_state.state)
                            st.session_state.state = obs
                            st.session_state.actions_taken.append(action_id)
                            st.session_state.total_reward += reward
                            st.session_state.episode_done = done
                            
                            metrics = obs.get("info", {})
                            entry = {
                                "step": len(st.session_state.actions_taken),
                                "action": action_id,
                                "reward": reward,
                                "status": obs["status"],
                                "message": obs.get("message", ""),
                                "info": metrics,
                                "reasoning": step_diag.get("reasoning", "") if isinstance(step_diag, dict) else ""
                            }
                            st.session_state.step_history.append(entry)
                            
                            # Update UI immediately for a "live" feel
                            st.toast(f"AI Step {entry['step']}: {action_id}", icon="🤖")
                            
                            if action_id == "no_action" and obs["status"] == "healthy":
                                st.success("🎯 AI Agent successfully resolved the incident!")
                                break
                            
                            # Small pause for visual effect
                            import time
                            time.sleep(1.2)
                            st.rerun()

        # Step log
        if st.session_state.step_history:
            st.markdown("### 📋 Step Log & Timeline")
            
            # Basic stats summary if info exists
            latest_info = st.session_state.step_history[-1].get("info", {})
            if latest_info:
                ok_actions = latest_info.get("total_correct_actions", 0)
                bad_actions = latest_info.get("total_wrong_actions", 0)
                st.markdown(f"**Metrics:** ✅ Correct Actions: `{ok_actions}` | ❌ Wrong Actions: `{bad_actions}`")
                
            for entry in reversed(st.session_state.step_history):
                reward = entry["reward"]
                cls = "positive" if reward > 0 else "negative" if reward < 0 else ""
                
                with st.expander(f"Step {entry['step']}: **{entry['action']}**", expanded=(entry['step'] == len(st.session_state.step_history))):
                    st.markdown(f"**Reward:** <span class='reward-{cls}'>{reward:+.4f}</span>", unsafe_allow_html=True)
                    st.markdown(f"**Status:** `{entry['status'].upper()}`")
                    if entry.get("reasoning"):
                        st.info(f"**AI Reasoning:** {entry['reasoning']}")
                    st.write(entry["message"])


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

                # Get the currently selected active task ID (works for both dropdown and random button)
                current_task = st.session_state.get("active_task_id", st.session_state.get("task_choice_sb", "easy"))

                with st.spinner(f"🤖 AI Agent is analyzing {current_task.upper()}..."):
                    # Clear UI history for a fresh auto-run visualization
                    st.session_state.actions_taken = []
                    st.session_state.step_history = []
                    result = api_call("POST", "/auto-run", {"task_id": current_task})

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
                    # Final observation is derived from the last step in result
                    if result.get("steps"):
                        last = result["steps"][-1]
                        st.session_state.state = {
                            "status": last["status"],
                            "cpu_usage": last["cpu_usage"],
                            "memory_usage": last["memory_usage"],
                            "db_latency": last["db_latency"],
                            "services": last.get("services", []), 
                            "logs": last.get("logs", ""), 
                            "step_count": last["step"],
                        }
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

