# frontend.py
# pip install streamlit requests
# Run: streamlit run frontend.py

import time
import html
import re
import requests
import streamlit as st

# ============================================================
# CONFIG
# ============================================================
API_BASE       = "http://127.0.0.1:8000"
POLL_INTERVAL  = 2
POLL_MAX_TRIES = 90
MAX_RETRIES    = 2

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="NutriLens",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0f0f0e;
    color: #e8e4dc;
}
.block-container { padding-top: 2rem !important; padding-bottom: 2rem !important; }

/* hero */
.nutri-header {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem; color: #c8f05a;
    letter-spacing: -0.5px; line-height: 1.1; margin-bottom: 0.2rem;
}
.nutri-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem; color: #6b6b5e;
    letter-spacing: 2px; text-transform: uppercase; margin-bottom: 1.5rem;
}

/* chat bubbles */
.bubble-user {
    background: #1e1e1a; border: 1px solid #2e2e28;
    border-radius: 18px 18px 4px 18px;
    padding: 0.65rem 1rem; margin: 0.25rem 0;
    font-size: 0.9rem; color: #e8e4dc;
}
.bubble-agent {
    background: #141410; border: 1px solid #c8f05a22;
    border-left: 3px solid #c8f05a;
    border-radius: 4px 18px 18px 18px;
    padding: 0.65rem 1rem; margin: 0.25rem 0;
    font-size: 0.82rem; color: #d4d0c8;
    font-family: 'DM Mono', monospace;
    white-space: pre-wrap; line-height: 1.5;
}
.bubble-system {
    background: #1a1a16; border: 1px dashed #3a3a30;
    border-radius: 6px; padding: 0.3rem 0.75rem; margin: 0.15rem 0;
    font-size: 0.72rem; color: #6b6b5e;
    font-family: 'DM Mono', monospace; letter-spacing: 0.5px;
}

/* interrupt card */
.interrupt-card {
    background: #141410; border: 1px solid #c8f05a44;
    border-top: 3px solid #c8f05a; border-radius: 12px;
    padding: 1.1rem 1.3rem; margin: 0.6rem 0;
}
.interrupt-title { font-family: 'DM Serif Display', serif; font-size: 1.05rem; color: #c8f05a; margin-bottom: 0.15rem; }
.interrupt-meta  { font-family: 'DM Mono', monospace; font-size: 0.68rem; color: #6b6b5e; letter-spacing: 1px; margin-bottom: 0.7rem; }
.interrupt-result {
    background: #0f0f0e; border: 1px solid #2a2a22; border-radius: 7px;
    padding: 0.7rem; font-family: 'DM Mono', monospace;
    font-size: 0.78rem; color: #b8b4ac; white-space: pre-wrap;
    max-height: 200px; overflow-y: auto; line-height: 1.45;
}

/* metric cards */
.metric-card { background: #141410; border: 1px solid #2a2a22; border-radius: 11px; padding: 0.85rem; text-align: center; }
.metric-value { font-family: 'DM Serif Display', serif; font-size: 1.7rem; color: #c8f05a; line-height: 1.1; }
.metric-label { font-family: 'DM Mono', monospace; font-size: 0.65rem; color: #6b6b5e; letter-spacing: 1.5px; text-transform: uppercase; margin-top: 0.15rem; }

/* result box */
.result-box {
    background: #0f0f0e; border: 1px solid #2a2a22; border-radius: 10px;
    padding: 0.9rem; font-family: 'DM Mono', monospace; font-size: 0.8rem;
    color: #b8b4ac; white-space: pre-wrap; max-height: 280px; overflow-y: auto;
    line-height: 1.5; margin-top: 0.6rem;
}

/* badge */
.badge { display: inline-block; font-family: 'DM Mono', monospace; font-size: 0.63rem; letter-spacing: 1.5px; padding: 2px 8px; border-radius: 20px; text-transform: uppercase; }
.badge-running { background:#1e2e10; color:#c8f05a; border:1px solid #c8f05a44; }
.badge-done    { background:#102e10; color:#5af07a; border:1px solid #5af07a44; }
.badge-wait    { background:#2e2010; color:#f0c05a; border:1px solid #f0c05a44; }
.badge-error   { background:#2e1010; color:#f05a5a; border:1px solid #f05a5a44; }
.badge-none    { background:#1e1e1a; color:#6b6b5e; border:1px solid #2e2e28; }

/* sidebar */
section[data-testid="stSidebar"] { background: #0c0c0a; border-right: 1px solid #1e1e1a; }
.sidebar-section { font-family: 'DM Mono', monospace; font-size: 0.67rem; letter-spacing: 2px; color: #6b6b5e; text-transform: uppercase; margin-bottom: 0.35rem; margin-top: 0.6rem; }

/* inputs */
.stTextInput>div>div>input,
.stTextArea>div>div>textarea {
    background: #141410 !important; border: 1px solid #2e2e28 !important;
    color: #e8e4dc !important; border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important; font-size: 0.83rem !important;
}
.stTextInput>div>div>input:focus,
.stTextArea>div>div>textarea:focus { border-color: #c8f05a !important; box-shadow: 0 0 0 2px #c8f05a22 !important; }

/* buttons */
.stButton>button {
    font-family: 'DM Mono', monospace !important; font-size: 0.76rem !important;
    letter-spacing: 0.8px !important; border-radius: 8px !important;
    border: 1px solid #2e2e28 !important; background: #1a1a16 !important;
    color: #e8e4dc !important; transition: all 0.15s ease !important;
}
.stButton>button:hover { border-color: #c8f05a !important; color: #c8f05a !important; background: #141410 !important; }

/* multiselect */
[data-testid="stMultiSelect"]>div { background: #141410 !important; border-color: #2e2e28 !important; border-radius: 8px !important; }

/* file uploader */
[data-testid="stFileUploader"] { background: #141410; border: 1px dashed #2e2e28; border-radius: 10px; padding: 0.3rem; }

/* tabs */
.stTabs [data-baseweb="tab"] { font-family: 'DM Mono', monospace; font-size: 0.76rem; letter-spacing: 0.5px; color: #6b6b5e; }
.stTabs [aria-selected="true"] { color: #c8f05a !important; }
.stTabs [data-baseweb="tab-border"] { background-color: #c8f05a !important; }
.stTabs [data-baseweb="tab-list"] { background-color: transparent !important; gap: 4px; }

hr { border-color: #1e1e1a !important; }

/* tighten vertical rhythm */
.element-container { margin-bottom: 0.25rem !important; }
div[data-testid="stVerticalBlock"] > div { gap: 0.25rem !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================
defaults = {
    "session_id": None,
    "status":     None,
    "messages":   [],
    "interrupt":  None,
    "result":     None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================
# HELPERS
# ============================================================
def safe(x: str) -> str:
    return html.escape(str(x))

def reset_state():
    for k, v in defaults.items():
        st.session_state[k] = [] if isinstance(v, list) else v

def add_msg(role: str, content: str):
    if not any(m["content"] == content for m in st.session_state.messages):
        st.session_state.messages.append({"role": role, "content": content})

def api(method: str, path: str, **kwargs):
    try:
        fn = getattr(requests, method.lower())
        r  = fn(f"{API_BASE}{path}", timeout=60, **kwargs)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot reach API — is nutrition_api.py running on port 8000?")
        return None
    except Exception as e:
        st.error(f"API Error: {e}")
        return None

def trim_result(text: str) -> str:
    """Collapse 3+ blank lines → 1, strip trailing spaces per line."""
    text = text.strip()
    text = re.sub(r'[ \t]+\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text

def apply_response(data: dict):
    if not data:
        return
    status = data["status"]
    st.session_state.status = status
    if status == "interrupted":
        st.session_state.interrupt = data.get("interrupt")
    elif status == "done":
        st.session_state.interrupt = None
        result = trim_result(data.get("result", ""))
        st.session_state.result = result
        add_msg("agent", result)
        add_msg("system", f"✓ complete — {data.get('total_tool_calls', 0)} tool calls")
    elif status == "error":
        st.session_state.interrupt = None
        add_msg("system", f"⚠ error: {data.get('result', 'unknown')}")

def poll_until_settled():
    sid = st.session_state.session_id
    if not sid:
        return
    for _ in range(POLL_MAX_TRIES):
        data = api("get", f"/status/{sid}")
        if not data:
            return
        if data["status"] != "running":
            apply_response(data)
            return
        time.sleep(POLL_INTERVAL)
    add_msg("system", "⚠ timed out.")

def do_resume(payload: dict):
    sid = st.session_state.session_id
    st.session_state.interrupt = None
    st.session_state.status    = "running"
    data = api("post", f"/resume/{sid}", json=payload)
    if not data:
        st.session_state.status = "error"
        add_msg("system", "⚠ resume failed.")
        return
    if data["status"] == "running":
        poll_until_settled()
    else:
        apply_response(data)

def parse_nutrition(text: str) -> dict:
    """
    Extract totals from the agent result regardless of formatting.
    Handles both pipe-separated and compact (no-space) tables.
    Finds the TOTAL row and reads numbers in order: Protein, Carbs, Fat, Calories.
    """
    total_match = re.search(r"TOTAL[^\n]*", text, re.IGNORECASE)
    if total_match:
        total_row = total_match.group(0)
        nums = re.findall(r"(\d+(?:\.\d+)?)", total_row)
        if len(nums) >= 4:
            return {"protein": nums[0], "carbs": nums[1], "fat": nums[2], "calories": nums[3]}
        elif len(nums) == 3:
            return {"protein": nums[0], "carbs": nums[1], "fat": nums[2], "calories": "—"}

    # Fallback: keyword search anywhere
    def find(pats):
        for p in pats:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                return m.group(1).strip()
        return "—"

    return {
        "calories": find([r"(\d+(?:\.\d+)?)\s*(?:kcal|calories)", r"calories\D{0,10}(\d+(?:\.\d+)?)"]),
        "protein":  find([r"protein\D{0,10}(\d+(?:\.\d+)?)"]),
        "carbs":    find([r"carbs?\D{0,10}(\d+(?:\.\d+)?)"]),
        "fat":      find([r"fat\D{0,10}(\d+(?:\.\d+)?)"]),
    }


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(
        '<div class="nutri-header" style="font-size:1.45rem;margin-bottom:0.05rem">NutriLens</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="nutri-sub">AI Nutrition</div>', unsafe_allow_html=True)

    # session
    st.markdown('<div class="sidebar-section">Session</div>', unsafe_allow_html=True)
    if st.session_state.session_id:
        st.markdown(
            f'<code style="font-size:0.63rem;color:#6b6b5e">{st.session_state.session_id[:22]}…</code>',
            unsafe_allow_html=True,
        )
    status_class = {
        "running":     "badge-running",
        "interrupted": "badge-wait",
        "done":        "badge-done",
        "error":       "badge-error",
    }.get(st.session_state.status or "", "badge-none")
    st.markdown(
        f'<span class="badge {status_class}">{st.session_state.status or "idle"}</span>',
        unsafe_allow_html=True,
    )

    st.divider()

    # preferences
    st.markdown('<div class="sidebar-section">Preferences</div>', unsafe_allow_html=True)
    prefs_data = api("get", "/preferences") or {}

    nutrients = st.multiselect(
        "Nutrients",
        ["protein", "carbs", "fat", "calories", "fiber", "sodium", "sugar"],
        default=prefs_data.get("preferred_nutrients", ["protein", "carbs", "fat", "calories"]),
        label_visibility="collapsed",
    )
    flags = st.multiselect(
        "Dietary flags",
        ["vegan", "vegetarian", "gluten-free", "low-carb", "diabetic", "low-sodium"],
        default=prefs_data.get("dietary_flags", []),
        label_visibility="collapsed",
    )

    col_save, col_clear = st.columns(2)
    with col_save:
        if st.button("Save prefs", use_container_width=True):
            res = api("put", "/preferences", json={
                "preferred_nutrients": nutrients,
                "dietary_flags": flags,
            })
            if res:
                st.toast("Saved!", icon="✓")
    with col_clear:
        if st.button("Clear history", use_container_width=True):
            api("delete", "/preferences/history")
            st.toast("History cleared!", icon="✓")

    st.divider()

    if st.button("New session", use_container_width=True):
        reset_state()
        st.rerun()

# ============================================================
# HERO
# ============================================================
st.markdown('<div class="nutri-header">What did you eat?</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="nutri-sub">Upload a meal photo — get instant nutrition breakdown</div>',
    unsafe_allow_html=True,
)

# ============================================================
# UPLOAD
# ============================================================
if not st.session_state.session_id:
    uploaded = st.file_uploader(
        "Drop your meal photo here",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )
    if uploaded:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(uploaded, use_container_width=True, caption=uploaded.name)

        if st.button("🔍  Analyse nutrition", use_container_width=True):
            with st.spinner("Starting analysis…"):
                resp = api("post", "/analyze", files={
                    "file": (uploaded.name, uploaded.getvalue(), uploaded.type)
                })
            if resp:
                st.session_state.session_id = resp["session_id"]
                st.session_state.status     = "running"
                add_msg("user",   f"📷 {uploaded.name}")
                add_msg("system", "analysis started…")
                with st.spinner("Identifying food items…"):
                    poll_until_settled()
                st.rerun()

# ============================================================
# RUNNING STATE
# ============================================================
if st.session_state.status == "running":
    with st.spinner("Agent is working…"):
        poll_until_settled()
    st.rerun()

# ============================================================
# CHAT HISTORY
# ============================================================
if st.session_state.messages:
    for msg in st.session_state.messages:
        txt = safe(msg["content"])
        role = msg["role"]
        if role == "user":
            st.markdown(f'<div class="bubble-user">{txt}</div>', unsafe_allow_html=True)
        elif role == "agent":
            st.markdown(f'<div class="bubble-agent">{txt}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bubble-system">{txt}</div>', unsafe_allow_html=True)

# ============================================================
# INTERRUPT UI
# ============================================================
if st.session_state.status == "interrupted" and st.session_state.interrupt:
    iv     = st.session_state.interrupt
    tool   = iv.get("tool", "")
    label  = iv.get("label", "Review Needed")
    result = iv.get("result", "")
    retry  = iv.get("retry_count", 0)

    st.markdown(f"""
    <div class="interrupt-card">
        <div class="interrupt-title">⏸ {safe(label)}</div>
        <div class="interrupt-meta">TOOL: {safe(tool.upper())} &nbsp;|&nbsp; ATTEMPT {retry + 1}/{MAX_RETRIES + 1}</div>
        <div class="interrupt-result">{safe(result)}</div>
    </div>""", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["✓  Approve", "✕  Reject", "✎  Edit"])

    with tab1:
        st.caption("Send this result to the next step as-is.")
        if st.button("Approve and continue →", key="btn_approve", use_container_width=True):
            add_msg("user",   f"✓ approved {label}")
            add_msg("system", f"resuming from {tool}…")
            with st.spinner("Agent is working…"):
                do_resume({"type": "approve"})
            st.rerun()

    with tab2:
        st.caption("Retry this step with feedback.")
        reason = st.text_input(
            "reason", placeholder="e.g. You missed the sauce on the left",
            key="reject_reason", label_visibility="collapsed",
        )
        if st.button("Reject and retry ↺", key="btn_reject", use_container_width=True):
            if reason.strip():
                add_msg("user",   f"✕ rejected — {reason}")
                add_msg("system", f"retrying {tool}…")
                with st.spinner("Retrying…"):
                    do_resume({"type": "reject", "reason": reason})
                st.rerun()
            else:
                st.warning("Please give a reason before rejecting.")

    with tab3:
        st.caption("Correct the output manually and continue.")
        edited = st.text_area(
            "edit", value=result, height=170,
            key="edit_content", label_visibility="collapsed",
        )
        if st.button("Save edit and continue →", key="btn_edit", use_container_width=True):
            add_msg("user",   "✎ edited output manually")
            add_msg("system", f"continuing with edited {tool} output…")
            with st.spinner("Agent is working…"):
                do_resume({"type": "edit", "content": edited})
            st.rerun()

# ============================================================
# FINAL REPORT
# ============================================================
if st.session_state.status == "done" and st.session_state.result:
    nutrition = parse_nutrition(st.session_state.result)
    metrics   = [
        (nutrition["calories"], "Calories"),
        (nutrition["protein"],  "Protein (g)"),
        (nutrition["carbs"],    "Carbs (g)"),
        (nutrition["fat"],      "Fat (g)"),
    ]
    cols = st.columns(4)
    for col, (value, lbl) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{safe(value)}</div>
                <div class="metric-label">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:0.5rem'></div>", unsafe_allow_html=True)
    if st.button("Analyse another meal →", use_container_width=True):
        reset_state()
        st.rerun()

# ============================================================
# ERROR STATE
# ============================================================
if st.session_state.status == "error":
    st.error("Something went wrong — check the API logs.")
    if st.button("Start over", use_container_width=True):
        reset_state()
        st.rerun()