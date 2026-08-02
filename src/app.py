"""
app.py
------
Streamlit chat interface for the Placement Readiness Bot.

Run (from the project root, with venv activated):
    streamlit run src/app.py
"""

import os
import uuid
import matplotlib.pyplot as plt
import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # loads OPENAI_API_KEY from a .env file if present

from agent import build_agent, chat
from tools import predict_placement_raw, get_last_prediction, clear_last_prediction


def render_prediction_chart(probability: float):
    """Renders a simple horizontal bar chart showing Placed vs Not-Placed
    probability. Labels are written directly on the bars when there's room;
    if a segment is too narrow to fit its own label, the label is placed
    just outside that segment (in its own color) instead of overlapping."""
    not_placed_pct = (1 - probability) * 100
    placed_pct = probability * 100
    THRESHOLD = 20  # segments narrower than this (in percentage points) get an outside label

    fig, ax = plt.subplots(figsize=(6, 1.8))
    ax.barh([0], [placed_pct], color="#2e7d32", height=0.5)
    ax.barh([0], [not_placed_pct], left=[placed_pct], color="#c62828", height=0.5)

    # "Placed" label (left/green segment)
    if placed_pct >= THRESHOLD:
        ax.text(placed_pct / 2, 0, f"Placed {placed_pct:.1f}%",
                ha="center", va="center", color="white", fontsize=11, fontweight="bold")
    else:
        ax.text(-2, 0, f"Placed {placed_pct:.1f}%",
                ha="right", va="center", color="#2e7d32", fontsize=11, fontweight="bold")

    # "Not placed" label (right/red segment)
    if not_placed_pct >= THRESHOLD:
        ax.text(placed_pct + not_placed_pct / 2, 0, f"Not placed {not_placed_pct:.1f}%",
                ha="center", va="center", color="white", fontsize=11, fontweight="bold")
    else:
        ax.text(102, 0, f"Not placed {not_placed_pct:.1f}%",
                ha="left", va="center", color="#c62828", fontsize=11, fontweight="bold")

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, 0.6)
    ax.set_yticks([])
    ax.set_title("Placement Chances", fontsize=13, pad=10)
    ax.set_xlabel("Probability (%)")
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.set_page_config(page_title="Placement Readiness Bot", page_icon="🎓", layout="wide")

st.title("🎓 Placement Readiness Bot")

# ---------- Sidebar: dataset + model charts ----------
with st.sidebar:
    st.header("📊 Model Insights")
    if os.path.exists("charts/feature_importance.png"):
        st.image("charts/feature_importance.png", caption="Feature importance")
    else:
        st.info("Run `python src/train_model.py` first to generate charts.")

    if os.path.exists("charts/dataset_overview.png"):
        st.image("charts/dataset_overview.png", caption="Dataset overview")

    st.divider()
    st.caption(
        "This bot uses a trained Decision Tree for placement-likelihood "
        "predictions and a RAG pipeline over the official placement policy "
        "document for eligibility/rule questions."
    )

tab_chat, tab_predict = st.tabs(["💬 Chat with the Bot", "⚡ Quick Predictor"])

# ---------- Session state (must be initialized before use below) ----------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "agent" not in st.session_state:
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    key_name = "GROQ_API_KEY" if provider == "groq" else "OPENAI_API_KEY"
    if not os.getenv(key_name):
        st.warning(
            f"No {key_name} found for LLM_PROVIDER='{provider}'. "
            f"Set it in a `.env` file in the project root (see README) before chatting."
        )
    st.session_state.agent = build_agent()

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": ..., "content": ...} for display only

# IMPORTANT: st.chat_input must be called at the page's top level (not
# nested inside st.tabs/columns/containers) for it to pin to the bottom of
# the screen like a normal chat app. We call it here, then use its value
# inside the Chat tab below.
user_input = st.chat_input("Ask about your placement eligibility or chances...")

# =====================================================================
# TAB 1: Chat (agent decides which tool(s) to use - predict and/or policy)
# =====================================================================
with tab_chat:
    st.caption(
        "Ask me things like *'Am I eligible for placements?'*, "
        "*'My CGPA is 7.5 with 1 backlog, 2 internships... what are my chances?'*, "
        "or *'Am I eligible AND likely to get placed?'* - the bot decides "
        "which tool(s) to use automatically."
    )

    # ---------- Render chat history ----------
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ---------- Handle new input (widget itself was created above) ----------
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                clear_last_prediction()
                try:
                    reply = chat(st.session_state.agent, st.session_state.thread_id, user_input)
                except Exception:
                    # Never show a raw/technical error to the user - this
                    # reads as normal conversational clarification.
                    reply = (
                        "I want to make sure I get this right - could you "
                        "tell me a bit more detail, or rephrase that for me?"
                    )
                st.markdown(reply)

                # If the agent called predict_placement during this turn,
                # show a chart of the chances right below the reply.
                prediction = get_last_prediction()
                if prediction is not None:
                    render_prediction_chart(prediction["probability"])

        st.session_state.messages.append({"role": "assistant", "content": reply})

# =====================================================================
# TAB 2: Quick Predictor - a plain form that calls the trained model
# directly (no LLM involved), for instant, no-typing predictions.
# =====================================================================
with tab_predict:
    st.caption(
        "Fill in your details below to get an instant placement-likelihood "
        "prediction straight from the trained model - no chat needed."
    )

    with st.form("quick_predict_form"):
        col1, col2 = st.columns(2)

        with col1:
            cgpa = st.number_input("CGPA (out of 10)", min_value=0.0, max_value=10.0, value=7.0, step=0.1)
            backlogs = st.number_input("Active backlogs", min_value=0, max_value=10, value=0, step=1)
            internships = st.number_input("Internships completed", min_value=0, max_value=10, value=1, step=1)
            projects = st.number_input("Projects completed", min_value=0, max_value=10, value=2, step=1)

        with col2:
            coding_score = st.slider("Coding score", 0, 100, 65)
            communication_score = st.slider("Communication score", 0, 100, 70)
            attendance_percent = st.slider("Attendance %", 0, 100, 80)

        submitted = st.form_submit_button("Predict my placement chances")

    if submitted:
        result = predict_placement_raw(
            cgpa=cgpa,
            backlogs=backlogs,
            internships=internships,
            projects=projects,
            coding_score=coding_score,
            communication_score=communication_score,
            attendance_percent=attendance_percent,
        )

        proba_pct = result["probability"] * 100

        if result["placed"] == 1:
            st.success(f"✅ Likely to be PLACED - estimated probability: {proba_pct:.1f}%")
        else:
            st.error(f"⚠️ Currently NOT likely to be placed - estimated probability: {proba_pct:.1f}%")

        render_prediction_chart(result["probability"])
        st.caption(
            "This uses the same trained model as the chat's predict_placement "
            "tool, so results will always match (see README test case TC5)."
        )
