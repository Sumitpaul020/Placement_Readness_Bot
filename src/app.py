"""
app.py
------
Streamlit chat interface for the Placement Readiness Bot.

Run (from the project root, with venv activated):
    streamlit run src/app.py
"""

import os
import uuid
import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # loads OPENAI_API_KEY from a .env file if present

from agent import build_agent, chat

st.set_page_config(page_title="Placement Readiness Bot", page_icon="🎓", layout="wide")

st.title("🎓 Placement Readiness Bot")
st.caption(
    "Ask me things like *'Am I eligible for placements?'*, "
    "*'What's my placement probability with CGPA 7.5, 0 backlogs, 2 internships...?'*, "
    "or *'Am I eligible AND likely to get placed?'*"
)

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

# ---------- Session state ----------
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

# ---------- Render chat history ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------- Chat input ----------
user_input = st.chat_input("Ask about your placement eligibility or chances...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                reply = chat(st.session_state.agent, st.session_state.thread_id, user_input)
            except Exception as e:
                reply = f"Something went wrong: {e}"
            st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
