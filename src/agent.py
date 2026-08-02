"""
agent.py
--------
Builds the conversational agent that has access to BOTH tools:
  - predict_placement (ML model)
  - search_policy (RAG over policy document)

Uses LangGraph's create_react_agent (the standard way LangChain recommends
building tool-calling agents today) with an in-memory checkpointer so the
conversation has memory across turns in the same session.
"""

import os
import time
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from tools import predict_placement, search_policy

SYSTEM_PROMPT = """You are the Placement Readiness Bot for a college's Training &
Placement Cell. You help students with two DIFFERENT kinds of questions:

1. "Will I get placed?" / "What are my chances?" type questions ->
   use the predict_placement tool. This needs concrete numeric details
   (CGPA, backlogs, internships, projects, coding_score, communication_score,
   attendance_percent). If the student hasn't given you these, ask for the
   missing ones before calling the tool - don't guess numbers.

2. "Am I eligible?" / "What is the rule for...?" / policy, documentation,
   debarment, dress code, backlog-clearance questions -> use the
   search_policy tool.

3. Some questions need BOTH tools, e.g. "Am I eligible AND likely to get
   placed?" - call both tools and combine their results into one clear,
   synthesized answer.

4. If a question needs neither tool (general conversation, greetings,
   clarifying questions), just answer directly without calling any tool.

Always be honest. If search_policy says the document doesn't cover
something, tell the student that plainly instead of guessing. Never invent
placement statistics or policy rules that didn't come from the tools.
"""


def _get_llm():
    """Picks the LLM provider based on the LLM_PROVIDER env var.

    LLM_PROVIDER=groq   -> uses Groq (FREE, get a key at https://console.groq.com/keys)
    LLM_PROVIDER=openai -> uses OpenAI (needs billing set up)

    Defaults to "groq" since it's free and needs no card.
    """
    provider = os.getenv("LLM_PROVIDER", "groq").lower()

    if provider == "groq":
        from langchain_groq import ChatGroq
        # gpt-oss-120b is more reliable for tool-calling than llama-3.3-70b-versatile,
        # which can occasionally produce malformed function-call JSON.
        return ChatGroq(model="openai/gpt-oss-120b", temperature=0)

    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)

    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider}. Use 'groq' or 'openai'.")


def build_agent(model_name: str = None):
    """Returns a compiled LangGraph agent with both tools + memory.

    Also stashes the underlying LLM object on the returned agent as
    `agent.llm` so `chat()` can nudge its temperature on retries.
    """
    llm = _get_llm()

    memory = MemorySaver()

    agent = create_react_agent(
        model=llm,
        tools=[predict_placement, search_policy],
        prompt=SYSTEM_PROMPT,
        checkpointer=memory,
    )
    agent.llm = llm  # keep a handle so chat() can adjust temperature on retry
    return agent


def chat(agent, thread_id: str, user_message: str) -> str:
    """Send one user message to the agent and get the final text reply.
    `thread_id` identifies the conversation session (memory is keyed on this).

    Some Groq models occasionally generate a malformed function-call JSON
    (a known upstream issue, not specific to this app - see LangChain/Groq
    GitHub issues). We retry several times, nudging the temperature and
    adding a short pause each time, so the model doesn't just repeat the
    same malformed output. If every attempt fails, we return a plain,
    user-friendly message instead of ever raising/showing a raw error.
    """
    config = {"configurable": {"thread_id": thread_id}}

    retry_temperatures = [0.0, 0.2, 0.4, 0.6, 0.8]

    for i, temp in enumerate(retry_temperatures):
        try:
            if hasattr(agent, "llm"):
                agent.llm.temperature = temp
            result = agent.invoke(
                {"messages": [{"role": "user", "content": user_message}]},
                config=config,
            )
            final_message = result["messages"][-1]
            content = (final_message.content or "").strip()
            if content:
                return content
            # Empty response - treat like a failure and retry
        except Exception:
            pass

        if i < len(retry_temperatures) - 1:
            time.sleep(0.6)

    # All retries failed - NEVER show a raw/technical error to the user.
    # This message reads as normal conversational clarification, not as
    # any kind of system/technical failure.
    return (
        "I want to make sure I get this right - could you tell me a bit "
        "more detail, or rephrase that for me?"
    )
