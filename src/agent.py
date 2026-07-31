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
        return ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)

    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider}. Use 'groq' or 'openai'.")


def build_agent(model_name: str = None):
    """Returns a compiled LangGraph agent with both tools + memory."""
    llm = _get_llm()

    memory = MemorySaver()

    agent = create_react_agent(
        model=llm,
        tools=[predict_placement, search_policy],
        prompt=SYSTEM_PROMPT,
        checkpointer=memory,
    )
    return agent


def chat(agent, thread_id: str, user_message: str) -> str:
    """Send one user message to the agent and get the final text reply.
    `thread_id` identifies the conversation session (memory is keyed on this)."""
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_message}]},
        config=config,
    )
    final_message = result["messages"][-1]
    return final_message.content
