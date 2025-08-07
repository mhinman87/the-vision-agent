from state.agent_state import AgentState
from llm_config import classifier_llm
from langchain_core.messages import SystemMessage


def should_continue_chatting(state: AgentState) -> dict:
    print("📍 Node: should_continue_chatting")

    name = state.get("form_data", {}).get("name")
    datetime_str = state.get("form_data", {}).get("datetime_str")

    # ✅ Only continue to booking if we have both
    if not (name and datetime_str):
        print("🛑 Missing info — keep chatting.")
        return {"next": "chat"}

    recent_messages = state["messages"][-3:]
    response = classifier_llm.invoke([
        SystemMessage(content="""
            You are deciding the next action in a conversation.

            Reply ONLY with: 'schedule_call' or 'chat'.

            Respond with 'schedule_call' ONLY if the user clearly asks to schedule, book, or set up a time to talk.

            Examples of 'schedule_call':
            - "Can I schedule a call?"
            - "I'd like to talk to someone."
            - "How do I book a time?"

            If the user is asking questions, chatting, or learning more, respond with 'chat'.
        """)
    ] + recent_messages)

    decision = response.content.strip().lower()
    print(f"🔍 LLM decision: {decision}")
    return {"next": "schedule_call"} if decision == "schedule_call" else {"next": "chat"}