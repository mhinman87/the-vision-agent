from state.agent_state import AgentState
from llm_config import classifier_llm
from langchain_core.messages import SystemMessage


def should_continue_chatting(state: AgentState) -> dict:
    print("📍 Node: should_continue_chatting")
    print(f"🔍 DEBUG: Full state form_data: {state.get('form_data', {})}")
    print(f"🔍 DEBUG: State ID: {id(state)}")
    print(f"🔍 DEBUG: State keys: {list(state.keys())}")
    print(f"🔍 DEBUG: form_data type: {type(state.get('form_data'))}")
    print(f"🔍 DEBUG: form_data ID: {id(state.get('form_data', {}))}")
    print(f"🔍 DEBUG: form_data content: {state.get('form_data', {})}")

    # First, determine what the user wants to do
    recent_messages = state["messages"][-3:]
    response = classifier_llm.invoke([
        SystemMessage(content="""
            You are deciding the next action in a conversation.

            Reply ONLY with: 'schedule_call', 'lookup_appointment', or 'chat'.

            Respond with 'schedule_call' ONLY if the user clearly asks to schedule, book, or set up a time to talk.

            Respond with 'lookup_appointment' if the user asks to:
            - Look up their appointment
            - Check their scheduled time
            - Find their booking
            - See when their meeting is
            - "What time is my appointment?"
            - "When am I scheduled?"

            Examples of 'schedule_call':
            - "Can I schedule a call?"
            - "I'd like to talk to someone."
            - "How do I book a time?"

            Examples of 'lookup_appointment':
            - "What time is my appointment?"
            - "When am I scheduled?"
            - "Can you look up my booking?"
            - "What's my meeting time?"

            If the user is asking questions, chatting, or learning more, respond with 'chat'.
        """)
    ] + recent_messages)

    decision = response.content.strip().lower()
    print(f"🔍 LLM decision: {decision}")
    
    # Now check requirements for each action
    if decision == "schedule_call":
        # For scheduling, we need both name and datetime
        name = state.get("form_data", {}).get("name")
        datetime_str = state.get("form_data", {}).get("datetime_str")
        
        if not (name and datetime_str):
            print("🛑 Missing info for scheduling — keep chatting.")
            state["next"] = "chat"
        else:
            state["next"] = "schedule_call"
            
    elif decision == "lookup_appointment":
        # For lookup, we need either name or email
        name = state.get("form_data", {}).get("name")
        email = state.get("form_data", {}).get("email")
        
        if not (name or email):
            print("🛑 Missing name or email for lookup — keep chatting.")
            state["next"] = "chat"
        else:
            state["next"] = "lookup_appointment"
            
    else:
        state["next"] = "chat"
    
    return state