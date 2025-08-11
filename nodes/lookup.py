# nodes/lookup.py
from state.agent_state import AgentState
from tools.calendar import get_upcoming_event
from langchain_core.messages import AIMessage

def lookup_appointment(state: AgentState) -> AgentState:
    print("📍 Node: lookup_appointment")

    email = state["form_data"].get("email")
    name = state["form_data"].get("name")

    if not email and not name:
        state["messages"].append(AIMessage(content="I need your name or email to look up your appointment."))
        return state

    result = get_upcoming_event(name=name, email=email)

    if result:
        message = f"You're scheduled for {result['start_time']} — let me know if you need to reschedule."
    else:
        message = "I couldn’t find any upcoming appointments under your name or email."

    state["messages"].append(AIMessage(content=message))
    return state
