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
        # Parse the ISO datetime and format it nicely
        try:
            from datetime import datetime
            start_time = datetime.fromisoformat(result['start_time'].replace('Z', '+00:00'))
            formatted_time = start_time.strftime("%A, %B %-d at %-I:%M %p")
            message = f"You're scheduled for {formatted_time} — let me know if you need to reschedule."
        except Exception as e:
            # Fallback to original format if parsing fails
            message = f"You're scheduled for {result['start_time']} — let me know if you need to reschedule."
    else:
        message = "I couldn't find any upcoming appointments under your name or email."

    state["messages"].append(AIMessage(content=message))
    return state
