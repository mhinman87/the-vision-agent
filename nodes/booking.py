from state.agent_state import AgentState
from langchain_core.messages import AIMessage
from tools.calendar import parse_datetime_with_llm, create_calendar_event

def run_booking_tool(state: AgentState) -> AgentState:
    print("📍 Node: run_booking_tool")
    form_data = state.get("form_data", {})

    # Gather fields
    name = form_data.get("name")
    datetime_str = form_data.get("datetime_str")
    business_name = form_data.get("business_name")
    address = form_data.get("address")
    phone = form_data.get("phone")
    email = form_data.get("email")

    # Check for required fields
    missing = []
    if not name:
        missing.append("name")
    if not datetime_str:
        missing.append("date & time")
    if not business_name:
        missing.append("business name")
    if not address:
        missing.append("address")
    if not phone:
        missing.append("phone")
    if not email:
        missing.append("email")

    if missing:
        msg = "Before I can book your appointment, I still need: " + ", ".join(missing)
        print(f"🛑 Missing fields: {missing}")
        state["messages"].append(AIMessage(content=msg))
        return state

    # ⏰ Parse and validate datetime
    parsed_datetime = parse_datetime_with_llm(datetime_str)

    if not parsed_datetime:
        state["messages"].append(AIMessage(content="I couldn’t understand that date and time. Could you rephrase it?"))
        return state

    # 🕙 Only allow appointments between 10 AM and 4 PM
    if not (10 <= parsed_datetime.hour < 16):
        state["messages"].append(AIMessage(
            content="I can only book appointments between 10 AM and 4 PM. Can you suggest a different time?"
        ))
        return state

    # ✅ Overwrite datetime_str with validated ISO string
    form_data["datetime_str"] = parsed_datetime.isoformat()

    # 📆 Try booking
    try:
        result = create_calendar_event(
            datetime_str=form_data["datetime_str"],
            name=name,
            business_name=business_name,
            address=address,
            phone=phone,
            email=email
        )
        print(f"📆 Tool result: {result}")
        state["messages"].append(AIMessage(content=result))

        # # 🧠 Save appointment globally
        # appointments_by_name[name] = {
        #     "datetime_str": form_data["datetime_str"],
        #     "business_name": business_name,
        #     "address": address,
        #     "contact": phone or email
        # }
        # print(f"🗃️ Saved appointment for {name}: {appointments_by_name[name]}")

        # 🎒 Clear backpack
        print("🧹 Clearing form_data after booking")
        state["form_data"] = {}
        print("🎒 form_data after clearing:", state["form_data"])

    except Exception as e:
        print(f"❌ Tool failed: {str(e)}")
        state["messages"].append(AIMessage(content="Sorry, I had trouble scheduling the event."))

    return state