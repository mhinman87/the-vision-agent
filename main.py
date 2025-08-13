from dotenv import load_dotenv
load_dotenv()


# main.py
import os
import re
from langgraph.graph import StateGraph, END

# from langgraph.checkpoint.sqlite import SqliteSaver
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from collections import defaultdict
from tools.calendar import create_calendar_event, get_upcoming_event, reschedule_appointment, parse_datetime_with_llm, get_available_slots_next_week, check_slot_available
from langgraph.prebuilt import ToolNode
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage
from langchain_core.messages import SystemMessage
from typing import Dict
from state.agent_state import AgentState
from llm_config import llm, llm_with_tools, classifier_llm



from pydantic import BaseModel




class MessageRequest(BaseModel):
    message: str


chat_sessions: Dict[str, AgentState] = defaultdict(lambda: {
    "messages": [SystemMessage(content="""You are Alfred, the helpful AI assistant for GhostStack — a company that builds custom AI agents for small businesses.

    Role:
    - Greet visitors and explain what GhostStack does.
    - Help them understand how AI can solve real problems in their business.
    - Listen to their pain points and suggest practical automation solutions.
    - Offer to schedule a quick call with Max (the founder) when appropriate.
    - Help users look up their existing appointments when they ask.

    Ghost Stack:
    - Prebuilt AI agents (email sorting, contract review, lead qualification, etc.)
    - Fully custom agents that integrate with a business’s APIs or tools
    - Full-service setup: frontend chat, backend logic, calendar, email, and more

    Style:
    - Clear, brief, and confident. No long paragraphs.
    - Technically competent, but never overly casual or robotic.
    - Never guess or hallucinate info — ask for clarification if needed.
    - Never answer general knowledge questions — redirect back to business problems and GhostStack services.
                               
    Availability:
    - Only book appointments Monday through Friday, between 10:00 AM and 4:00 PM Central Time.
    - If a user suggests a time outside of this range, politely explain the availability and ask them to pick a time within that window.

    Capabilities:
    - Schedule new appointments with Max
    - Look up existing appointments using name or email
    - Help with business automation questions


    Never say you're ChatGPT or mention OpenAI.
    Only talk about GhostStack and how it can help small businesses automate workflows using AI.
    Keep your responses concise — no more than 1–2 sentences. You are technically competent and clear, not verbose or chatty. Avoid long explanations unless the user directly asks.


    """)],
    "classification": None,
    "next_action": None,
    "form_data": {},
    "pending_action": None
})


# --- Define the LLM chat node ---


def _get_last_human_message(state: AgentState) -> str:
    last_user_message = ""
    for msg in reversed(state["messages"]):
        if msg.type == "human":
            last_user_message = msg.content
            break
    return last_user_message


def interpret_slot_selection_with_llm(user_message: str, available_slots: list) -> dict:
    """Use the LLM to interpret the user's selection, LLM-first and JSON-free.
    Returns:
      {"type": "iso", "iso": "2025-08-18T10:00:00"} if they picked one of the shown options
      {"type": "datetime", "datetime_str": "monday at 11am"} if they propose a new time
      {"type": "none"} otherwise
    """
    if not user_message or not available_slots:
        return {"type": "none"}

    # Build options text
    options_lines = []
    for i, slot in enumerate(available_slots, 1):
        options_lines.append(f"{i}. {slot['display']} ({slot['datetime']})")
    options_text = "\n".join(options_lines)

    system_prompt = (
        "You help interpret which appointment option the user picked.\n"
        "We have a list of numbered options; each includes an ISO datetime in parentheses.\n"
        "Rules for your reply (no extra text):\n"
        "- If the user picked one of the shown options (by number or restating it), reply with the exact ISO datetime string from that option.\n"
        "- If they proposed a different time (not in the list), reply with: NEW: <their time>\n"
        "- If unclear, reply exactly: NONE"
    )

    human_prompt = (
        f"Options:\n{options_text}\n\n"
        f"User reply: {user_message}\n\n"
        "Respond using the rules above."
    )

    try:
        resp = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)])
        raw = (resp.content or "").strip()
        # Exact ISO match from options
        for slot in available_slots:
            if raw == slot["datetime"]:
                return {"type": "iso", "iso": raw}
        # New time proposal
        if raw.lower().startswith("new:"):
            return {"type": "datetime", "datetime_str": raw.split(":", 1)[1].strip()}
        # None/unclear
        if raw.upper() == "NONE":
            return {"type": "none"}
    except Exception:
        pass

    return {"type": "none"}


def chat_with_user(state: AgentState) -> AgentState:
    print("📍 Node: chat_with_user")

    # Generate Alfred’s response
    response = llm_with_tools.invoke(state["messages"])
    ai_msg = (response.content or "").strip()
    if ai_msg:
        state["messages"].append(AIMessage(content=ai_msg))

    # Only initialize if not already present
    if "form_data" not in state:
        state["form_data"] = {}

    # Get the most recent user message
    last_user_message = ""
    for msg in reversed(state["messages"]):
        if msg.type == "human":
            last_user_message = msg.content
            break

    print(f"👤 User said: {last_user_message}")
    if ai_msg:
        print(f"🤖 Alfred said: {ai_msg}")

    # Prompt to extract field updates
    extract_prompt = [
        SystemMessage(content="""
            You are helping extract appointment booking info from a single user message.

            Return each field found in this format:
            field: value

            Only include fields you actually see in the message. No extra text, no formatting.

            Supported fields:
            - name
            - datetime_str
            - business_name
            - address
            - phone
            - email
            """),
        HumanMessage(content=last_user_message)
    ]

    # Call LLM to extract values and update the backpack
    try:
        # Skip extraction when the user is selecting a numbered slot we just showed
        skip_extraction = False
        if state.get("form_data", {}).get("available_slots"):
            lm = last_user_message.strip().lower()
            if re.search(r"\b(?:number|no\.?|option|slot|choice|pick)\s*(1|2|3)\b", lm) or \
               re.search(r"\b(1|2|3|one|two|three|first|second|third)\b", lm):
                skip_extraction = True

        # Also skip for single digits (1, 2, 3)
        if not ((last_user_message.strip().isdigit() and len(last_user_message.strip()) == 1) or skip_extraction):
            extract_response = llm_with_tools.invoke(extract_prompt)
            raw = extract_response.content.strip()

            for line in raw.splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip().lower()
                    value = value.strip()
                    if key in ["name", "datetime_str", "business_name", "address", "phone", "email"]:
                        state["form_data"][key] = value

            # Heuristic fallback to capture common cases when the LLM doesn't format strictly
            lm_text = last_user_message.strip()
            form_data = state.get("form_data", {})

            # Email
            if "email" not in form_data or not form_data.get("email"):
                em = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", lm_text)
                if em:
                    form_data["email"] = em.group(0)

            # Name via phrases: "my name is X", "I'm X", "I am X"
            if "name" not in form_data or not form_data.get("name"):
                m = re.search(r"\b(?:my name is|i am|i'm)\s+([A-Z][a-zA-Z\-']+(?:\s+[A-Z][a-zA-Z\-']+)?)", lm_text, re.IGNORECASE)
                if m:
                    form_data["name"] = m.group(1).strip()

            # Business name via phrases: "business name is X", "our business is X"
            if "business_name" not in form_data or not form_data.get("business_name"):
                m = re.search(r"\b(?:business name is|our business is|the business is)\s*[:\-]?\s*(.+)$", lm_text, re.IGNORECASE)
                if m:
                    possible_bn = m.group(1).strip().strip(". ")
                    if possible_bn and "@" not in possible_bn:
                        form_data["business_name"] = possible_bn

            # Dash pattern: "LeftPart - rightpart" with email on right; infer left as name or business
            if "-" in lm_text and ("email" in form_data and form_data.get("email")):
                left = lm_text.split("-", 1)[0].strip()
                token_count = len([t for t in re.split(r"\s+", left) if t])
                if left and "@" not in left:
                    if (not form_data.get("business_name") and (form_data.get("name") or token_count > 2)):
                        form_data["business_name"] = left
                    elif (not form_data.get("name") and token_count <= 2):
                        form_data["name"] = left

            state["form_data"] = form_data


    except Exception as e:
        # print(f"⚠️ Extraction failed: {e}")
        pass

    return state


def should_continue_chatting(state: AgentState) -> AgentState:
    print(f"📍 Node: should_continue_chatting")
    
    # First, determine what the user wants to do
    recent_messages = state["messages"][-3:]
    response = classifier_llm.invoke([
        SystemMessage(content="""
            You are deciding the next action in a conversation.

            Reply ONLY with: 'schedule_call', 'lookup_appointment', 'reschedule_appointment', or 'chat'.

            Respond with 'schedule_call' ONLY if the user clearly asks to schedule, book, or set up a time to talk. 
            If the user is just providing information (like a preferred time) without explicitly asking to book, respond with 'chat'.
            
            Respond with 'schedule_call' if the user:
            - Explicitly asks to schedule/book an appointment
            - Provides a time AND asks to book it
            - Says "book me for..." or "schedule me for..."
            
            Respond with 'chat' if the user:
            - Just provides a time without asking to book
            - Is responding to a question about availability
            - Is providing information in conversation
            
            Respond with 'reschedule_appointment' if the user asks to:
            - Reschedule their appointment
            - Change their appointment time
            - Move their meeting to a different time
            - "I need to move that appointment"
            - "Can you reschedule my appointment?"
            - "I need to change my appointment time"
            - "Can we move this to a different time?"
            - "I'd like to reschedule"

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
            - "Book me for Monday at 11am"
            - "Schedule me for Wednesday at 2pm"

            Examples of 'lookup_appointment':
            - "What time is my appointment?"
            - "When am I scheduled?"
            - "Can you look up my booking?"
            - "What's my meeting time?"

            Examples of 'reschedule_appointment':
            - "Can you reschedule my appointment?"
            - "I need to change my appointment time"
            - "Can we move this to a different time?"
            - "I'd like to reschedule"

            If the user is asking questions, chatting, or learning more, respond with 'chat'.
        """)
    ] + recent_messages)

    decision = response.content.strip().lower()
    # Normalize noisy classifier outputs to valid tokens
    for token in ["schedule_call", "lookup_appointment", "reschedule_appointment", "chat"]:
        if token in decision:
            decision = token
            break
    # print(f"🔍 LLM decision: {decision}")
    # Persist intent for downstream nodes
    state["intent"] = decision
    # Persist pending action when an actionable intent appears
    if decision in ("schedule_call", "lookup_appointment", "reschedule_appointment"):
        state["pending_action"] = decision

    # If user just picked a numbered option while classifier says 'chat', still route to availability
    if decision == "chat":
        form_data = state.get("form_data", {})
        available_slots = form_data.get("available_slots")
        last_user_message = _get_last_human_message(state)
        if available_slots:
            choice = interpret_slot_selection_with_llm(last_user_message, available_slots)
            if choice.get("type") in {"index", "datetime"}:
                state["next"] = "check_availability"
                return state
        # Honor sticky pending_action during chat
        if state.get("pending_action") == "schedule_call":
            state["next"] = "check_availability"
            return state
        if state.get("pending_action") == "reschedule_appointment":
            state["next"] = "check_availability"
            return state
        if state.get("pending_action") == "lookup_appointment":
            name = form_data.get("name")
            email = form_data.get("email")
            state["next"] = "lookup_appointment" if (name or email) else "chat"
            return state

    # Now check requirements for each action normally
    if decision == "schedule_call":
        # For scheduling, always check availability first to show available slots
        # print("🎯 User wants to schedule - routing to availability check")
        state["next"] = "check_availability"
            
    elif decision == "lookup_appointment":
        # For lookup, we need either name or email
        name = state.get("form_data", {}).get("name")
        email = state.get("form_data", {}).get("email")
        
        if not (name or email):
            # print("🛑 Missing name or email for lookup — keep chatting.")
            state["next"] = "chat"
        else:
            state["next"] = "lookup_appointment"
            
    elif decision == "reschedule_appointment":
        # For rescheduling, we need name/email, but NOT datetime yet
        name = state.get("form_data", {}).get("name")
        email = state.get("form_data", {}).get("email")
        
        if not (name or email):
            # print("🛑 Missing name or email for rescheduling — keep chatting.")
            state["next"] = "chat"
        else:
            # print("🔄 User wants to reschedule - routing to availability check")
            state["next"] = "check_availability"
            
    else:
        state["next"] = "chat"
    
    return state


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


def reschedule_appointment_node(state: AgentState) -> AgentState:
    print("📍 Node: reschedule_appointment_node")
    
    # Get the current form data
    name = state.get("form_data", {}).get("name")
    email = state.get("form_data", {}).get("email")
    datetime_str = state.get("form_data", {}).get("datetime_str")
    
    #print(f"🔄 Rescheduling appointment for {name or email} to {datetime_str}")
    
    # Call the reschedule tool with the datetime string
    result = reschedule_appointment(
        name=name,
        email=email,
        new_datetime_str=datetime_str
    )
    
    # Add the result to the messages
    state["messages"].append(AIMessage(content=result))
    
    return state


def check_availability_node(state: AgentState) -> AgentState:
    """Node for checking available appointment slots and recommending times."""
    print(f"📍 Node: check_availability")
    
    # Check if this is a rescheduling request (based on explicit intent, not presence of name/email)
    form_data = state.get("form_data", {})
    name = form_data.get("name")
    email = form_data.get("email")
    is_reschedule = (state.get("pending_action") == "reschedule_appointment")
    datetime_str = form_data.get("datetime_str")

    # For new bookings, collect required info BEFORE showing slots, unless a specific time was given
    if not is_reschedule and not datetime_str:
        required_order = ["name", "business_name", "address", "phone", "email"]
        labels = {
            "name": "your name",
            "business_name": "your business name",
            "address": "your address",
            "phone": "your phone number",
            "email": "your email",
        }
        next_field = next((f for f in required_order if not form_data.get(f)), None)
        if next_field:
            response = f"To book your appointment, what's {labels[next_field]}?"
            state["messages"].append(AIMessage(content=response))
            return state
    
    # If rescheduling, show existing appointment first
    if is_reschedule and (name or email):
        # print(f"🔍 Checking for existing appointments for {name or email}")
        result = get_upcoming_event(name=name, email=email)
        
        if result:
            # Parse the ISO datetime and format it nicely
            try:
                from datetime import datetime
                start_time = datetime.fromisoformat(result['start_time'].replace('Z', '+00:00'))
                formatted_time = start_time.strftime("%A, %B %-d at %-I:%M %p")
                existing_appointment_msg = f"📅 You currently have an appointment scheduled for {formatted_time}."
            except Exception as e:
                # Fallback to original format if parsing fails
                existing_appointment_msg = f"📅 You currently have an appointment scheduled for {result['start_time']}."
        else:
            existing_appointment_msg = "📅 I couldn't find any existing appointments under your name or email."
    
    # Prefer previously shown slots if available to keep numbering consistent
    available_slots = form_data.get("available_slots") or get_available_slots_next_week()
    
    if not available_slots:
        response = (
            "❌ I'm having trouble checking my availability right now. "
            "Please try again later or let me know a specific time you'd like to book."
        )
        state["messages"].append(AIMessage(content=response))
        return state
    
    # Format the available slots for display
    slots_text = []
    for i, slot in enumerate(available_slots, 1):
        slots_text.append(f"{i}. {slot['display']}")
    
    slots_display = "\n".join(slots_text)
    
    # Check if we have a datetime_str to validate (may have been set above)
    
    # Check if user selected a slot by natural language via LLM (LLM-first)
    if not datetime_str:
        last_user_message = _get_last_human_message(state)
        llm_choice = interpret_slot_selection_with_llm(last_user_message, available_slots)
        if llm_choice.get("type") == "iso":
            # They picked one of the shown options
            iso = llm_choice.get("iso")
            datetime_str = iso
            form_data["datetime_str"] = datetime_str
            state["form_data"] = form_data
            # Find the display for confirmation
            selected_slot = next((s for s in available_slots if s["datetime"] == iso), None)
            if is_reschedule and (name or email):
                state["next"] = "reschedule_appointment"
                return state
            else:
                display = selected_slot["display"] if selected_slot else "that time"
                # Ask one missing field at a time
                required_order = ["name", "business_name", "address", "phone", "email"]
                labels = {
                    "name": "your name",
                    "business_name": "your business name",
                    "address": "your address",
                    "phone": "your phone number",
                    "email": "your email",
                }
                next_field = next((f for f in required_order if not form_data.get(f)), None)
                if next_field:
                    response = (
                        f"✅ Great choice! I'll schedule you for {display}.\n\n"
                        f"Great — what's {labels[next_field]}?"
                    )
                    state["messages"].append(AIMessage(content=response))
                    return state
                else:
                    return run_booking_tool(state)
        elif llm_choice.get("type") == "datetime":
            datetime_str = llm_choice.get("datetime_str")
    
    if datetime_str:
        # User provided a specific time - check if it's available
        # print(f"🔍 Checking availability for specific slot: {datetime_str}")
        
        parsed_datetime = parse_datetime_with_llm(datetime_str)
        if parsed_datetime:
            iso_datetime = parsed_datetime.isoformat()
            # print(f"✅ Parsed datetime: {datetime_str} → {iso_datetime}")
            
            # Check if this specific slot is available
            availability_result = check_slot_available(iso_datetime)
            
            if availability_result["available"]:
                # If this is a rescheduling request, proceed to reschedule
                if is_reschedule and (name or email):
                    # print("🔄 Rescheduling - proceeding with new time")
                    # Update form_data with the new datetime
                    form_data["datetime_str"] = iso_datetime
                    state["form_data"] = form_data
                    # Route to reschedule_appointment node
                    state["next"] = "reschedule_appointment"
                    return state
                
                # For new bookings, check required fields
                missing = []
                business_name = form_data.get("business_name")
                address = form_data.get("address")
                phone = form_data.get("phone")
                email = form_data.get("email")
                
                if not business_name:
                    missing.append("business name")
                if not address:
                    missing.append("address")
                if not phone:
                    missing.append("phone")
                if not email:
                    missing.append("email")
                
                if missing:
                    # Still need more info
                    response = (
                        f"✅ {availability_result['message']}\n\n"
                        f"Perfect! That time works. To complete your booking, I still need:\n"
                        f"• {', '.join(missing)}\n\n"
                        f"What's your {'business name' if 'business_name' in missing else 'address' if 'address' in missing else 'phone' if 'phone' in missing else 'email'}?"
                    )
                    state["messages"].append(AIMessage(content=response))
                    return state
                else:
                    # We have all the info - proceed to booking
                    # print("🎯 All info collected - proceeding to booking")
                    return run_booking_tool(state)
            else:
                # Slot is not available - show alternatives
                response = (
                    f"❌ {availability_result['message']}\n\n"
                    f"Here are some alternative times that are available:\n\n"
                    f"{slots_display}\n\n"
                    f"**If these times don't work, we are available Monday through Friday, 10 AM - 4 PM Central Time.**\n\n"
                    f"You can:\n"
                    f"• Pick one of these times by saying the number (1, 2, or 3)\n"
                    f"• Suggest a different time within business hours\n\n"
                    f"What would you prefer?"
                )
                state["messages"].append(AIMessage(content=response))
                return state
        else:
            # Couldn't parse the datetime (likely business hours violation)
            response = (
                "❌ I can only schedule appointments Monday through Friday, between 10 AM and 4 PM Central Time.\n\n"
                f"Here are some available times:\n\n{slots_display}\n\n"
                "Please pick one of these times or suggest a different time during business hours."
            )
            state["messages"].append(AIMessage(content=response))
            return state
    
    # No specific time provided - show general availability
    # print("🔍 No specific time provided - showing general availability")
    
    # Include existing appointment info if rescheduling
    if is_reschedule and (name or email):
        response = (
            f"{existing_appointment_msg}\n\n"
            f"🎯 Here are 3 available appointment slots for the next week:\n\n"
            f"{slots_display}\n\n"
            f"**Please note:** I cannot schedule appointments within 2 hours of now.\n"
            f"**If these times don't work, we are available Monday through Friday, 10 AM - 4 PM Central Time.**\n\n"
            f"You can:\n"
            f"• Pick one of these times by saying the number (1, 2, or 3)\n"
            f"• Suggest a different time within business hours\n\n"
            f"What would you prefer?"
        )
    else:
        response = (
            f"🎯 Here are 3 available appointment slots for the next week:\n\n"
            f"{slots_display}\n\n"
            f"**Please note:** I cannot schedule appointments within 2 hours of now.\n"
            f"**If these times don't work, we are available Monday through Friday, 10 AM - 4 PM Central Time.**\n\n"
            f"You can:\n"
            f"• Pick one of these times by saying the number (1, 2, or 3)\n"
            f"• Suggest a different time within business hours\n\n"
            f"What would you prefer?"
        )
    
    # Store the available slots in form_data for later use
    if "form_data" not in state:
        state["form_data"] = {}
    state["form_data"]["available_slots"] = available_slots
    
    state["messages"].append(AIMessage(content=response))
    return state


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
        # print(f"🛑 Missing fields: {missing}")
        state["messages"].append(AIMessage(content=msg))
        return state

    # ⏰ Parse and validate datetime
    parsed_datetime = parse_datetime_with_llm(datetime_str)

    if not parsed_datetime:
        state["messages"].append(AIMessage(content="I couldn't understand that date and time. Could you rephrase it?"))
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
        # print(f"📆 Tool result: {result}")
        state["messages"].append(AIMessage(content=result))

        # 🎒 Keep form data for potential follow-up actions

    except Exception as e:
        # print(f"❌ Tool failed: {str(e)}")
        state["messages"].append(AIMessage(content="Sorry, I had trouble scheduling the event."))

    return state


# --- Graph setup ---
builder = StateGraph(AgentState)

#---- Nodes ----
#booking_tool_node = ToolNode([create_calendar_event])
builder.add_node("chat", chat_with_user)
builder.add_node("should_continue_chatting", should_continue_chatting)
builder.add_node("check_availability", check_availability_node)
builder.add_node("lookup_appointment", lookup_appointment)
builder.add_node("reschedule_appointment", reschedule_appointment_node)


builder.set_entry_point("chat")
builder.add_edge("chat", "should_continue_chatting")
builder.add_conditional_edges(
    "should_continue_chatting",
    lambda state: state["next"],
    {
        "check_availability": "check_availability",
        "lookup_appointment": "lookup_appointment",
        "reschedule_appointment": "reschedule_appointment",
        "chat": END  
    }
)

# Add conditional edges from check_availability
builder.add_conditional_edges(
    "check_availability",
    lambda state: state.get("next", "chat"),
    {
        "reschedule_appointment": "reschedule_appointment",
        "chat": "should_continue_chatting"
    }
)

builder.add_edge("lookup_appointment", END)
builder.add_edge("reschedule_appointment", END)



graph = builder.compile()

# --- Run loop in terminal ---
if __name__ == "__main__":
    state = {"messages": []}

    print("🎯 Talk to The Vision (type 'exit' to quit)")
    while True:
        user_input = input("👤 You: ")
        if user_input.lower() == "exit":
            break
        state["messages"].append(HumanMessage(content=user_input))
        state = graph.invoke(state)



# Code for deploying Alfred

app = FastAPI()

# Allow CORS from your frontend domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace * with your frontend later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import uuid
from fastapi import Body

@app.post("/vision")
async def vision_chat(request: Request, body: dict = Body(...)):
    # Generate unique session ID if not provided
    session_id = body.get("session_id") or str(uuid.uuid4())
    print(f"🔑 Session ID: {session_id}")
    user_input = body["message"]

    # Retrieve per-session AgentState
    state = chat_sessions[session_id]

    # Append user message to AgentState
    state["messages"].append(HumanMessage(content=user_input))

    # Run the graph
    updated_state = graph.invoke(state)

    # Save the updated state back to the session
    chat_sessions[session_id] = updated_state

    # Get the last AI message
    reply = None
    for msg in reversed(updated_state["messages"]):
        if msg.type == "ai":
            reply = msg.content
            break

    if not reply:
        reply = "I'm sorry, I encountered an error processing your request."

    return {"reply": reply}



from fastapi.responses import RedirectResponse
from google_auth_oauthlib.flow import Flow
from tools.calendar import store_token  # Pull in our tool’s token handler

def get_google_flow():
    return Flow.from_client_config(
        {
            "web": {
                "client_id": os.getenv("GOOGLE_CLIENT_ID"),
                "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": ["https://the-vision-agent.onrender.com/oauth2callback"],
            }
        },
        scopes=["https://www.googleapis.com/auth/calendar"],
        redirect_uri="https://the-vision-agent.onrender.com/oauth2callback"
    )

@app.get("/auth")
def authorize_user():
    flow = get_google_flow()
    auth_url, _ = flow.authorization_url(
        prompt="consent", access_type="offline", include_granted_scopes="true"
    )
    print("🔗 Redirecting user to:", auth_url)  # 👈 Add this
    return RedirectResponse(auth_url)


@app.get("/oauth2callback")
def oauth_callback(request: Request):
    print("📥 Callback URL:", str(request.url))  # shows if `?code=...` is present
    flow = get_google_flow()
    flow.fetch_token(authorization_response=str(request.url))
    creds = flow.credentials
    store_token(creds.to_json())
    print("🔐 Token JSON:\n", creds.to_json())
    return {"status": "✅ Authorization complete — Alfred can now access your calendar."}



@app.get("/test-calendar")
def test_event():
    return create_calendar_event({
    "title": "Alfred Test Call",
    "datetime": "2025-08-01T15:00:00",
    "description": "Created from /test-calendar endpoint"
        })

