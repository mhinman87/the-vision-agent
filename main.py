from dotenv import load_dotenv
load_dotenv()


# main.py
import os
from langgraph.graph import StateGraph, END

# from langgraph.checkpoint.sqlite import SqliteSaver
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from collections import defaultdict
from tools.calendar import create_calendar_event
from langgraph.prebuilt import ToolNode
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage
from langchain_core.messages import SystemMessage
from typing import Dict
from state.agent_state import AgentState
from llm_config import llm, llm_with_tools, classifier_llm
from nodes.booking import run_booking_tool


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
    "form_data": {}
})


# --- Define the LLM chat node ---


def chat_with_user(state: AgentState) -> AgentState:
    print("📍 Node: chat_with_user")

    # Generate Alfred's response
    response = llm_with_tools.invoke(state["messages"])
    ai_msg = response.content.strip()
    state["messages"].append(AIMessage(content=ai_msg))
    
    print(f"🤖 Alfred said: {ai_msg}")

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
        extract_response = llm_with_tools.invoke(extract_prompt)
        raw = extract_response.content.strip()
        print(f"🔍 Form data extracted from user message: {raw}")

        for line in raw.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower()
                value = value.strip()
                if key in ["name", "datetime_str", "business_name", "address", "phone", "email"]:
                    state["form_data"][key] = value

        print(f"📝 Form data updated: {state['form_data']}")

    except Exception as e:
        print(f"⚠️ Extraction failed: {e}")

    return state


def should_continue_chatting(state: AgentState) -> dict:
    print("📍 Node: should_continue_chatting")
    print("🚨 NEW CODE IS RUNNING - VERSION 2.0!")
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


# --- Graph setup ---
builder = StateGraph(AgentState)

#---- Nodes ----
#booking_tool_node = ToolNode([create_calendar_event])
builder.add_node("chat", chat_with_user)
# builder.add_node("schedule_call", alfred_booking_tool)
builder.add_node("schedule_call", run_booking_tool)
builder.add_node("should_continue_chatting", should_continue_chatting)
builder.add_node("lookup_appointment", lookup_appointment)


builder.set_entry_point("chat")
builder.add_edge("chat", "should_continue_chatting")
builder.add_conditional_edges(
    "should_continue_chatting",
    lambda state: state["next"],
    {
        "schedule_call": "schedule_call",
        "lookup_appointment": "lookup_appointment",
        "chat": END  
    }
)
builder.add_edge("schedule_call", END)
builder.add_edge("lookup_appointment", END)



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
    print(f"🔍 DEBUG: Graph returned state keys: {list(updated_state.keys())}")
    print(f"🔍 DEBUG: Graph returned form_data: {updated_state.get('form_data', {})}")

    # Save the updated state back to the session
    chat_sessions[session_id] = updated_state

    # Append assistant reply
    reply = updated_state["messages"][-1].content
    chat_sessions[session_id]["messages"].append(AIMessage(content=reply))

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

