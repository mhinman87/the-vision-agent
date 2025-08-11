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
from llm_config import llm, llm_with_tools
from nodes.keep_chatting import should_continue_chatting
from nodes.booking import run_booking_tool
from nodes.lookup import lookup_appointment


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

    # Only initialize if not already present
    if "form_data" not in state:
        state["form_data"] = {}

    # Get the most recent user message
    last_user_message = ""
    for msg in reversed(state["messages"]):
        if msg.type == "human":
            last_user_message = msg.content
            break

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
        print("🧠 LLM extracted:\n" + raw)

        for line in raw.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower()
                value = value.strip()
                if key in ["name", "datetime_str", "business_name", "address", "phone", "email"]:
                    state["form_data"][key] = value

        print("🎒 Updated form_data:", state["form_data"])

    except Exception as e:
        print(f"⚠️ Extraction failed: {e}")

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

