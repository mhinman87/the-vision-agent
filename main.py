from dotenv import load_dotenv
load_dotenv()


# main.py
import os
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
# from langgraph.checkpoint.sqlite import SqliteSaver
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from collections import defaultdict
from tools.calendar import create_calendar_event
from langgraph.prebuilt import ToolNode
from langgraph.graph import MessagesState
from langchain_core.messages import BaseMessage 
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage
from langchain_core.messages import SystemMessage
from typing import Dict
import json
from typing import TypedDict, List, Optional

from pydantic import BaseModel









# --- Define the agent state ---


class MessageRequest(BaseModel):
    message: str


class AgentState(TypedDict):
    messages: List[BaseMessage]
    classification: Optional[str]
    next_action: Optional[str]
    form_data: Dict[str, Optional[str]] 


chat_sessions: Dict[str, AgentState] = defaultdict(lambda: {
    "messages": [SystemMessage(content="""You are Alfred, the helpful AI assistant for GhostStack — a company that builds custom AI agents for small businesses.

    Role:
    - Greet visitors and explain what GhostStack does.
    - Help them understand how AI can solve real problems in their business.
    - Listen to their pain points and suggest practical automation solutions.
    - Offer to schedule a quick call with Max (the founder) when appropriate.

    Ghost Stack:
    - Prebuilt AI agents (email sorting, contract review, lead qualification, etc.)
    - Fully custom agents that integrate with a business’s APIs or tools
    - Full-service setup: frontend chat, backend logic, calendar, email, and more

    Style:
    - Clear, brief, and confident. No long paragraphs.
    - Technically competent, but never overly casual or robotic.
    - Never guess or hallucinate info — ask for clarification if needed.
    - Never answer general knowledge questions — redirect back to business problems and GhostStack services.

    Never say you're ChatGPT or mention OpenAI.
    Only talk about GhostStack and how it can help small businesses automate workflows using AI.
    Keep your responses concise — no more than 1–2 sentences. You are technically competent and clear, not verbose or chatty. Avoid long explanations unless the user directly asks.


    """)],
    "classification": None,
    "next_action": None,
    "form_data": {}
})




def should_continue_chatting(state: AgentState) -> dict:
    print("📍 Node: should_continue_chatting")
    recent_messages = state["messages"][-3:]

    response = classifier_llm.invoke([
        SystemMessage(content="""
        You are deciding the next action for the user conversation.

        If the user has shown interest in scheduling a call (even without full details like name or time), return 'schedule_call'. Otherwise, return 'chat'.

        Respond ONLY with the word: 'schedule_call' or 'chat'.
                      
        Most times you are going to reply with 'chat' - probably 95 out of 100 times you will respond with 'chat'
                      
        Only time you will respond with 'schedule_call' is if the user is asking to get in touch with a human
    """)
    ] + recent_messages)

    decision = response.content.strip().lower()
    print(f"🔍 LLM decision: {decision}")
    return {"next": "schedule_call"} if "schedule_call" in decision else {"next": "chat"}

def run_booking_tool(state: AgentState) -> AgentState:
    print("📍 Node: run_booking_tool")
    try:
        name = state["form_data"].get("name")
        datetime_str = state["form_data"].get("datetime_str")
        print(f"✅ Booking event for {name} at {datetime_str}")
        result = create_calendar_event(datetime_str, name)
        print(f"📆 Tool result: {result}")
        state["messages"].append(AIMessage(content=result))
    except Exception as e:
        print(f"❌ Tool failed: {str(e)}")
        state["messages"].append(AIMessage(content="Sorry, I had trouble scheduling the event."))
    return state



# --- Define the LLM chat node ---
llm = ChatOpenAI(model="gpt-4o")
tools = [create_calendar_event]
llm_with_tools = llm.bind_tools(tools)
classifier_llm = ChatOpenAI(model="gpt-4o") 

def chat_with_user(state: AgentState) -> AgentState:
    print("📍 Node: chat_with_user")
    response = llm_with_tools.invoke(state["messages"])

    if response.tool_calls:
        # You can handle tool calls here if needed
        print("🛠️ Tool call detected — skipping extraction.")
        ai_msg = response.content.strip()
        state["messages"].append(AIMessage(content=ai_msg))
        return state

    # --- Default message behavior ---
    ai_msg = response.content.strip()
    state["messages"].append(AIMessage(content=ai_msg))

    # --- Try extracting booking info even if no tool was called ---
    extraction_prompt = [
        SystemMessage(content="""
            Extract name and datetime from the conversation. 
            Respond in JSON like:
            {"name": "Max", "datetime_str": "August 3rd at 5p"}

            Use null for any missing values.
        """)
    ] + state["messages"][-3:]

    try:
        extraction_response = classifier_llm.invoke(extraction_prompt)
        extracted = json.loads(extraction_response.content)
        print(f"🔍 Extracted info: {extracted}")

        # Initialize form_data if needed
        state["form_data"] = state.get("form_data", {})

        for key in ("name", "datetime_str"):
            if extracted.get(key):
                state["form_data"][key] = extracted[key]

        name = state["form_data"].get("name")
        datetime_str = state["form_data"].get("datetime_str")

        if name and datetime_str:
            result = create_calendar_event(name=name, datetime_str=datetime_str)
            state["messages"].append(AIMessage(content=result))
        elif name or datetime_str:
            missing = []
            if not name:
                missing.append("your name")
            if not datetime_str:
                missing.append("date and time")
            ask_msg = f"No problem! I just need {', and '.join(missing)} to book your appointment."
            state["messages"].append(AIMessage(content=ask_msg))
        else:
            print("ℹ️ Not enough info to attempt booking yet.")

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


builder.set_entry_point("chat")
builder.add_edge("chat", "should_continue_chatting")
builder.add_conditional_edges(
    "should_continue_chatting",
    lambda state: state["next"],
    {
        "schedule_call": "schedule_call",
        "chat": END  
    }
)
builder.add_edge("schedule_call", END)



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

from fastapi import Body

@app.post("/vision")
async def vision_chat(request: Request, body: dict = Body(...)):
    session_id = body.get("session_id") or request.client.host
    user_input = body["message"]

    # Retrieve per-session AgentState
    state = chat_sessions[session_id]

    # Append user message to AgentState
    state["messages"].append(HumanMessage(content=user_input))

    # Run the graph
    updated_state = graph.invoke(state)

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
    return {"status": "✅ Authorization complete — Alfred can now access your calendar."}



@app.get("/test-calendar")
def test_event():
    return create_calendar_event({
    "title": "Alfred Test Call",
    "datetime": "2025-08-01T15:00:00",
    "description": "Created from /test-calendar endpoint"
        })

