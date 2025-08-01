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




chat_sessions = defaultdict(list)



# --- Define the agent state ---
from typing import TypedDict, List, Optional

from pydantic import BaseModel

class MessageRequest(BaseModel):
    message: str


class AgentState(TypedDict):
    messages: List[BaseMessage]  # 👈 Not dicts anymore
    classification: Optional[str]
    next_action: Optional[str] 
    form_data: Optional[dict] 






# def decide_next_action(state: AgentState) -> AgentState:
#     recent_messages = state["messages"][-3:]  # just a small chunk
#     response = llm_with_tools.invoke([
#         {"role": "system", "content": "Decide the user's intent. If they have clearly asked to book a meeting and provided details, set next_action to 'book_call'. Otherwise, set to 'chat'."},
#         *recent_messages
#     ])
#     if "book_call" in response.content.lower():
#         state["next_action"] = "book_call"
#     else:
#         state["next_action"] = "chat"
#     return state

def should_continue_chatting(state: AgentState) -> dict:
    print("📍 Node: should_continue_chatting")
    recent_messages = state["messages"][-3:]

    response = llm_with_tools.invoke([
        {"role": "system", "content": """
            You are deciding the next action for the user conversation. If the user has clearly asked to schedule a call *and* provided a date/time (or complete scheduling info), return 'schedule_call'. Otherwise, return 'chat'.
            Respond only with the keyword: 'schedule_call' or 'chat'.
            """}
                ] + recent_messages)

    decision = response.content.strip().lower()
    print(f"🔍 LLM decision: {decision}")
    return {"next": "schedule_call"} if "schedule_call" in decision else {"next": "chat"}




# --- Define the LLM chat node ---
llm = ChatOpenAI(model="gpt-4o")
tools = [create_calendar_event]
llm_with_tools = llm.bind_tools(tools)

def chat_with_user(state: AgentState) -> AgentState:
    print("📍 Node: chat_with_user")
    response = llm_with_tools.invoke(state["messages"])
    state["messages"].append(AIMessage(content=response.content))
    return state

# --- Graph setup ---
builder = StateGraph(AgentState)

#---- Nodes ----
booking_tool_node = ToolNode([create_calendar_event])
builder.add_node("chat", chat_with_user)
# builder.add_node("schedule_call", alfred_booking_tool)
builder.add_node("schedule_call", booking_tool_node)
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
async def vision_chat(body: dict = Body(...)):
    session_id = body.get("session_id", "default")
    user_input = body["message"]

    # Add user message to session history
    chat_sessions[session_id].append({"role": "user", "content": user_input})

    system_message = {
        "role": "system",
        "content": """You are Alfred, the helpful AI assistant for GhostStack — a company that builds custom AI agents for small businesses. Your job is to greet visitors, understand their needs, and explain how GhostStack can help automate workflows using advanced AI.

            GhostStack offers:
            - Prebuilt agents for quick deployment (like email sorting, contract review, or lead qualification)
            - Fully custom agents tailored to a business’s unique tools or APIs
            - Full-service integration (including backend APIs and frontend chat interfaces)

            Your tone is clear, calm, and technically competent. You are not pushy or salesy. You are conversational, but focused. If someone asks about pricing, you explain that pricing starts with a setup fee followed by a low monthly subscription, and they can request a custom quote for more complex work.

            Always refer to yourself as “Alfred.” Never say you are ChatGPT or built by OpenAI.

            If you don’t know the answer, offer to collect the user’s contact info and promise a human will follow up.
            """
    }

    state = {
        "messages": [system_message] + chat_sessions[session_id]
    }

    updated_state = graph.invoke(state)

    # Add Alfred's reply to session
    reply = updated_state["messages"][-1]["content"]
    chat_sessions[session_id].append({"role": "assistant", "content": reply})

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

