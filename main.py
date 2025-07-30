from dotenv import load_dotenv
load_dotenv()


# main.py
import os
from langgraph.graph import StateGraph, END
from langchain_community.chat_models import ChatOpenAI
# from langgraph.checkpoint.sqlite import SqliteSaver
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from collections import defaultdict

chat_sessions = defaultdict(list)



# --- Define the agent state ---
from typing import TypedDict, List, Optional

from pydantic import BaseModel

class MessageRequest(BaseModel):
    message: str


class AgentState(TypedDict):
    messages: List[dict]
    classification: Optional[str]

# --- Define basic tools ---
def classify_intent(state: AgentState) -> AgentState:
    last_message = state["messages"][-1]["content"]
    # simple mock classifier
    if "contract" in last_message.lower():
        classification = "Contract Review Agent"
    elif "estimate" in last_message.lower():
        classification = "Estimating Agent"
    else:
        classification = "General Inquiry"
    print(f"[🧠 CLASSIFIED] {classification}")
    return {**state, "classification": classification}

# --- Define the LLM chat node ---
llm = ChatOpenAI(model="gpt-4o")

def chat_with_user(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    print(f"[🤖 VISION]: {response.content}")
    state["messages"].append({"role": "assistant", "content": response.content})
    return state

# --- Graph setup ---
builder = StateGraph(AgentState)
builder.add_node("chat", chat_with_user)
builder.add_node("classify", classify_intent)

builder.set_entry_point("chat")
builder.add_edge("chat", "classify")
builder.add_edge("classify", END)

graph = builder.compile()

# --- Run loop in terminal ---
if __name__ == "__main__":
    state = {"messages": []}

    print("🎯 Talk to The Vision (type 'exit' to quit)")
    while True:
        user_input = input("👤 You: ")
        if user_input.lower() == "exit":
            break
        state["messages"].append({"role": "user", "content": user_input})
        state = graph.invoke(state)



# Code for deploying The Vision

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

    state = {
        "messages": chat_sessions[session_id]
    }

    updated_state = graph.invoke(state)

    # Add Alfred's reply to session
    reply = updated_state["messages"][-1]["content"]
    chat_sessions[session_id].append({"role": "assistant", "content": reply})

    return {"reply": reply}

