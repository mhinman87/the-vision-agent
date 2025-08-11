from state.agent_state import AgentState
from llm_config import llm_with_tools
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

def chat_with_user(state: AgentState) -> AgentState:
    print("📍 Node: chat_with_user")

    # Generate Alfred's response
    response = llm_with_tools.invoke(state["messages"])
    ai_msg = response.content.strip()
    state["messages"].append(AIMessage(content=ai_msg))

    # Initialize backpack
    state["form_data"] = state.get("form_data", {})

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

    print("🔍 DEBUG: Final state form_data:", state["form_data"])
    print("🔍 DEBUG: State ID:", id(state))
    
    return state