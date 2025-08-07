from langchain_openai import ChatOpenAI
from tools.calendar import create_calendar_event

llm = ChatOpenAI(model="gpt-4o")
tools = [create_calendar_event]
llm_with_tools = llm.bind_tools(tools)
classifier_llm = ChatOpenAI(model="gpt-4o") 