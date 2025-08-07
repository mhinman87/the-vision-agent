from typing import TypedDict, List, Optional, Dict
from langchain_core.messages import BaseMessage 

class AgentState(TypedDict):
    messages: List[BaseMessage]
    classification: Optional[str]
    next_action: Optional[str]
    form_data: Dict[str, Optional[str]] 