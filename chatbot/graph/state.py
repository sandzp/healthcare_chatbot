from typing import List, Dict, Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class GraphState(TypedDict):
    """State schema for the healthcare analytics chatbot graph."""
    input: str
    answer: str
    messages: Annotated[List[BaseMessage], add_messages]
    cohorts: Dict[str, str]
    is_relevant: bool
    is_cohort: bool
