from typing import TypedDict, List
from langgraph.graph import MessagesState

class AgentState(MessagesState):
    question: str
    context: str
    rewritten_question: str