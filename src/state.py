from typing import Optional, TypedDict, Annotated, Sequence, List 
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AnimeGraphState(TypedDict):
    """"based on the user input have a conversation with the user about travel"""
    messages : Annotated[Sequence[BaseMessage], add_messages]
    user_input: str
    context: Optional[str] = None
