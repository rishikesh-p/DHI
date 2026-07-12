from typing import TypedDict, Annotated, Union, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
    input_text: str          # What the user said
    plan: str                # The LLM's explanation (or Router's decision)
    command: str             # The specific Bash command to run
    command_output: str      # What happened when we ran it
    error: Union[str, None]  # Any error messages
    retry_count: int         # How many times we tried to fix bugs
    route_confidence: float  # How confident the router was (0.0 - 1.0)
