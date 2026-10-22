from typing import Annotated, Sequence, TypedDict, Any
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class RagObotState(TypedDict):
    """Unified state schema for RagObot's multi-agent graph."""
    # Conversation history with LangGraph message reducer
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
    # Current active query
    query: str
    
    # Retrieved Markdown documentation chunks from ChromaDB
    retrieved_docs: list[dict[str, Any]]
    
    # Decision 1: Relevant evidence available?
    evidence_available: bool
    evidence_reason: str
    
    # Parallel worker node thought/calculation streams
    ollama_output: str
    groq_output: str
    
    # Consensus Arbiter preliminary synthesis
    preliminary_verdict: str
    
    # Decision 2: Answer grounded in sources?
    answer_grounded: bool
    grounding_reason: str
    grounded_answer: str
    
    # Abstention tracking
    is_abstained: bool
    abstention_reason: str
    
    # Final synthesized consensus from Arbiter node
    final_response: str
    arbiter_debate: str
    
    # Diagnostic monitors
    token_count: int
    was_trimmed: bool
    turn_count: int
    
    # Dynamic Model Selection
    model_mode: str
    local_model: str
    cloud_model: str
    
    # Audio dispatch control (False for web clients, True for terminal/CLI)
    trigger_backend_tts: bool
    
    # Conversational context awareness
    contextualized_query: str
    uploaded_doc: str
    target_doc: str
