from typing import Sequence
from langchain_core.messages import BaseMessage, trim_messages
import config

def estimate_tokens(messages: Sequence[BaseMessage]) -> int:
    """Roughly estimate tokens across a sequence of messages (approx. 4 chars per token)."""
    total_chars = sum(len(str(m.content)) for m in messages)
    return max(1, total_chars // 4)

def prune_context_window(
    messages: Sequence[BaseMessage],
    max_tokens: int = config.MAX_CONTEXT_TOKENS
) -> tuple[list[BaseMessage], int, bool]:
    """
    Applies the Context Window Trimmer guard:
    - Prunes oldest turns using strategy='last' when exceeding budget
    - Anchors system prompt using include_system=True
    - Returns (trimmed_messages, current_token_estimate, was_trimmed)
    """
    initial_count = len(messages)
    
    # Run langchain_core's trim_messages
    trimmed = trim_messages(
        messages=list(messages),
        max_tokens=max_tokens,
        token_counter="approximate",
        strategy="last",
        include_system=True,
        start_on="human",
        allow_partial=False
    )
    
    # If trimmed dropped any user/assistant turns
    was_trimmed = len(trimmed) < initial_count
    current_tokens = estimate_tokens(trimmed)
    
    return trimmed, current_tokens, was_trimmed
