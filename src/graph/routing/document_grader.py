from typing import Literal

def document_grader(state) -> Literal["generate_answer", "rewrite_question"]:
    """Stub document grader - always returns generate_answer for now."""
    if state.get("has_been_rewritten"):
        return "generate_answer"
    else:
        # For now, always assume documents are relevant
        return "generate_answer"
