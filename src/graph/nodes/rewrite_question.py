from langchain_core.messages import AIMessage

def rewrite_question(state):
    """Stub rewrite question - just marks as rewritten for now."""
    messages = state["messages"]
    question = messages[-2].content
    # For now, just return the original question as rewritten
    response = AIMessage(content=f"Rewritten: {question}")
    return {"messages": [response], "has_been_rewritten": True}
