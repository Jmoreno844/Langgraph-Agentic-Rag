from langchain.chat_models import init_chat_model
from src.graph.state import CustomMessagesState

response_model = init_chat_model(
    "openai:gpt-4o-mini",
    streaming=True,
    temperature=0,
)

GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "Use three sentences maximum and keep the answer concise.\n"
    "Question: {question}\n"
    "Context: {context}"
)


async def generate_answer(state: CustomMessagesState):
    """Generate an answer using context stored in state."""
    question = state.get("current_question", "")
    context = state.get("retrieved_context", "")

    if not question:
        # Fallback to extracting from messages if not in state
        from .extract_context import _latest_user_content

        question = _latest_user_content(state["messages"])

    if not context:
        # Fallback to extracting from messages if not in state
        from .extract_context import _extract_tool_context

        context = _extract_tool_context(state["messages"])

    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = await response_model.ainvoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}
