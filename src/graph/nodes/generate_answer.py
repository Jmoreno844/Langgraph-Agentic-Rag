from langgraph.graph import MessagesState
from langchain.chat_models import init_chat_model

response_model = init_chat_model(
    "openai:gpt-4o-mini",
    streaming=True,
    temperature=0,
)

BASE_RULES = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "Use three sentences maximum and keep the answer concise.\n"
)

DB_RULES = (
    "When stating price or availability for any product, only use facts from the products tool output. "
    "Append the product ID as [[DB:<id>]] immediately after the claim. "
)


def _build_prompt(question: str, context: str, *, source_db: bool) -> str:
    parts = [BASE_RULES]
    if source_db:
        parts.append(DB_RULES)
    parts.append("Question: ")
    parts.append(f" {question}\n")
    parts.append("Context: ")
    parts.append(context)
    return "".join(parts)


def generate_answer(state: MessagesState):
    """Generate an answer."""
    question = state["messages"][-3].content
    context = state["messages"][-1].content

    # Flags from source_router
    source_db = bool(state.get("source_db", False))

    prompt = _build_prompt(question=question, context=context, source_db=source_db)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}
