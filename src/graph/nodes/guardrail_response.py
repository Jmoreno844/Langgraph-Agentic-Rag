from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage
from src.graph.state import CustomMessagesState
from src.graph.nodes.guardrail import ALLOWED_TOPICS

response_model = init_chat_model(
    "openai:gpt-4o-mini",
    streaming=True,
    temperature=0,
)

PROMPT_TEMPLATE = (
    "The user's request is out of scope for this assistant.\n"
    "Allowed topics: {allowed_topics}.\n"
    "User message: {user_message}\n\n"
    "Write a brief, friendly response (max 2 sentences) that:\n"
    "- States the assistant's scope clearly\n"
    "- Invites the user to ask about an allowed topic\n"
    "- Optionally suggests a few example questions from the allowed topics."
)


def guardrail_response(state: CustomMessagesState) -> CustomMessagesState:
    if not state.get("messages"):
        return {}
    user_text = state["messages"][-1].content
    prompt = PROMPT_TEMPLATE.format(
        allowed_topics=", ".join(ALLOWED_TOPICS),
        user_message=user_text,
    )
    ai = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [AIMessage(content=ai.content)]}
