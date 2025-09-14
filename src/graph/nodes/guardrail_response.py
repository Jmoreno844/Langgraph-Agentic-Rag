from langchain_core.messages import AIMessage

# Stub response model
class StubResponseModel:
    async def ainvoke(self, messages):
        return AIMessage(content="This is a stub response. Please configure a proper LLM model.")

response_model = StubResponseModel()

PROMPT_TEMPLATE = (
    "The user's request is out of scope for this assistant.\n"
    "Allowed topics: {allowed_topics}.\n"
    "User message: {user_message}\n\n"
    "Write a brief, friendly response (max 2 sentences) that:\n"
    "- States the assistant's scope clearly\n"
    "- Invites the user to ask about an allowed topic\n"
    "- Optionally suggests a few example questions from the allowed topics."
)

async def guardrail_response(state):
    if not state.get("messages"):
        return {}
    user_text = state["messages"][-1].content
    prompt = PROMPT_TEMPLATE.format(
        allowed_topics="PC parts, peripherals, build services",
        user_message=user_text,
    )
    ai = await response_model.ainvoke([{"role": "user", "content": prompt}])
    return {"messages": [ai]}
