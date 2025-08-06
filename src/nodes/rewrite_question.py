from langgraph.graph import MessagesState
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from src.state import CustomMessagesState

load_dotenv()

response_model = init_chat_model("openai:gpt-4.1", temperature=0)

REWRITE_PROMPT = (
    "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
    "Here is the initial question:"
    "\n ------- \n"
    "{question}"
    "\n ------- \n"
    "Formulate an improved question:"
)


def rewrite_question(state: CustomMessagesState) -> CustomMessagesState:
    """Rewrite the original user question."""
    messages = state["messages"]
    question = messages[0].content
    prompt = REWRITE_PROMPT.format(question=question)
    response = response_model.invoke([HumanMessage(content=prompt)])
    return {"messages": [response], "has_been_rewritten": True}
