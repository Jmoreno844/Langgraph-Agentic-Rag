from langgraph.graph import MessagesState
from langchain_core.messages import SystemMessage
from langchain.chat_models import init_chat_model
from src.tools.retriever import retriever_tool
from dotenv import load_dotenv

from src.state import CustomMessagesState

load_dotenv()

# Initialize the chat model and retriever tool
response_model = init_chat_model("openai:gpt-4.1", temperature=0)


def generate_answer_or_rag(state: CustomMessagesState) -> CustomMessagesState:
    """Call the model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply respond to the user.
    """
    prompt = "You are a helpful assistant that can answer questions and retrieve information from a knowledge base."
    if "has_been_rewritten" in state and state["has_been_rewritten"]:
        response = response_model.bind_tools([retriever_tool]).invoke(state["messages"])
        has_been_rewritten = state["has_been_rewritten"]
    else:
        response = response_model.bind_tools([retriever_tool]).invoke(
            [SystemMessage(content=prompt)] + state["messages"]
        )
        has_been_rewritten = False
    return {"messages": [response], "has_been_rewritten": has_been_rewritten}
