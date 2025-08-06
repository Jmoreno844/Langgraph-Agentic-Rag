from langgraph.graph import MessagesState
from langchain.chat_models import init_chat_model
from src.vectorstores.in_memory import create_in_memory_retriever_tool
from dotenv import load_dotenv

load_dotenv()

# Initialize the chat model and retriever tool
response_model = init_chat_model("openai:gpt-4.1", temperature=0)

retriever_tool = create_in_memory_retriever_tool()


def generate_answer_or_rag(state: MessagesState):
    """Call the model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply respond to the user.
    """
    response = response_model.bind_tools([retriever_tool]).invoke(state["messages"])
    return {"messages": [response]}


input = {
    "messages": [
        {"role": "user", "content": "What is Aetherix Dynamics privacy policy?"}
    ]
}
generate_answer_or_rag(input)["messages"][-1].pretty_print()
