from langgraph.graph import MessagesState
from langchain_core.messages import SystemMessage
from langchain.chat_models import init_chat_model
from src.graph.tools.hybrid_retriever import retriever_tool
from dotenv import load_dotenv

from src.graph.state import CustomMessagesState
from src.graph.tools.query_products import query_products_tool
from src.graph.tools.list_product_categories import list_product_categories_tool

load_dotenv()

# Initialize the chat model and retriever tool
response_model = init_chat_model(
    "openai:gpt-4o-mini",
    temperature=0,
    streaming=True,
)


async def generate_answer_or_rag(state: CustomMessagesState) -> CustomMessagesState:
    """Call the model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply respond to the user.
    """
    prompt = (
        "You are a helpful customer service assistant  of the shop TechForge Components, a retailer of PC parts, peripherals, and build services and you can answer questions and retrieve information from a knowledge base. "
        "For general availability questions like 'what products do you have', first call the 'list_product_categories' tool "
        "to provide available categories and ask a follow-up question. When the user specifies filters (category, price, etc.), "
        "use 'query_products'."
        "For policy, refund, returns, warranty, shipping, FAQs, setup/support, or documentation questions, call the 'retrieve_rag_docs' tool with the user's question to ground your answer. "
        "For general messages not related to the customer service of the shop, respond directly without calling the tools"
    )
    if "has_been_rewritten" in state and state["has_been_rewritten"]:
        response = await response_model.bind_tools(
            [retriever_tool, query_products_tool, list_product_categories_tool]
        ).ainvoke(
            state["messages"],
        )

        has_been_rewritten = state["has_been_rewritten"]
    else:
        response = await response_model.bind_tools(
            [retriever_tool, query_products_tool, list_product_categories_tool]
        ).ainvoke([SystemMessage(content=prompt)] + state["messages"])
        has_been_rewritten = False
    return {"messages": [response], "has_been_rewritten": has_been_rewritten}
