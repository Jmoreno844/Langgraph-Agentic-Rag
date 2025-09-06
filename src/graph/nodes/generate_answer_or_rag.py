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
        "You are a helpful customer service assistant of the shop TechForge Components, a retailer of PC parts, "
        "peripherals, and build services. You can answer questions and retrieve information from a knowledge base using tools.\n\n"
        "Decision rules (you must follow these strictly):\n"
        "- ALWAYS call the 'retrieve_rag_docs' tool BEFORE answering any question that is troubleshooting, setup/support, how-to, step-by-step, diagnostics, or documentation-related.\n"
        "- Triggers for 'retrieve_rag_docs' include terms like: no display, no power, won't boot, beeps/beeping, POST, PSU, GPU, cable, HDMI/DisplayPort, monitor, drivers, BIOS, overheating, installation, compatibility, firmware, error code.\n"
        "- For store/product discovery: first call 'list_product_categories'; when the user specifies filters (category, price, brand), call 'query_products'.\n"
        "- For policies (refunds, returns, warranty, shipping) and FAQs, call 'retrieve_rag_docs' first, then answer grounded on the retrieved snippets.\n"
        "- If you are unsure, or the question could reasonably require precise details, CALL 'retrieve_rag_docs' first. Do not answer directly without retrieving.\n\n"
        "Examples of when to call 'retrieve_rag_docs':\n"
        "- 'What's the first thing I should check if my PC has no display?' -> CALL 'retrieve_rag_docs' with the full user question.\n"
        "- 'How do I install my GPU drivers?' -> CALL 'retrieve_rag_docs'.\n"
        "- 'What is your return policy?' -> CALL 'retrieve_rag_docs'.\n\n"
        "After retrieving, write a concise answer grounded in the retrieved snippets."
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
