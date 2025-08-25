from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.prebuilt import tools_condition, ToolNode

from src.graph.routing.document_grader import document_grader

from src.graph.nodes.generate_answer_or_rag import generate_answer_or_rag
from src.graph.nodes.generate_answer import generate_answer
from src.graph.nodes.rewrite_question import rewrite_question
from src.graph.tools.retriever import retriever_tool
from src.graph.tools.query_products import query_products_tool
from src.graph.tools.list_product_categories import list_product_categories_tool
from src.graph.state import CustomMessagesState
from src.graph.nodes.guardrail import topic_guardrail
from src.graph.nodes.guardrail_response import guardrail_response

tools = ToolNode([retriever_tool, query_products_tool, list_product_categories_tool])
graph = StateGraph(CustomMessagesState)

graph.add_node("topic_guardrail", topic_guardrail)
graph.add_node("guardrail_response", guardrail_response)
graph.add_node(generate_answer_or_rag)
graph.add_node(generate_answer)
# graph.add_node(rewrite_question)
graph.add_node("tools", tools)

graph.add_edge(START, "topic_guardrail")
graph.add_conditional_edges(
    "topic_guardrail",
    lambda state: "guardrail_response"
    if state.get("blocked_by_guardrail")
    else "generate_answer_or_rag",
    {
        "guardrail_response": "guardrail_response",
        "generate_answer_or_rag": "generate_answer_or_rag",
    },
)
graph.add_conditional_edges(
    "generate_answer_or_rag",
    # Assess LLM decision (call `retriever_tool` tool or respond to the user)
    tools_condition,
    {
        # Translate the condition outputs to nodes in our graph
        "tools": "tools",
        END: END,
    },
)
graph.add_edge("tools", "generate_answer")
graph.add_edge("generate_answer", END)
