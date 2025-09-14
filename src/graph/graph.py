from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState

from src.graph.routing.document_grader import document_grader

from src.graph.nodes.generate_answer_or_rag import generate_answer_or_rag
from src.graph.nodes.generate_answer import generate_answer
from src.graph.nodes.rewrite_question import rewrite_question
from src.graph.nodes.extract_context import extract_context
from src.graph.tools.hybrid_retriever import retriever_tool
from src.graph.tools.query_products import query_products_tool
from src.graph.tools.list_product_categories import list_product_categories_tool
from src.graph.state import CustomMessagesState
from src.graph.guardrails.topic_restriction import topic_guardrail
from src.graph.nodes.guardrail_response import guardrail_response
from src.graph.routing.source_router import source_router
from langchain_core.messages import ToolMessage


def tools_condition(state):
    """Custom tool routing condition to replace langgraph.prebuilt.tools_condition"""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "__end__"


def tool_node(tools_list):
    """Custom tool node to replace langgraph.prebuilt.ToolNode"""

    async def run_tools(state):
        """Execute tools based on the last message's tool calls"""
        last_message = state["messages"][-1]
        if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
            return state

        tool_results_messages = []
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call.get("args", {})
            tool_id = tool_call.get("id")

            # Execute the tool via ainvoke
            if tool_name == "retrieve_rag_docs":
                result = await retriever_tool.ainvoke(tool_args)
            elif tool_name == "query_products":
                result = await query_products_tool.ainvoke(tool_args)
            elif tool_name == "list_product_categories":
                result = await list_product_categories_tool.ainvoke(tool_args)
            else:
                continue

            # Normalize result to string content for ToolMessage
            if isinstance(result, (list, dict)):
                from json import dumps

                content = dumps(result)
            else:
                content = str(result)

            tool_results_messages.append(
                ToolMessage(tool_call_id=tool_id or tool_name, content=content)
            )

        # Append tool messages so downstream nodes can extract context
        return {
            "messages": state["messages"] + tool_results_messages,
        }

    return run_tools


# Create the tool node
tools = tool_node([retriever_tool, query_products_tool, list_product_categories_tool])
graph = StateGraph(CustomMessagesState)

graph.add_node("topic_guardrail", topic_guardrail)
graph.add_node("guardrail_response", guardrail_response)
graph.add_node("generate_answer_or_rag", generate_answer_or_rag)
graph.add_node("generate_answer", generate_answer)
graph.add_node("extract_context", extract_context)
# graph.add_node(rewrite_question)
graph.add_node("tools", tools)
graph.add_node("source_router", source_router)

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
    # Assess LLM decision (call tools or respond to the user)
    tools_condition,
    {
        # Translate the condition outputs to nodes in our graph
        "tools": "tools",
        "__end__": END,
    },
)
# Extract context from tool results and store in state
graph.add_edge("tools", "extract_context")
# Route through source_router after extracting context
graph.add_edge("extract_context", "source_router")

graph.add_edge("source_router", "generate_answer")

graph.add_edge("generate_answer", END)
