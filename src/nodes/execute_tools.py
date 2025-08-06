from langgraph.graph import MessagesState
from langchain_core.messages import ToolMessage
from src.vectorstores.in_memory import create_in_memory_retriever_tool

# Create a registry of available tools
AVAILABLE_TOOLS = {"retrieve_rag_docs": create_in_memory_retriever_tool()}


def execute_tools(state: MessagesState):
    """Execute any tool calls from the last message."""
    last_message = state["messages"][-1]

    if not last_message.tool_calls:
        return {"messages": []}

    tool_messages = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]

        if tool_name in AVAILABLE_TOOLS:
            try:
                # Execute the tool dynamically
                tool = AVAILABLE_TOOLS[tool_name]
                result = tool.invoke(tool_call["args"])
                tool_message = ToolMessage(
                    content=str(result), tool_call_id=tool_call["id"]
                )
                tool_messages.append(tool_message)
            except Exception as e:
                # Handle tool execution errors
                error_message = ToolMessage(
                    content=f"Error executing tool {tool_name}: {str(e)}",
                    tool_call_id=tool_call["id"],
                )
                tool_messages.append(error_message)
        else:
            # Handle unknown tools
            error_message = ToolMessage(
                content=f"Unknown tool: {tool_name}", tool_call_id=tool_call["id"]
            )
            tool_messages.append(error_message)

    return {"messages": tool_messages}
