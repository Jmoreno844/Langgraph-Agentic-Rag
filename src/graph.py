from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.prebuilt import tools_condition, ToolNode

from src.routing.document_grader import document_grader

from src.nodes.generate_answer_or_rag import generate_answer_or_rag
from src.nodes.generate_answer import generate_answer
from src.nodes.rewrite_question import rewrite_question
from src.tools.retriever import retriever_tool
from src.state import CustomMessagesState

retriever = ToolNode([retriever_tool])
graph = StateGraph(CustomMessagesState)

graph.add_node(generate_answer_or_rag)
graph.add_node(generate_answer)
graph.add_node(rewrite_question)
graph.add_node("retriever", retriever)

graph.add_edge(START, "generate_answer_or_rag")
graph.add_conditional_edges(
    "generate_answer_or_rag",
    # Assess LLM decision (call `retriever_tool` tool or respond to the user)
    tools_condition,
    {
        # Translate the condition outputs to nodes in our graph
        "tools": "retriever",
        END: END,
    },
)
graph.add_conditional_edges(
    "retriever",
    document_grader,
    {
        "rewrite_question": "rewrite_question",
        "generate_answer": "generate_answer",
    },
)
graph.add_edge("rewrite_question", "generate_answer_or_rag")
graph.add_edge("generate_answer", END)

app = graph.compile()

for chunk in app.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "What is the privacy policy of Aetherix Dynamics?",
            }
        ],
        "has_been_rewritten": False,
    }
):
    for node, update in chunk.items():
        print("Update from node", node)
        update["messages"][-1].pretty_print()
        print("\n\n")
