from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.prebuilt import tools_condition

from src.nodes import document_grader
from src.nodes import execute_tools
from src.nodes import generate_answer_or_rag
from src.nodes import generate_answer
from src.nodes import rewrite_question
from src.vectorstores.in_memory import create_in_memory_retriever_tool

retriever = create_in_memory_retriever_tool()

graph = StateGraph(MessagesState)

graph.add_node(document_grader)
graph.add_node(generate_answer_or_rag)
graph.add_node(generate_answer)
graph.add_node(rewrite_question)
graph.add_node("retriever", retriever)

graph.add_edge(START, "generate_answer_or_rag")
graph.add_conditionaledge(
    "generate_answer_or_rag",
    # Assess LLM decision (call `retriever_tool` tool or respond to the user)
    tools_condition,
    {
        # Translate the condition outputs to nodes in our graph
        "tools": "retriever",
        END: END,
    },
)
graph.add_edge("retriever", "grade_documents")
graph.add_conditional_edges(
    "grade_documents",
    tools_condition,
    {
        "rewrite_question": "rewrite_question",
        "generate_answer": "generate_answer",
    },
)
graph.add_edge("rewrite_question", "generate_answer")
graph.add_edge("generate_answer", END)

app = graph.compile()

app.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What is the privacy policy of Aetherix Dynamics?",
            }
        ]
    }
)
