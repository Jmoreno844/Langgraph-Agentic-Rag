# src/tools/retriever.py
from src.graph.vectorstores.in_memory import create_in_memory_retriever_tool

# Create the tool once
retriever_tool = create_in_memory_retriever_tool()
