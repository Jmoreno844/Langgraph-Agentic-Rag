#!/usr/bin/env python3
"""
Quick test script to verify RAG evaluation works with a single test case.
This helps debug the integration before running the full test suite.
"""

import sys
import os
import asyncio
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from pathlib import Path
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from src.graph.graph import graph

# Use MemorySaver for testing instead of PostgresSaver
from langgraph.checkpoint.memory import MemorySaver

async def main():
    print("🚀 Quick RAG Evaluation Test")
    
    # Load one test case from the dataset
    dataset_path = Path("evals/synthetic_dataset.csv")
    if not dataset_path.exists():
        print("❌ Dataset not found. Run scripts/generate_eval_dataset.py first.")
        return
    
    df = pd.read_csv(dataset_path)
    
    # Take the first test case
    row = df.iloc[0]
    print(f"📝 Testing question: {row['input'][:100]}...")
    
    # Get the LangGraph application with MemorySaver for testing
    print("🔄 Initializing RAG system with MemorySaver...")
    memory_checkpointer = MemorySaver()
    graph_app = graph.compile(checkpointer=memory_checkpointer)
    
    # Run the question through the system
    session_id = "quick-test-session"
    config = {"configurable": {"thread_id": session_id}}
    
    try:
        print("🤖 Running question through RAG system...")
        final_state = await graph_app.ainvoke(
            {"messages": [("user", row["input"])]}, 
            config
        )
        
        # Extract results
        actual_answer = final_state["messages"][-1].content
        retrieved_documents = final_state.get("documents", [])
        retrieved_context = [doc.page_content for doc in retrieved_documents]
        
        print(f"✅ Got answer: {actual_answer[:200]}...")
        print(f"📚 Retrieved {len(retrieved_context)} documents")
        
        # Create test case for evaluation
        test_case = LLMTestCase(
            input=str(row["input"]),
            actual_output=actual_answer,
            expected_output=str(row["expected_output"]) if pd.notna(row["expected_output"]) else "",
            retrieval_context=retrieved_context,
            context=[str(row["context"])] if pd.notna(row["context"]) else []
        )
        
        # Run basic evaluation
        print("🔍 Evaluating with DeepEval...")
        
        faithfulness = FaithfulnessMetric(model="gpt-4o-mini", include_reason=True)
        relevancy = AnswerRelevancyMetric(model="gpt-4o-mini", include_reason=True)
        
        await faithfulness.a_measure(test_case)
        await relevancy.a_measure(test_case)
        
        print(f"📊 Faithfulness Score: {faithfulness.score:.2f}")
        print(f"📊 Answer Relevancy Score: {relevancy.score:.2f}")
        
        if faithfulness.score >= 0.7 and relevancy.score >= 0.7:
            print("✅ Test PASSED!")
        else:
            print("⚠️  Test FAILED - Scores below threshold")
            
    except Exception as e:
        print(f"❌ Error running RAG system: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
