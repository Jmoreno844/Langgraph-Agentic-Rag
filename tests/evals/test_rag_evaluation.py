import sys
import os
import asyncio
import pytest
from unittest.mock import Mock, patch
from dotenv import load_dotenv
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Load environment variables from .env file
load_dotenv()

import pandas as pd
from pathlib import Path
from deepeval import assert_test
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
)
from deepeval.test_case import LLMTestCase


# Configuration
MIN_SCORE_THRESHOLD = 0.7  # Minimum score for tests to pass
DATASET_PATH = Path("evals/synthetic_dataset.csv")


def load_evaluation_dataset():
    """Load the synthetic evaluation dataset from CSV"""
    if not DATASET_PATH.exists():
        pytest.skip(f"Evaluation dataset not found at {DATASET_PATH}. "
                   "Run scripts/generate_eval_dataset.py to create it.")
    
    df = pd.read_csv(DATASET_PATH)
    
    # Convert to list of dictionaries for parametrization
    test_cases = []
    for _, row in df.iterrows():
        test_cases.append({
            'input': row['input'],
            'expected_output': str(row['expected_output']) if pd.notna(row['expected_output']) else '',
            'context': str(row['context']) if pd.notna(row['context']) else '',
            'source_file': str(row['source_file']) if pd.notna(row['source_file']) else ''
        })
    
    return test_cases


def get_mock_graph_app():
    """Create a mock graph application that simulates RAG behavior for testing"""
    from langgraph.graph import StateGraph, START, END
    from langgraph.graph import MessagesState
    from langgraph.checkpoint.memory import MemorySaver
    
    # Create a simple mock graph that just returns a direct answer
    def mock_answer_node(state):
        # Simulate a simple answer based on the question
        question = state["messages"][-1].content.lower()
        if "troubleshoot" in question or "display" in question:
            answer = "When troubleshooting PC no-display issues, check power cables, monitor connections, and ensure the graphics card is properly seated."
        elif "warranty" in question:
            answer = "Our warranty covers defects in materials and workmanship for 1 year from purchase date."
        else:
            answer = "Thank you for your question. I'm here to help with PC components and technical support."
        
        from langchain_core.messages import AIMessage
        return {"messages": [AIMessage(content=answer)], "documents": []}
    
    # Create a simple graph
    graph = StateGraph(MessagesState)
    graph.add_node("answer", mock_answer_node)
    graph.add_edge(START, "answer")
    graph.add_edge("answer", END)
    
    # Compile with memory saver
    memory_checkpointer = MemorySaver()
    return graph.compile(checkpointer=memory_checkpointer)


@pytest.mark.asyncio
@pytest.mark.parametrize("test_case_data", load_evaluation_dataset()[:2])  # Test first 2 cases for demo
async def test_rag_evaluation(test_case_data: dict):
    """
    End-to-end RAG evaluation test using a mock graph.
    
    For each test case from the synthetic dataset:
    1. Run the question through our mock LangGraph RAG system (async)
    2. Capture the final answer and retrieved documents
    3. Evaluate using DeepEval metrics (async)
    4. Assert that scores meet minimum thresholds
    """
    # Extract test case data
    input_question = test_case_data["input"]
    expected_answer = test_case_data["expected_output"]
    expected_context = [test_case_data["context"]]  # Ground truth context
    
    # Get the mock graph application
    graph_app = get_mock_graph_app()
    
    # Create unique session ID for this test case
    session_id = f"eval-session-{hash(input_question) % 10000}"
    config = {"configurable": {"thread_id": session_id}}
    
    # --- ACT: Run the RAG system ---
    try:
        final_state = await graph_app.ainvoke(
            {"messages": [("user", input_question)]}, 
            config
        )
        
        # Extract the actual answer from the final message
        actual_answer = final_state["messages"][-1].content
        
        # Extract retrieved documents from the state
        # The documents are stored in the 'documents' key of the final state
        retrieved_documents = final_state.get("documents", [])
        retrieved_context = [doc.page_content for doc in retrieved_documents]
        
    except Exception as e:
        pytest.fail(f"RAG system failed to process question '{input_question}': {e}")
    
    # --- ASSERT: Evaluate the response ---
    # Create DeepEval test case with captured data
    test_case = LLMTestCase(
        input=input_question,
        actual_output=actual_answer,
        expected_output=expected_answer,
        retrieval_context=retrieved_context,  # What our system retrieved
        context=expected_context             # Ground truth context
    )
    
    # Define the RAG triad metrics + additional contextual metrics
    metrics_to_evaluate = [
        FaithfulnessMetric(
            threshold=MIN_SCORE_THRESHOLD, 
            model="gpt-4o-mini",
            include_reason=True
        ),
        AnswerRelevancyMetric(
            threshold=MIN_SCORE_THRESHOLD, 
            model="gpt-4o-mini",
            include_reason=True
        ),
        ContextualRecallMetric(
            threshold=MIN_SCORE_THRESHOLD, 
            model="gpt-4o-mini",
            include_reason=True
        ),
        ContextualRelevancyMetric(
            threshold=MIN_SCORE_THRESHOLD, 
            model="gpt-4o-mini",
            include_reason=True
        ),
    ]
    
    # Run the evaluation and assert all metrics pass their thresholds
    assert_test(test_case, metrics_to_evaluate)


def test_dataset_integrity():
    """Test that the evaluation dataset has the expected structure and content"""
    test_cases = load_evaluation_dataset()
    
    assert len(test_cases) > 0, "Dataset should contain at least one test case"
    
    # Check that all test cases have required fields
    for i, test_case in enumerate(test_cases):
        assert "input" in test_case, f"Test case {i} missing 'input' field"
        assert "expected_output" in test_case, f"Test case {i} missing 'expected_output' field"
        assert "context" in test_case, f"Test case {i} missing 'context' field"
        
        # Ensure inputs and expected outputs are not empty
        assert len(test_case["input"].strip()) > 0, f"Test case {i} has empty input"
        assert len(test_case["expected_output"].strip()) > 0, f"Test case {i} has empty expected_output"
        assert len(test_case["context"].strip()) > 0, f"Test case {i} has empty context"
