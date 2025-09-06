import os
import sys
import asyncio
from pathlib import Path
import ast

# --- Add project root to sys.path ---
# This allows the script to import modules from the 'src' directory
# irrespective of where the script is run from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
# -----------------------------------------

from dotenv import load_dotenv
from langsmith import Client
from langsmith.evaluation import aevaluate
from langsmith.schemas import Run, Example

# --- DEEPEVAL METRICS AS LANGSMITH EVALUATORS ---
# This is the core integration. We wrap each DeepEval metric
# in a function that LangSmith's `evaluate` can understand.
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
)
from deepeval.models import GPTModel

import warnings
from itertools import islice

# Suppress guardrails event loop warning
warnings.filterwarnings(
    "ignore", category=UserWarning, module="guardrails.validator_service"
)

# Load environment variables from .env file
load_dotenv()

# Ensure critical environment variables are set
for var in [
    "LANGSMITH_API_KEY",
    "LANGSMITH_ENDPOINT",
    "LANGSMITH_PROJECT",
    "OPENAI_API_KEY",
]:
    if var not in os.environ:
        raise EnvironmentError(f"Missing required environment variable: {var}")

# Constants
DEEPEVAL_MODEL_NAME = os.getenv("DEEPEVAL_METRIC_MODEL", "gpt-5-nano")
try:
    # Configure the model for DeepEval metrics, setting temperature to 1.0
    # This is necessary because the default temperature of 0 is not supported by certain models.
    DEEPEVAL_MODEL = GPTModel(model=DEEPEVAL_MODEL_NAME, temperature=1.0)
except Exception as e:
    print(f"Error configuring DeepEval model: {e}")
    # Fallback to the original model name as a string if instantiation fails
    DEEPEVAL_MODEL = DEEPEVAL_MODEL_NAME

MIN_SCORE_THRESHOLD = 0.5
LANGSMITH_DATASET_NAME = "synthetic_rag_slices"  # UPDATE THIS IF YOURS IS DIFFERENT


def _create_test_case(run: Run, example: Example) -> LLMTestCase | None:
    """Helper to construct a DeepEval LLMTestCase from LangSmith run/example."""
    # The 'run' object contains the outputs of your RAG system
    # The 'example' object contains the ground truth data from your dataset
    if not run.outputs:
        return None

    actual_output = str(run.outputs.get("output", "")).strip()

    # The input to your RAG system is on the example
    question = example.inputs["input"]

    # Ground truth data from our new v2 dataset schema
    expected_generation_output = example.outputs["expected_generation_output"]
    expected_retrieval_context = example.outputs["expected_retrieval_context"]
    if isinstance(expected_retrieval_context, str):
        parsed_list = None
        try:
            parsed = ast.literal_eval(expected_retrieval_context)
            if isinstance(parsed, list):
                parsed_list = [str(x) for x in parsed]
        except Exception:
            parsed_list = None
        expected_retrieval_context = (
            parsed_list if parsed_list is not None else [expected_retrieval_context]
        )

    # The actual context retrieved by your RAG system is on the run's outputs
    # NOTE: This requires the RAG graph to return a 'context' key.
    raw_context = run.outputs.get("context", [])

    def _to_text(doc_obj):
        if hasattr(doc_obj, "page_content"):
            try:
                return str(getattr(doc_obj, "page_content"))
            except Exception:
                pass
        if isinstance(doc_obj, dict) and "page_content" in doc_obj:
            return str(doc_obj.get("page_content"))
        return str(doc_obj)

    retrieval_context = [_to_text(doc) for doc in raw_context]

    return LLMTestCase(
        input=question,
        actual_output=str(actual_output),
        expected_output=str(
            expected_generation_output
        ),  # DeepEval uses this for AnswerRelevancy
        retrieval_context=retrieval_context,
        context=expected_retrieval_context,  # DeepEval uses this for ContextualRecall
    )


# Each metric gets its own evaluator function
def faithfulness_evaluator(run: Run, example: Example):
    test_case = _create_test_case(run, example)
    if not test_case:
        return {"score": 0, "reason": "Run had no outputs"}

    metric = FaithfulnessMetric(threshold=MIN_SCORE_THRESHOLD, model=DEEPEVAL_MODEL)
    metric.measure(test_case)
    return {"key": "faithfulness", "score": metric.score, "reason": metric.reason}


def answer_relevancy_evaluator(run: Run, example: Example):
    test_case = _create_test_case(run, example)
    if not test_case:
        return {"score": 0, "reason": "Run had no outputs"}

    metric = AnswerRelevancyMetric(threshold=MIN_SCORE_THRESHOLD, model=DEEPEVAL_MODEL)
    metric.measure(test_case)
    return {
        "key": "answer_relevancy",
        "score": metric.score,
        "reason": metric.reason,
    }


# --- NEW: ANSWER CORRECTNESS EVALUATOR ---
# This directly compares the generated answer to the ground truth answer.
from deepeval.metrics import SummarizationMetric


def answer_correctness_evaluator(run: Run, example: Example):
    """
    Compares the generated answer against the ground truth answer for semantic similarity.
    """
    if not run.outputs or not example.outputs:
        return {"score": 0, "reason": "Missing output or example data"}

    actual_output = str(run.outputs.get("output", "")).strip()  # Now a clean string
    expected_generation_output = example.outputs.get("expected_generation_output")
    # --- DEBUG: Inspect evaluator inputs ---
    try:
        print(
            f"DEBUG: answer_correctness inputs: expected_len={len(str(expected_generation_output or ''))} actual_len={len(actual_output)}"
        )
    except Exception:
        pass

    # We can use SummarizationMetric to check for semantic overlap and factual consistency.
    metric = SummarizationMetric(threshold=MIN_SCORE_THRESHOLD, model=DEEPEVAL_MODEL)
    test_case = LLMTestCase(
        input=example.inputs["input"],
        actual_output=str(actual_output),
        expected_output=str(expected_generation_output),
    )
    metric.measure(test_case)
    # --- DEBUG: Inspect evaluator result ---
    try:
        _reason_prev = (metric.reason or "")[:200].replace("\n", " ")
        print(
            f"DEBUG: answer_correctness score={metric.score} reason_preview={_reason_prev}"
        )
    except Exception:
        pass
    return {
        "key": "answer_correctness",
        "score": metric.score,
        "reason": metric.reason,
    }


def contextual_relevancy_evaluator(run: Run, example: Example):
    test_case = _create_test_case(run, example)
    if not test_case:
        return {"score": 0, "reason": "Run had no outputs"}

    # --- DEBUG: Inspect evaluator inputs ---
    try:
        _ctx_count = len(test_case.context or [])
        _ret_count = len(test_case.retrieval_context or [])
        _ctx_prev = (
            str(test_case.context[0])[:200].replace("\n", " ") if _ctx_count else ""
        )
        _ret_prev = (
            str(test_case.retrieval_context[0])[:200].replace("\n", " ")
            if _ret_count
            else ""
        )
        print(
            f"DEBUG: contextual_relevancy inputs: expected_items={_ctx_count} retrieved_items={_ret_count} expected_preview={_ctx_prev} retrieved_preview={_ret_prev}"
        )
    except Exception:
        pass

    metric = ContextualRelevancyMetric(
        threshold=MIN_SCORE_THRESHOLD, model=DEEPEVAL_MODEL
    )
    metric.measure(test_case)
    # --- DEBUG: Inspect evaluator result ---
    try:
        _reason_prev = (metric.reason or "")[:200].replace("\n", " ")
        print(
            f"DEBUG: contextual_relevancy score={metric.score} reason_preview={_reason_prev}"
        )
    except Exception:
        pass
    return {
        "key": "contextual_relevancy",
        "score": metric.score,
        "reason": metric.reason,
    }


def retrieval_recall_evaluator(run: Run, example: Example):
    test_case = _create_test_case(run, example)
    if not test_case:
        return {"score": 0, "reason": "Run had no outputs"}

    # --- DEBUG: Inspect evaluator inputs ---
    try:
        _ctx_count = len(test_case.context or [])
        _ret_count = len(test_case.retrieval_context or [])
        _ctx_prev = (
            str(test_case.context[0])[:200].replace("\n", " ") if _ctx_count else ""
        )
        _ret_prev = (
            str(test_case.retrieval_context[0])[:200].replace("\n", " ")
            if _ret_count
            else ""
        )
        print(
            f"DEBUG: retrieval_recall inputs: expected_items={_ctx_count} retrieved_items={_ret_count} expected_preview={_ctx_prev} retrieved_preview={_ret_prev}"
        )
    except Exception:
        pass

    metric = ContextualRecallMetric(threshold=MIN_SCORE_THRESHOLD, model=DEEPEVAL_MODEL)
    metric.measure(test_case)
    # --- DEBUG: Inspect evaluator result ---
    try:
        _reason_prev = (metric.reason or "")[:200].replace("\n", " ")
        print(
            f"DEBUG: retrieval_recall score={metric.score} reason_preview={_reason_prev}"
        )
    except Exception:
        pass
    return {
        "key": "retrieval_recall",
        "score": metric.score,
        "reason": metric.reason,
    }


# --- RAG SYSTEM AS THE "TARGET" FOR EVALUATION ---
# We need a function that takes a LangSmith example and runs our RAG system.


async def get_rag_app():
    """Dynamically import and compile the graph.
    This helps avoid circular dependencies and config issues.
    """
    from src.graph.graph import graph
    from langgraph.checkpoint.memory import MemorySaver

    return graph.compile(checkpointer=MemorySaver())


async def rag_system_target(example: dict):
    """
    This is the "target" function that LangSmith will run for each example.
    It runs the RAG application and formats the output for evaluation.
    """
    app = await get_rag_app()
    question = example["input"]
    session_id = f"eval-{abs(hash(question)) % 100000}"
    config = {"configurable": {"thread_id": session_id}}

    # The output of the RAG app is the entire graph state
    output_state = await app.ainvoke({"messages": [("user", question)]}, config)
    # --- DEBUG: Inspect raw output_state and messages ---
    print("DEBUG: output_state type:", type(output_state))
    try:
        print("DEBUG: output_state keys:", list(output_state.keys()))
    except Exception as e:
        print("DEBUG: cannot list output_state keys:", e)
    _dbg_messages = output_state.get("messages", [])
    print(f"DEBUG: messages count: {len(_dbg_messages)}")
    if isinstance(_dbg_messages, list) and _dbg_messages:
        for idx, msg in enumerate(_dbg_messages[-3:]):
            _msg_type = getattr(msg, "type", None) or msg.__class__.__name__
            _content = getattr(msg, "content", None)
            _preview = (
                str(_content)[:200].replace("\n", " ") if _content is not None else ""
            )
            print(
                f"DEBUG: message[{len(_dbg_messages)-min(3,len(_dbg_messages))+idx}] type={_msg_type} preview={_preview}"
            )

    # Extract the final AI-generated message
    final_answer = ""
    if messages := output_state.get("messages"):
        if isinstance(messages, list) and messages:
            # The agent's response is typically the last message in the list
            try:
                final_answer = messages[-1].content
            except Exception:
                final_answer = str(messages[-1])
    final_answer = str(final_answer).strip()
    # --- DEBUG: Inspect final answer ---
    print(
        f"DEBUG: final_answer length: {len(final_answer)} preview: {final_answer[:200].replace('\n',' ')}"
    )

    print(f"DEBUG: Final state keys: {list(output_state.keys())}")
    if "retrieved_context" in output_state:
        print(
            f"DEBUG: retrieved_context type: {type(output_state['retrieved_context'])}, value: {output_state['retrieved_context']}"
        )
    else:
        print(
            "DEBUG: retrieved_context not found in final state - checking tool messages"
        )
        # Check for tool messages in the final state
        messages = output_state.get("messages", [])
        tool_contexts = [
            msg.content
            for msg in messages
            if hasattr(msg, "type") and msg.type == "tool"
        ]
        output_state["retrieved_context"] = tool_contexts
        print(
            "DEBUG: retrieved_context not found in final state - checking tool messages"
        )
        # Check for tool messages in the final state
        messages = output_state.get("messages", [])
        tool_contexts = [
            msg.content
            for msg in messages
            if hasattr(msg, "type") and msg.type == "tool"
        ]
        output_state["retrieved_context"] = tool_contexts
        print(
            f"DEBUG: retrieved_context type: {type(output_state['retrieved_context'])}, value: {output_state['retrieved_context']}"
        )
    # Extract the retrieved context for evaluation.
    # NOTE: This assumes your graph's final state includes a 'retrieved_context' key
    # containing the retrieved documents. Your RAG agent must return this.
    raw_context = output_state.get("retrieved_context", [])
    if isinstance(raw_context, list):
        raw_context = [
            {"page_content": str(item)} if not hasattr(item, "page_content") else item
            for item in raw_context
            if item is not None
        ]
    elif isinstance(raw_context, str) and raw_context.strip():
        raw_context = [{"page_content": raw_context}]
    else:
        raw_context = []

    def _to_text(doc_obj):
        if hasattr(doc_obj, "page_content"):
            try:
                return str(getattr(doc_obj, "page_content"))
            except Exception:
                pass
        if isinstance(doc_obj, dict) and "page_content" in doc_obj:
            return str(doc_obj.get("page_content"))
        return str(doc_obj)

    retrieved_context = [_to_text(doc) for doc in raw_context]
    # --- DEBUG: Inspect normalized retrieved context ---
    print(f"DEBUG: retrieved_context normalized count: {len(retrieved_context)}")
    if retrieved_context:
        _rc_prev = str(retrieved_context[0])[:200].replace("\n", " ")
        print(f"DEBUG: retrieved_context[0] preview: {_rc_prev}")

    return {
        "output": final_answer,
        "llm_output": final_answer,
        "answer": final_answer,
        "output_preview": final_answer[:200],
        "context": retrieved_context,
        "retrieved_context_count": len(retrieved_context),
        "retrieved_context_preview": retrieved_context[0] if retrieved_context else "",
    }


async def main():
    """
    The main function to run the evaluation experiment.
    """
    client = Client()
    # Check if the dataset exists
    if not client.has_dataset(dataset_name=LANGSMITH_DATASET_NAME):
        print(
            f"❌ Dataset '{LANGSMITH_DATASET_NAME}' not found in LangSmith project '{os.getenv('LANGSMITH_PROJECT')}'."
        )
        print("Please upload it first or correct the LANGSMITH_DATASET_NAME constant.")
        return

    # --- New: Add ability to limit test rows via environment variable ---
    try:
        limit_str = os.getenv("EVAL_LIMIT")
        limit = int(limit_str) if limit_str else None
    except (ValueError, TypeError):
        limit = None
        print("⚠️ Invalid value for EVAL_LIMIT. Running on full dataset.")

    dataset_or_examples = LANGSMITH_DATASET_NAME
    if limit and limit > 0:
        print(f"🔬 Limiting evaluation to {limit} example(s).")
        examples_iterator = client.list_examples(
            dataset_name=LANGSMITH_DATASET_NAME, limit=limit, offset=0
        )
        limited_examples = list(islice(examples_iterator, limit))
        if not limited_examples:
            print(
                f"⚠️ No examples found in dataset '{LANGSMITH_DATASET_NAME}'. Aborting."
            )
            return
        dataset_or_examples = limited_examples
    # --- End new section ---

    # Determine what we're actually evaluating on for accurate logging
    if isinstance(dataset_or_examples, list):
        run_target_desc = f"{len(dataset_or_examples)} in-memory example(s) from dataset: '{LANGSMITH_DATASET_NAME}'"
    else:
        run_target_desc = f"dataset: '{dataset_or_examples}'"

    print(f"🚀 Starting evaluation experiment on {run_target_desc}")

    # This is the magic function that runs the whole experiment
    await aevaluate(
        rag_system_target,
        data=dataset_or_examples,  # Use limited examples or the full dataset name
        evaluators=[
            faithfulness_evaluator,
            answer_relevancy_evaluator,
            retrieval_recall_evaluator,
            contextual_relevancy_evaluator,
            answer_correctness_evaluator,  # Add the new metric
        ],
        experiment_prefix="RAG E2E Eval V2 - DeepEval",
        max_concurrency=4,  # Adjust based on your machine and API rate limits
    )
    print("\n✅ Evaluation experiment complete!")
    print("👉 View results in LangSmith:")
    # Construct the URL to the project
    project_name_slug = os.getenv("LANGSMITH_PROJECT").replace(" ", "-")
    url = f"{os.getenv('LANGSMITH_ENDPOINT')}/projects/p/{project_name_slug}"
    print(url)


if __name__ == "__main__":
    # This allows the script to be run from the command line
    # e.g., `python scripts/evals/run_langsmith_experiment.py`
    # Mock services for local run if needed
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "test")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "test")
    os.environ.setdefault("AWS_S3_RAG_DOCUMENTS_BUCKET", "")

    asyncio.run(main())
