import types
import pytest

from langchain_core.messages import HumanMessage, AIMessage


@pytest.fixture
def fake_state_factory():
    """Factory to build a minimal `CustomMessagesState`-compatible dict.

    The router relies only on:
      - `has_been_rewritten` flag
      - `messages[-3].content` as question
      - `messages[-1].content` as context
    """

    def build_state(
        question: str = "What is the mission?",
        context: str = "Some context",
        has_been_rewritten: bool = False,
    ):
        messages = [
            AIMessage(content="previous ai"),  # [-4]
            HumanMessage(content=question),  # [-3] <- question consumed by router
            AIMessage(content="assistant reply"),
            AIMessage(content=context),  # [-1] <- context consumed by router
        ]
        return {"messages": messages, "has_been_rewritten": has_been_rewritten}

    return build_state


def test_document_grader_bypass_when_already_rewritten(monkeypatch, fake_state_factory):
    """If the question was already rewritten, the router must go straight to `generate_answer`.

    Also verify the LLM-backed grader is not invoked in this branch.
    """
    from src.graph.routing import document_grader as dg

    # Install a fake grader model that would fail if invoked
    class FailIfCalled:
        def with_structured_output(
            self, *_args, **_kwargs
        ):  # pragma: no cover - safety guard
            raise AssertionError(
                "grader_model should not be called when has_been_rewritten=True"
            )

    monkeypatch.setattr(dg, "grader_model", FailIfCalled(), raising=True)

    state = fake_state_factory(has_been_rewritten=True)
    result = dg.document_grader(state)

    assert result == "generate_answer"


def test_document_grader_routes_to_generate_answer_on_relevant(
    monkeypatch, fake_state_factory
):
    """When the grader returns a 'yes' score, the router must go to `generate_answer`."""
    from src.graph.routing import document_grader as dg

    # Fake GradeDocuments-like object
    relevant = types.SimpleNamespace(binary_score="yes")

    class FakeModel:
        def __init__(self):
            self.last_prompt = None

        def with_structured_output(self, *_args, **_kwargs):
            return self

        def invoke(self, messages):
            # capture prompt for sanity checks
            self.last_prompt = messages
            return relevant

    fake_model = FakeModel()
    monkeypatch.setattr(dg, "grader_model", fake_model, raising=True)

    state = fake_state_factory(
        question="Q?", context="Relevant ctx", has_been_rewritten=False
    )
    result = dg.document_grader(state)

    assert result == "generate_answer"
    # Basic sanity: prompt should include both question and context strings
    flat_prompt = " ".join(
        [
            m["content"] if isinstance(m, dict) else getattr(m, "content", "")
            for m in (fake_model.last_prompt or [])
        ]
    )
    assert "Q?" in flat_prompt
    assert "Relevant ctx" in flat_prompt


def test_document_grader_routes_to_rewrite_question_on_irrelevant(
    monkeypatch, fake_state_factory
):
    """When the grader returns a 'no' score, the router must go to `rewrite_question`."""
    from src.graph.routing import document_grader as dg

    irrelevant = types.SimpleNamespace(binary_score="no")

    class FakeModel:
        def with_structured_output(self, *_args, **_kwargs):
            return self

        def invoke(self, _messages):
            return irrelevant

    monkeypatch.setattr(dg, "grader_model", FakeModel(), raising=True)

    state = fake_state_factory(
        question="Q?", context="Off-topic", has_been_rewritten=False
    )
    result = dg.document_grader(state)

    assert result == "rewrite_question"
