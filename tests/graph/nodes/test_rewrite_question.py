import pytest
from langchain_core.messages import HumanMessage, AIMessage


@pytest.fixture
def base_messages():
    # Minimal conversation where the user's original question is at messages[-2]
    return [
        AIMessage(content="system or assistant"),
        HumanMessage(content="How can I reset my password?"),  # [-2]
        AIMessage(content="previous response"),
    ]


def test_rewrite_question_prompt_and_state(monkeypatch, base_messages):
    from src.graph.nodes import rewrite_question as rq

    captured_prompt = {"value": None}

    class FakeModel:
        def invoke(self, messages):
            # Expect a single HumanMessage prompt with formatted content
            assert len(messages) == 1
            msg = messages[0]
            assert isinstance(msg, HumanMessage)
            captured_prompt["value"] = msg.content
            # Return a rewritten question as AIMessage-like object
            return AIMessage(
                content="What steps are required to perform a password reset?"
            )

    monkeypatch.setattr(rq, "response_model", FakeModel(), raising=True)

    state = {"messages": base_messages.copy()}
    new_state = rq.rewrite_question(state)

    # The prompt should contain the original question
    assert "How can I reset my password?" in captured_prompt["value"]

    # Verify outputs
    assert isinstance(new_state, dict)
    assert new_state.get("has_been_rewritten") is True

    out_messages = new_state.get("messages")
    assert isinstance(out_messages, list)
    assert len(out_messages) == 1
    assert isinstance(out_messages[0], AIMessage)
    assert "password reset" in out_messages[0].content.lower()
