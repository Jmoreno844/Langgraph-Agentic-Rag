import pytest
from langchain_core.messages import AIMessage, HumanMessage


class FakeLLMRecordPrompt:
    def __init__(self, response_text: str = "ok"):
        self.last_messages = None
        self.response_text = response_text

    def invoke(self, messages):
        self.last_messages = messages
        return AIMessage(content=self.response_text)


def test_generate_answer_formats_prompt_with_question_and_context(monkeypatch):
    import src.graph.nodes.generate_answer as node

    fake_llm = FakeLLMRecordPrompt(response_text="answer")
    monkeypatch.setattr(node, "response_model", fake_llm)

    # State layout used by node: user question at -3, context at -1
    state = {
        "messages": [
            HumanMessage(content="ignored"),
            AIMessage(content="ignored"),
            HumanMessage(content="What is the warranty?"),  # -3
            AIMessage(content="tool output header"),
            AIMessage(content="Warranty is 12 months."),  # -1 context
        ]
    }

    out = node.generate_answer(state)

    # Verify prompt composition contains question and context
    sent = fake_llm.last_messages[0]
    assert sent["role"] == "user"
    assert "What is the warranty?" in sent["content"]
    assert "Warranty is 12 months." in sent["content"]

    # Response shape
    assert isinstance(out["messages"][0], AIMessage)


def test_generate_answer_handles_short_state(monkeypatch):
    import src.graph.nodes.generate_answer as node

    fake_llm = FakeLLMRecordPrompt(response_text="answer")
    monkeypatch.setattr(node, "response_model", fake_llm)

    # Deliberately too short; expect IndexError or a clear failure
    state = {"messages": [HumanMessage(content="hi")]}

    with pytest.raises(Exception):
        node.generate_answer(state)
