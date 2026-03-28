import pytest
from unittest.mock import MagicMock
from synthesis.generator import ResponseSynthesizer
from synthesis.prompt_builder import build_synthesis_prompt
from langchain.llms import OpenAI


@pytest.fixture
def mock_llm():
    llm = MagicMock(spec=OpenAI)
    llm.return_value = "Final synthesized answer."
    return llm


@pytest.fixture
def synthesizer(mock_llm):
    return ResponseSynthesizer(llm=mock_llm)


def test_build_synthesis_prompt():
    query = "What is innovation?"
    route_result = {
        "query_type": "SUBJECT_CENTERED",
        "source": "neo4j",
        "results": [{"node": {"name": "Innovation Concept"}, "score": 0.9}]
    }
    prompt = build_synthesis_prompt(query, route_result)
    assert "What is innovation?" in prompt
    assert "SUBJECT_CENTERED" in prompt
    assert "Innovation Concept" in prompt


def test_synthesizer_synthesize(synthesizer, mock_llm):
    query = "Explain Lean Startup"
    route_payload = {
        "query_type": "SUBJECT_CENTERED",
        "source": "neo4j",
        "results": [{"node": {"name": "Lean Startup"}, "score": 0.95}]
    }
    result = synthesizer.synthesize(query, route_payload)
    assert result == "Final synthesized answer."
    mock_llm.assert_called_once()
    call_args = mock_llm.call_args[0][0]
    assert "Explain Lean Startup" in call_args
    assert "SUBJECT_CENTERED" in call_args
