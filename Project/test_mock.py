import asyncio
import os
os.environ["MOCK_MODE"] = "true"

from unittest.mock import MagicMock
from routing.router import OmniGraphRouter
from omni_graph_rag.router import QueryRouter
from drift.engine import DriftEngine
from synthesis.generator import ResponseSynthesizer

# Mock Neo4j client
mock_client = MagicMock()
mock_client.subject_search.return_value = [{"node": {"name": "Mocked Innovation Concept"}, "score": 0.9}]
mock_client.object_discovery.return_value = [{"node": {"name": "Mocked Related Entity"}, "score": 0.8}]
mock_client.fact_check.return_value = [{"claim": {"claimText": "Mocked Claim Verified"}, "score": 0.95}]

# Mock LLM
mock_llm = MagicMock()
mock_llm.return_value = "Mocked LLM response for synthesis."

# Create components
mock_router = QueryRouter(neo4j_client=mock_client, drift_engine=DriftEngine(mock_client, mock_llm), llm=mock_llm)
router = OmniGraphRouter(router=mock_router)
synthesizer = ResponseSynthesizer(mock_llm)

async def test():
    # Test routing
    route_data = await router.route("What is innovation?")
    print("Route result:", route_data)

    # Test synthesis
    final = synthesizer.synthesize("What is innovation?", route_data)
    print("Synthesis result:", final)

    # Simulate API response
    response = {
        "query": "What is innovation?",
        "query_type": route_data.get("query_type", "UNKNOWN"),
        "source": route_data.get("source", "UNKNOWN"),
        "final_text": final,
    }
    print("API Response:", response)

asyncio.run(test())