import pytest
from unittest.mock import AsyncMock, MagicMock
from routing.router import OmniGraphRouter
from routing.taxonomy import QueryType
from omni_graph_rag.neo4j_client import Neo4jClient


@pytest.fixture
def mock_classifier():
    classifier = MagicMock()
    classifier.classify = AsyncMock()
    return classifier


@pytest.fixture
def mock_neo4j_client():
    client = MagicMock(spec=Neo4jClient)
    client.subject_search.return_value = [{"node": {"name": "Lean Startup"}, "score": 0.9}]
    client.object_discovery.return_value = [{"node": {"name": "Agile"}, "score": 0.8}]
    client.fact_check.return_value = [{"claim": {"claimText": "True"}, "score": 0.95}]
    return client


@pytest.fixture
def mock_router(mock_neo4j_client):
    router = MagicMock()
    router.neo4j_client = mock_neo4j_client
    router.route.return_value = {"path": "drift_nested", "results": {"final_answer": "Mocked DRIFT response"}}
    return router


@pytest.fixture
def omni_router(mock_classifier, mock_router):
    return OmniGraphRouter(classifier=mock_classifier, router=mock_router)


@pytest.mark.asyncio
async def test_subject_centered_route(omni_router, mock_classifier, mock_neo4j_client):
    mock_classifier.classify.return_value.query_type = QueryType.SUBJECT_CENTERED
    result = await omni_router.route("What is Lean Startup?")
    assert result["query_type"] == QueryType.SUBJECT_CENTERED
    assert result["source"] == "neo4j"
    mock_neo4j_client.subject_search.assert_called_once_with("What is Lean Startup?")


@pytest.mark.asyncio
async def test_object_discovery_route(omni_router, mock_classifier, mock_neo4j_client):
    mock_classifier.classify.return_value.query_type = QueryType.OBJECT_DISCOVERY
    result = await omni_router.route("Find frameworks related to innovation")
    assert result["query_type"] == QueryType.OBJECT_DISCOVERY
    assert result["source"] == "neo4j"
    mock_neo4j_client.object_discovery.assert_called_once_with("Find frameworks related to innovation")


@pytest.mark.asyncio
async def test_fact_checking_route(omni_router, mock_classifier, mock_neo4j_client):
    mock_classifier.classify.return_value.query_type = QueryType.FACT_CHECKING
    result = await omni_router.route("Is Lean Startup by Eric Ries?")
    assert result["query_type"] == QueryType.FACT_CHECKING
    assert result["source"] == "neo4j"
    mock_neo4j_client.fact_check.assert_called_once_with("Is Lean Startup by Eric Ries?")


@pytest.mark.asyncio
async def test_nested_route(omni_router, mock_classifier, mock_router):
    mock_classifier.classify.return_value.query_type = QueryType.NESTED
    result = await omni_router.route("How did Toyota influence modern management?")
    assert result["query_type"] == QueryType.NESTED
    assert result["source"] == "drift"
    mock_router.route.assert_called_once_with("How did Toyota influence modern management?")
