from __future__ import annotations
import asyncio
from typing import Any, Dict, Optional

from routing.classifier import PolyGClassifier
from routing.taxonomy import QueryType
from omni_graph_rag.router import make_router_from_env, QueryRouter
from retrieval.local_search import local_concept_search
from retrieval.graph_expansion import expand_related_entities
from retrieval.fact_verifier import verify_claim


class OmniGraphRouter:
    """Unified routing layer combining PolyG classification and the OmniGraph engine."""

    def __init__(self, classifier: Optional[PolyGClassifier] = None, router: Optional[QueryRouter] = None):
        self.classifier = classifier or PolyGClassifier()
        self.router = router or make_router_from_env()

    async def _classify(self, query: str) -> QueryType:
        result = await self.classifier.classify(query)
        return result.query_type

    async def route(self, query: str) -> Dict[str, Any]:
        qtype = await self._classify(query)

        dispatch = {
            QueryType.SUBJECT_CENTERED: lambda: local_concept_search(self.router.neo4j_client, query),
            QueryType.OBJECT_DISCOVERY: lambda: expand_related_entities(self.router.neo4j_client, query),
            QueryType.FACT_CHECKING: lambda: verify_claim(self.router.neo4j_client, query),
            QueryType.NESTED: lambda: self.router.route(query),
        }

        data = dispatch[qtype]()
        if asyncio.iscoroutine(data):
            data = await data

        return {
            "query": query,
            "query_type": qtype,
            "results": data,
            "source": "drift" if qtype == QueryType.NESTED else "neo4j",
        }
