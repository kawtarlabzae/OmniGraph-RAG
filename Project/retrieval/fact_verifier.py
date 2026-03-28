from __future__ import annotations
from typing import Any, Dict, List

from omni_graph_rag.neo4j_client import Neo4jClient


def verify_claim(client: Neo4jClient, claim: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Fact check wrapper over Neo4j claim retrieval."""
    return client.fact_check(claim, limit=limit)
