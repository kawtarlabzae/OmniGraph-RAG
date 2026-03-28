from __future__ import annotations
from typing import Any, Dict, List

from omni_graph_rag.neo4j_client import Neo4jClient


def local_concept_search(client: Neo4jClient, query: str, limit: int = 8) -> List[Dict[str, Any]]:
    """Subject-centered local search wrapper."""
    return client.subject_search(query, limit=limit)
