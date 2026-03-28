from __future__ import annotations
from typing import Any, Dict, List

from omni_graph_rag.neo4j_client import Neo4jClient


def expand_related_entities(client: Neo4jClient, subject_concept: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Object discovery style graph expansion."""
    return client.object_discovery(subject_concept, limit=limit)
