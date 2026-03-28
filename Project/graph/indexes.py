from __future__ import annotations
from typing import Any

from graph.driver import Neo4jGraphDriver


def ensure_graph_indexes(driver: Neo4jGraphDriver) -> None:
    """Create schema and full-text indexes for the OmniGraph-RAG knowledge graph."""
    queries = [
        "CREATE INDEX IF NOT EXISTS FOR (n:InnovationConcept) ON (n.name)",
        "CREATE INDEX IF NOT EXISTS FOR (n:Claim) ON (n.claimText)",
        "CALL db.index.fulltext.createNodeIndex('contentIdx', ['InnovationConcept','ManagementEntity','Claim'], ['name','description','claimText'])",
        "CALL db.index.fulltext.createNodeIndex('claimsIdx', ['Claim'], ['claimText'])",
    ]

    for q in queries:
        try:
            driver.run(q)
        except Exception:
            # Silence errors for being already present; real code should log.
            continue
