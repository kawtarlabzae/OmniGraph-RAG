from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List

from neo4j import GraphDatabase, Driver, Session


@dataclass
class Neo4jClient:
    """Neo4jBridge encapsulating hybrid graph + vector operations.

    Complexity:
    - Connection: O(1)
    - Query execution: depends on `Cypher` plan and index availability.
      - Subject-centered vector search: O(log n) if vector index exists.
      - Fact-check structural match: O(path_length * degree)
    """
    uri: str
    user: str
    password: str
    driver: Driver | None = None

    def connect(self) -> None:
        if self.driver is not None:
            return
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    def close(self) -> None:
        if self.driver is None:
            return
        self.driver.close()
        self.driver = None

    def _execute(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        self.connect()
        assert self.driver is not None
        with self.driver.session() as session:
            return session.run(query, parameters or {}).data()

    def subject_search(self, text: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Subject-Centered retrieval using standard Cypher (AuraDB Free compatible)."""
        cypher = """
        MATCH (n)
        WHERE toLower(n.name) CONTAINS toLower($query) 
           OR toLower(n.id) CONTAINS toLower($query)
        OPTIONAL MATCH (n)-[:RELATES_TO]->(m)
        RETURN n {.id, .name, .description} AS node, 1.0 AS score, count(m) AS degree
        ORDER BY degree DESC
        LIMIT $limit
        """
        # We pass the raw text query instead of computing a query_vector
        return self._execute(cypher, {"query": text, "limit": limit})

    def object_discovery(self, subject_concept: str, limit: int = 10) -> List[Dict[str, Any]]:
        cypher = """
        MATCH (n:InnovationConcept {name: $subject})-[:BUILD_ON|RELATES_TO]->(o)
        RETURN o {.id, .name, .type, .metadata} AS object
        LIMIT $limit
        """
        return self._execute(cypher, {"subject": subject_concept, "limit": limit})

    def fact_check(self, claim: str, limit: int = 10) -> List[Dict[str, Any]]:
        cypher = """
        CALL db.index.fulltext.queryNodes('claimsIdx', $claim) YIELD node, score
        OPTIONAL MATCH (node)-[:EVIDENCE_OF]->(e:Claim)
        RETURN node {.id, .claimText, .source} AS claim, score, collect(e {.id, .claimText}) AS evidence
        ORDER BY score DESC
        LIMIT $limit
        """
        return self._execute(cypher, {"claim": claim, "limit": limit})