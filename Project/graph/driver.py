from __future__ import annotations
from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase, Driver, Session


class Neo4jGraphDriver:
    """Neu4j connection and query utility corresponding to OmniGraph-RAG backend."""

    def __init__(self, uri: str, user: str, password: str, encrypted: bool = False):
        self.uri = uri
        self.user = user
        self.password = password
        self.encrypted = encrypted
        self.driver: Optional[Driver] = None

    def connect(self) -> None:
        if self.driver is None:
            auth = (self.user, self.password)
            self.driver = GraphDatabase.driver(self.uri, auth=auth)

    def close(self) -> None:
        if self.driver is not None:
            self.driver.close()
            self.driver = None

    def run(self, cypher_query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Safe session executor returning list of dict results."""
        self.connect()
        assert self.driver is not None
        with self.driver.session() as session:
            result = session.run(cypher_query, params or {})
            return [record.data() for record in result]
