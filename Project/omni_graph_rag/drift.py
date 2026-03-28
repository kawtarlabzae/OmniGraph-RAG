from __future__ import annotations
from typing import Any, Dict, List

from langchain_openai import OpenAI
from .neo4j_client import Neo4jClient


class DriftEngine:
    """DRIFT search pipeline: dynamic reasoning + inference loops.

    Algorithmic pattern:
    - Step 1: seed with global context (map reduce style summaries from Neo4j)
    - Step 2: iterative follow-up with query re-scoping
    - Step 3: final aggregation and synthesis via LLM

    Complexity:
    - O(iterations * retrieval_cost + llm_cost)
    - `retrieval_cost` depends on Neo4j index use and graph degree.
    """

    def __init__(self, neo4j_client: Neo4jClient, llm: OpenAI, max_iterations: int = 3):
        self.neo4j_client = neo4j_client
        self.llm = llm
        self.max_iterations = max_iterations

    def _seed_summary(self, query: str) -> List[Dict[str, Any]]:
        # Basic map phase: globally aggregate top relevant nodes using claim and text match
        q = """
        CALL db.index.fulltext.queryNodes('contentIdx', $q) YIELD node, score
        RETURN node {.id, .name, .description, .type} AS doc, score
        ORDER BY score DESC
        LIMIT 15
        """
        return self.neo4j_client._execute(q, {"q": query})

    def _refine_query(self, base: str, evidence: List[Dict[str, Any]]) -> str:
        followups = ", ".join([doc["doc"]["name"] for doc in evidence[:5] if "doc" in doc])
        prompt = (
            f"You are a reasoning agent. Given base query: '{base}', and evidence labels: {followups}, "
            "produce one concise follow-up query for deeper graph focus."
        )
        resp = self.llm(prompt)
        return resp.strip()

    def iterate(self, query: str) -> Dict[str, Any]:
        chain = []
        evidence = self._seed_summary(query)
        chain.append({"phase": "seed", "data": evidence})

        for i in range(self.max_iterations):
            followup_query = self._refine_query(query, evidence)
            if not followup_query or followup_query.lower() == query.lower():
                break
            corridor = self.nearby_nodes(followup_query)
            chain.append({"phase": f"iteration_{i+1}", "query": followup_query, "data": corridor})
            evidence = corridor

        final_prompt = (
            f"Given the original user question: '{query}', "
            "and the staged iterative findings, build a fact-backed synthesis with a recommended next step." 
        )
        for step in chain:
            final_prompt += f"\n\n{step['phase']}: {step.get('query', '')}\n{step.get('data', '')}"

        synthesis = self.llm(final_prompt)

        return {
            "query": query,
            "chain": chain,
            "synthesis": synthesis,
        }

    def nearby_nodes(self, query: str, limit: int = 8) -> List[Dict[str, Any]]:
        cypher = """
        CALL db.index.fulltext.queryNodes('contentIdx', $query) YIELD node, score
        MATCH (node)-[:RELATES_TO|BUILD_ON]->(neighbor)
        RETURN neighbor {.id, .name, .description} AS node, score
        ORDER BY score DESC
        LIMIT $limit
        """
        return self.neo4j_client._execute(cypher, {"query": query, "limit": limit})
