from __future__ import annotations
from typing import Any, Dict, List

from langchain_core.language_models import BaseLanguageModel

from drift.context_merger import merge_contexts
from drift.subquery_planner import SubqueryPlanner
from routing.taxonomy import DRIFT_MAX_ITERATIONS
from core.state import DriftIteration
from omni_graph_rag.neo4j_client import Neo4jClient


class DriftEngine:
    """DRIFT engine from the OmniGraph-RAG pipeline."""

    def __init__(self, neo4j_client: Neo4jClient, llm: BaseLanguageModel, max_iterations: int = DRIFT_MAX_ITERATIONS):
        self.neo4j = neo4j_client
        self.llm = llm
        self.max_iterations = max_iterations

    def _global_seed(self, query: str) -> List[Dict[str, Any]]:
        cypher = """
        CALL db.index.fulltext.queryNodes('contentIdx', $query) YIELD node, score
        RETURN node {.id, .name, .description, .type} AS node, score
        ORDER BY score DESC
        LIMIT 20
        """
        return self.neo4j._execute(cypher, {"query": query})

    def _iterate_once(self, user_query: str, evidence: List[Dict[str, Any]], iteration: int) -> Dict[str, Any]:
        subquery = SubqueryPlanner.plan(user_query, evidence)

        cypher = """
        CALL db.index.fulltext.queryNodes('contentIdx', $subquery) YIELD node, score
        MATCH (node)-[:RELATES_TO|BUILD_ON|EVIDENCE_OF]-(adj)
        RETURN adj {.id, .name, .description, .labels} AS node, score
        ORDER BY score DESC
        LIMIT 15
        """
        local_results = self.neo4j._execute(cypher, {"subquery": subquery})

        context_block = merge_contexts(local_results)
        llm_prompt = (
            f"Iteration {iteration}: base user query: '{user_query}'\n"
            f"Subquery: '{subquery}'\n"
            f"Context:\n{context_block}\n"
            "Given the above, generate a short summary and one follow-up subquery."
        )

        response = self.llm.invoke(llm_prompt)
        llm_response = response.content if hasattr(response, "content") else str(response)

        confidence_delta = min(1.0, len(llm_response.split()) / 200.0)

        return {
            "subquery": subquery,
            "results": local_results,
            "context_block": context_block,
            "summary": llm_response,
            "metadata": {
                "confidence_delta": confidence_delta,
                "retrieved": len(local_results),
            },
        }

    def run(self, query: str) -> Dict[str, Any]:
        chain: List[DriftIteration] = []
        evidence = self._global_seed(query)

        for i in range(1, self.max_iterations + 1):
            snapshot = self._iterate_once(query, evidence, i)
            chain.append(
                DriftIteration(
                    iteration=i,
                    sub_query=snapshot["subquery"],
                    nodes_retrieved=snapshot["metadata"]["retrieved"],
                    confidence_delta=snapshot["metadata"]["confidence_delta"],
                )
            )

            if snapshot["metadata"]["confidence_delta"] < 0.1:
                break

            evidence = snapshot["results"]

        final_context = merge_contexts(evidence)
        summary_lines = []
        for item in chain:
            sq = item.sub_query if hasattr(item, 'sub_query') else item.get('sub_query', 'N/A')
            nr = item.nodes_retrieved if hasattr(item, 'nodes_retrieved') else item.get('nodes_retrieved', 0)
            summary_lines.append(f"{sq} (retrieved {nr})")

        synthesis_prompt = (
            f"User question: {query}\n"
            f"DRIFT summary from {len(chain)} iterations:\n"
            + "\n".join(summary_lines)
            + "\n\n" + final_context
            + "\n\nProvide a final concise answer with reasoning and an explicit next research step."
        )

        final_res = self.llm.invoke(synthesis_prompt)
        final_answer = final_res.content if hasattr(final_res, "content") else str(final_res)

        return {
            "drift_chain": chain,
            "final_context": final_context,
            "final_answer": final_answer,
        }