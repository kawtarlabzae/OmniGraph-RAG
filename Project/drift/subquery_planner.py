from __future__ import annotations
from typing import Any, Dict, List


class SubqueryPlanner:
    """DRIFT subquery planner for adaptive next-hop generation.

    This planner is lightweight and deterministic.
    It extracts key entity mentions from the current node set and uses them
    to produce focused follow-up queries.

    Complexity:
    - O(n*m) where n is evidence node set and m is number of extracted keywords.
    - Keeps the number of generated tokens bounded by configuration.
    """

    @staticmethod
    def plan(primary_query: str, evidence_nodes: List[Dict[str, Any]], max_tokens: int = 128) -> str:
        if not evidence_nodes:
            return primary_query

        top_labels = []
        for node in evidence_nodes[:8]:
            node_data = node.get("node") or node.get("doc") or node
            name = node_data.get("name") if isinstance(node_data, dict) else None
            if name and name not in top_labels:
                top_labels.append(name)

        anchor_terms = ", ".join(top_labels[:4])
        if not anchor_terms:
            return primary_query

        subquery = (
            f"Deep-dive into the following terms from the context: {anchor_terms}. "
            f"Based on the original goal '{primary_query}', generate a concise follow-up question "
            "to explore relationships and causal chains among those terms."
        )

        # trim to max tokens assumption
        return subquery[:max_tokens]
