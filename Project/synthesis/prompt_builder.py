from __future__ import annotations
from typing import Any, Dict, List, Union


def build_synthesis_prompt(query: str, route_result: Dict[str, Any], extras: Union[Dict[str, Any], List[Any]] = None) -> str:
    """Create a high-quality LLM synthesis prompt from retrieved routing data."""
    extras = extras or {}

    header = (
        f"You are OmniGraph-RAG synthesis agent. Compose an answer for the question:\n{query}\n"
        "Use the retrieved context and explicitly cite supporting nodes.\n"
    )

    route_type = route_result.get("query_type", "UNKNOWN")
    source = route_result.get("source", "unknown")
    results = route_result.get("results", [])

    lines: List[str] = [header, f"Routing type: {route_type} (source {source})"]

    if isinstance(results, list) and results:
        lines.append("Retrieved nodes and scores:")
        for idx, node in enumerate(results[:12], start=1):
            # Safety net: ensure node is actually a dictionary before calling .get()
            if isinstance(node, dict):
                node_info = node.get("node") or node.get("claim") or node
                name = node_info.get("name") if isinstance(node_info, dict) else str(node_info)
                score = node.get("score", "n/a")
            else:
                name = str(node)
                score = "n/a"
                
            lines.append(f"{idx}. {name} (score={score})")

    # THE BUG FIX: Safely check if extras is a dictionary before looking for the drift chain
    if isinstance(extras, dict) and extras.get("drift_chain"):
        lines.append("\nDRIFT iteration trace:")
        for drop in extras["drift_chain"]:
            lines.append(f"* {drop.get('sub_query', 'unknown')} -> retrieved {drop.get('nodes_retrieved', 0)} nodes")

    lines.append("\nSynthesize a concise and referenced answer with a next-step research suggestion.")

    return "\n".join(lines)