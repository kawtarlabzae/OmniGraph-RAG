from __future__ import annotations
from typing import Any, Dict, List


def merge_contexts(retrieved_nodes: List[Dict[str, Any]], max_chars: int = 2500) -> str:
    """Merge multiple retrieved node summaries into one prompt-ready context block."""
    fragments: List[str] = []

    for node in retrieved_nodes:
        data = node.get("node") or node.get("doc") or node
        if not isinstance(data, dict):
            continue

        title = data.get("name") or data.get("id") or "unknown"
        desc = data.get("description") or data.get("summary") or ""
        fragment = f"- {title}: {desc}" if desc else f"- {title}"
        fragments.append(fragment)

    merged = "\n".join(fragments)
    if len(merged) > max_chars:
        merged = merged[:max_chars].rsplit("\n", 1)[0]

    if not merged:
        return "No context could be merged from this iteration."

    return merged
