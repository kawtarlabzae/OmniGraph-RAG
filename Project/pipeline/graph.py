from __future__ import annotations
from typing import Any, Dict, List

from omni_graph_rag.router import QueryRouter
from synthesis.generator import ResponseSynthesizer


class OmniGraphPipeline:
    """End-to-end OmniGraph-RAG pipeline orchestration component."""

    def __init__(self, router: QueryRouter, synthesizer: ResponseSynthesizer):
        self.router = router
        self.synthesizer = synthesizer

    def execute(self, query: str) -> Dict[str, Any]:
        """Run query routing + retrieval + synthesis workflow."""
        route_payload = self.router.route(query)

        source_context = route_payload.get("results")

        final_text = self.synthesizer.synthesize(query, route_payload)

        return {
            "route": route_payload,
            "final_text": final_text,
        }
