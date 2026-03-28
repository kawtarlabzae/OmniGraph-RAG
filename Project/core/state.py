"""
OmniGraph-RAG shared pipeline state.

The OmniGraphState is the single source of truth threaded through every
LangGraph node. It is deliberately flat (TypedDict, not a class) to remain
trivially serialisable across async checkpointers (Redis, Postgres) while
still providing O(1) key-access semantics for the hot retrieval path.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal
from typing_extensions import TypedDict
import operator


QueryType = Literal[
    "SUBJECT_CENTERED",
    "OBJECT_DISCOVERY",
    "FACT_CHECKING",
    "NESTED",
]


class RetrievedNode(TypedDict):
    """A single scored node returned from any Neo4j retrieval strategy."""
    node_id: str
    labels: list[str]
    properties: dict[str, Any]
    score: float                    # cosine sim | BFS depth score | 1.0 for exact
    source_strategy: str            # which retrieval handler produced this


class DriftIteration(TypedDict):
    """One complete DRIFT loop iteration snapshot (for debugging & tracing)."""
    iteration: int
    sub_query: str
    nodes_retrieved: int
    confidence_delta: float         # marginal info gain vs previous iteration


class OmniGraphState(TypedDict):
    """
    Immutable-by-convention shared state for the OmniGraph-RAG LangGraph pipeline.

    Design rationale:
    - All fields are explicitly typed to catch schema drift at import time.
    - `retrieved_nodes` uses `operator.add` as the reducer so parallel
      retrieval branches can safely append without clobbering each other.
    - `drift_iterations` accumulates audit breadcrumbs; never overwritten.
    - `final_answer` is None until the synthesis node writes it exactly once.
    """
    # ── Input ──────────────────────────────────────────────────────────────
    query: str

    # ── Routing ────────────────────────────────────────────────────────────
    query_type: QueryType | None
    classifier_confidence: float            # [0, 1]; route is trusted if > 0.85
    classifier_reasoning: str              # LLM chain-of-thought for the label

    # ── Retrieval (reducer enables parallel branch merging) ─────────────
    retrieved_nodes: Annotated[list[RetrievedNode], operator.add]

    # ── DRIFT-specific ─────────────────────────────────────────────────────
    drift_iterations: Annotated[list[DriftIteration], operator.add]
    drift_context_satisfied: bool           # True once threshold met

    # ── Synthesis ──────────────────────────────────────────────────────────
    assembled_context: str                  # Prompt-ready merged context block
    final_answer: str | None
    citations: list[str]                    # node_ids cited in the final answer

    # ── Observability ──────────────────────────────────────────────────────
    trace_id: str
    latency_ms: dict[str, float]           # keyed by node name