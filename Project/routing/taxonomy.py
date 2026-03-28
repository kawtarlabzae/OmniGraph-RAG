"""
PolyG Query Taxonomy.

Defines the four canonical query archetypes that govern the entire retrieval
strategy selection. Each type maps to a distinct graph traversal algorithm:

  SUBJECT_CENTERED  →  Vector KNN on entity nodes + 1-hop property expansion.
                        Complexity: O(k · d) where k = top-K neighbours,
                        d = embedding dimensionality. Dominated by HNSW ANN.

  OBJECT_DISCOVERY  →  Breadth-first relationship expansion from a seed node.
                        Complexity: O(V + E) in the worst case, bounded in
                        practice by a configurable hop_limit (default: 3).

  FACT_CHECKING     →  Exact-match Cypher triple lookup (subject, predicate,
                        object). Complexity: O(1) via Neo4j relationship index.
                        Falls back to fuzzy vector match if exact fails.

  NESTED            →  Triggers the DRIFT engine: iterative global→local
                        decomposition. Complexity: O(I · k · d) where I is the
                        number of DRIFT iterations (bounded by MAX_DRIFT_ITERS).
"""

from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, Field, field_validator


class QueryType(str, Enum):
    SUBJECT_CENTERED = "SUBJECT_CENTERED"
    OBJECT_DISCOVERY = "OBJECT_DISCOVERY"
    FACT_CHECKING    = "FACT_CHECKING"
    NESTED           = "NESTED"


# ── Routing thresholds ──────────────────────────────────────────────────────

CONFIDENCE_THRESHOLD      = 0.80   # Below this → fallback to NESTED (safer)
DRIFT_MAX_ITERATIONS      = 3      # Hard cap on DRIFT loop depth
DRIFT_CONFIDENCE_FLOOR    = 0.72   # Min cumulative confidence to stop early
DRIFT_MIN_INFO_GAIN       = 0.05   # Stop if marginal gain drops below this


class ClassificationResult(BaseModel):
    """
    Structured output from the PolyG LLM classifier.

    Pydantic model is used here (not TypedDict) because it is the boundary
    between the LLM string output and the typed pipeline state. Validation
    at this boundary catches hallucinated query_type values before they can
    corrupt routing logic downstream.
    """
    query_type: QueryType
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(
        description="One-sentence chain-of-thought justifying the label."
    )
    detected_entities: list[str] = Field(
        default_factory=list,
        description="Named entities extracted from the query for graph anchoring.",
    )
    detected_predicates: list[str] = Field(
        default_factory=list,
        description="Relationship types inferred from the query surface form.",
    )

    @field_validator("confidence")
    @classmethod
    def _clamp_confidence(cls, v: float) -> float:
        """Guard against LLM returning values marginally outside [0, 1]."""
        return max(0.0, min(1.0, v))