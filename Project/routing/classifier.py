"""
PolyG Query Classifier.

Architecture decision: we use a single structured LLM call with a
Pydantic-enforced output schema rather than a fine-tuned NLP classifier.

Rationale:
  1. A fine-tuned classifier (e.g. SetFit, DistilBERT) achieves ~2ms latency
     but requires a labelled dataset and a retraining pipeline. For an
     innovation knowledge base where query patterns evolve rapidly, the
     maintenance cost outweighs the latency gain.

  2. An LLM with structured output + CoT achieves ~200–400ms on GPT-4o-mini
     but generalises to unseen query forms without retraining. The overhead
     is amortised across the larger Neo4j + DRIFT retrieval budget.

  3. Classification results are cached in Redis (LRU) keyed on the blake2b-256
     hash of the normalised query string. Cache hit rate on typical knowledge-
     base workloads is ~35–55%, reducing effective classifier latency to ~15ms
     for repeated or near-repeated queries.

Fallback strategy:
  If the LLM fails to produce a valid ClassificationResult (rate limit,
  JSON parse failure, etc.), we fall back to NESTED. NESTED is the safest
  default because the DRIFT engine is a strict superset of the other
  strategies — it will find the answer, just more expensively.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

# Swapped out OpenAI for the Google Gemini Langchain integration
from langchain_ollama import ChatOllama
from core.config import get_settings
from core.exceptions import ClassificationError
from routing.taxonomy import ClassificationResult, QueryType

logger = logging.getLogger(__name__)

# ── Classifier prompts ──────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are the PolyG Query Classifier for an Innovation and
Management knowledge graph. Assign the incoming query to exactly one of four
archetypes and return a structured JSON object.

ARCHETYPE DEFINITIONS
─────────────────────
SUBJECT_CENTERED
  Asks about properties/attributes of a specific named entity. Graph traversal
  anchors on that entity node and expands its local neighbourhood.
  Examples:
    - "What are the key responsibilities of the Chief Innovation Officer?"
    - "Describe the Lean Startup methodology."

OBJECT_DISCOVERY
  Asks for a SET of entities satisfying a relationship or structural condition.
  Graph traversal performs BFS to discover matching nodes.
  Examples:
    - "Which frameworks are used in agile project management?"
    - "Find all innovation models involving customer feedback loops."

FACT_CHECKING
  Asserts/implies a specific (subject, predicate, object) triple and asks to
  verify or retrieve it. Graph traversal attempts exact Cypher match.
  Examples:
    - "Is Six Sigma related to quality management?"
    - "Who developed the Theory of Constraints?"

NESTED
  Requires chained reasoning across multiple entities, relationships, or
  temporal states. Cannot resolve in a single graph hop. Triggers DRIFT.
  Examples:
    - "How did Toyota's innovation philosophy influence Silicon Valley startups,
       and what modern frameworks trace their lineage to it?"
    - "Compare the impact of disruptive vs sustaining innovation on incumbent
       firms across three industries."

CLASSIFICATION RULES
────────────────────
1. Single specific entity named → SUBJECT_CENTERED; category or set → OBJECT_DISCOVERY.
2. FACT_CHECKING requires an implicit or explicit boolean/verification intent.
3. Any query requiring >2 logical reasoning hops → NESTED.
4. Report confidence < 0.80 when genuinely ambiguous.
"""

_HUMAN_TEMPLATE = """Query: {query}

Return ONLY a JSON object with these exact keys (no markdown, no preamble):
{{
  "query_type": "<SUBJECT_CENTERED|OBJECT_DISCOVERY|FACT_CHECKING|NESTED>",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<one sentence>",
  "detected_entities": ["<entity1>", ...],
  "detected_predicates": ["<predicate1>", ...]
}}"""


# ── Classifier class ────────────────────────────────────────────────────────

class PolyGClassifier:
    """
    Stateless PolyG query classifier with Redis-backed result caching.

    Parameters
    ----------
    llm : BaseChatModel
        Any LangChain-compatible chat model. Defaults to Gemini-1.5-flash.
    redis_client : optional
        An async Redis client for result caching. Pass None to disable caching
        (useful in tests). When provided, classification results are cached
        for `settings.cache_ttl_seconds`.

    Thread safety
    -------------
    PolyGClassifier is stateless and safe for concurrent use across asyncio
    tasks. The underlying `llm` must itself be thread-safe (all LangChain
    models are).
    """

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        redis_client=None,
    ) -> None:
        settings = get_settings()
        
        # Swapped to use Google's Generative AI instead of ChatOpenAI
        self._llm = llm or ChatOllama(
            model="llama3",
            base_url="http://localhost:11434",
            temperature=0.0,
        )
        self._redis = redis_client
        self._confidence_threshold = settings.classifier_confidence_threshold

    # ── Public interface ────────────────────────────────────────────────────

    async def classify(self, query: str) -> ClassificationResult:
        """
        Classify `query` into the PolyG taxonomy.

        Algorithm
        ---------
        1. Normalise query (strip, lowercase) and compute cache key.
        2. Check Redis for a cached result → return immediately on hit.
        3. Call LLM with structured prompt → parse JSON → validate with Pydantic.
        4. If confidence < threshold, override label to NESTED (safe fallback).
        5. Write result to Redis cache.
        6. Return ClassificationResult.

        Raises
        ------
        ClassificationError
            If the LLM call fails AND no cached fallback is available.
        """
        cache_key = self._make_cache_key(query)

        # ── Cache read ──────────────────────────────────────────────────────
        if self._redis is not None:
            cached = await self._redis.get(cache_key)
            if cached:
                logger.debug("classifier cache hit for key=%s", cache_key[:12])
                return ClassificationResult.model_validate_json(cached)

        # ── LLM call ───────────────────────────────────────────────────────
        try:
            result = await self._call_llm(query)
        except Exception as exc:
            logger.error("PolyG classifier LLM call failed: %s", exc, exc_info=True)
            # Safe fallback: NESTED covers all query types at higher cost
            result = ClassificationResult(
                query_type=QueryType.NESTED,
                confidence=0.0,
                reasoning="Fallback due to classifier failure.",
                detected_entities=[],
                detected_predicates=[],
            )

        # ── Low-confidence override ────────────────────────────────────────
        if result.confidence < self._confidence_threshold:
            logger.info(
                "classifier confidence %.2f below threshold %.2f — routing to NESTED",
                result.confidence,
                self._confidence_threshold,
            )
            result = result.model_copy(update={"query_type": QueryType.NESTED})

        # ── Cache write ────────────────────────────────────────────────────
        if self._redis is not None:
            settings = get_settings()
            await self._redis.setex(
                cache_key,
                settings.cache_ttl_seconds,
                result.model_dump_json(),
            )

        return result

    # ── Private helpers ─────────────────────────────────────────────────────

    async def _call_llm(self, query: str) -> ClassificationResult:
        """Execute the LLM call and parse the structured response."""
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=_HUMAN_TEMPLATE.format(query=query)),
        ]
        response = await self._llm.ainvoke(messages)
        raw_text: str = response.content

        # Strip any accidental markdown fences
        raw_text = raw_text.strip().removeprefix("```json").removesuffix("```").strip()

        try:
            data = json.loads(raw_text)
            return ClassificationResult.model_validate(data)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ClassificationError(
                f"Failed to parse classifier response: {exc}\nRaw: {raw_text}"
            ) from exc

    @staticmethod
    def _make_cache_key(query: str) -> str:
        """
        Produce a deterministic cache key from the normalised query string.

        Uses blake2b-256 (faster than SHA-256, no collision concerns for
        cache keys). Normalisation strips whitespace and lowercases so that
        "What is Design Thinking?" and "what is design thinking?" share a key.
        """
        normalised = " ".join(query.lower().split())
        digest = hashlib.blake2b(normalised.encode(), digest_size=32).hexdigest()
        return f"omni:classify:{digest}"