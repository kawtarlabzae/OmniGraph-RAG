# OmniGraph-RAG Architecture Blueprint

## 1. System Overview
OmniGraph-RAG unifies:
- PolyG Query Planning (query taxonomy classifier)
- DRIFT Search (iterative inference + dynamic subquery generation)
- Neo4j Backbone (hybrid vector/graph retrieval)

It is designed for Innovation and Management knowledge domains with an emphasis on relevance, reasoning, and traceable decision paths.

## 2. Data Flow (User Prompt → Answer)
1. User sends natural language prompt to API.
2. FastAPI `query` endpoint forwards prompt to `QueryRouter`.
3. `QueryRouter.classify_query()` assigns one of the PolyG taxonomy categories:
   - `SUBJECT_CENTERED`
   - `OBJECT_DISCOVERY`
   - `FACT_CHECK`
   - `NESTED`
4. Based on category:
   - `SUBJECT_CENTERED` → direct Neo4j vector/graph retrieval (`Neo4jBridge.subject_search`).
   - `OBJECT_DISCOVERY` → broader graph exploration query plus related local contexts.
   - `FACT_CHECK` → Neo4j constraint-based lookup + citation evidence chain.
   - `NESTED` → DRIFT workflow (`DriftEngine.iterate()`) for multi-hop reasoning and follow-ups.
5. Raw retrieval blocks are fused by `LLMSynthesizer` for a coherent response.

## 3. PolyG + DRIFT Integration
- `QueryRouter` is the brain; bounds query path decisions via cost heuristics and taxonomy.
- `DriftEngine` handles iterative reasoning for nested flows:
  - map stage: global community summaries from initial retrieval
  - reduce stage: candidate hypothesis ranking
  - followup stage: dynamic node-level cypher expansions

## 4. Neo4j Hybrid Search
- Vector embeddings drive similarity search using `gds.alpha.knn.stream` or `CALL db.index.fulltext.queryNodes`.
- Graph structural constraints use labeled paths and pattern matching.
- Schema conventions:
  - `InnovationConcept`, `ManagementEntity`, `Claim` nodes
  - `RELATES_TO`, `EVIDENCE_OF`, `BUILD_ON` edges

## 5. Extensibility
- Each component can be replaced via DI:
  - swap `OpenAI` with Mistral or local LLM
  - swap Neo4j with `Neo4j Aura` by changing URI/
  - add `metrics` in router decisions

---
## 6. Component TL;DR
- `omni_graph_rag.neo4j_client.py`: Neo4j wrapper.
- `omni_graph_rag.drift.py`: DRIFT pipeline numerical loop.
- `omni_graph_rag.router.py`: polyglot classifier + conditional router.
- `api/main.py`: FastAPI endpoint and startup wiring.
