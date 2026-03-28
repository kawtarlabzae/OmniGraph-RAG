from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional
import os

# 🔴 REMOVED: from langchain_google_genai import ChatGoogleGenerativeAI
# 🟢 ADDED: Ollama import
from langchain_ollama import ChatOllama
from langchain_core.language_models import BaseLanguageModel
from langchain_core.outputs import LLMResult

from core.config import get_settings
from drift.engine import DriftEngine
from .neo4j_client import Neo4jClient


class PolyGCategory(str, Enum):
    SUBJECT_CENTERED = "subject_centered"
    OBJECT_DISCOVERY = "object_discovery"
    FACT_CHECK = "fact_check"
    NESTED = "nested"


@dataclass
class QueryRouter:
    """OmniGraph-RAG adaptive query router."""
    neo4j_client: Neo4jClient
    drift_engine: DriftEngine
    llm: BaseLanguageModel

    def classify_query(self, query: str) -> PolyGCategory:
        q = query.strip().lower()

        if any(token in q for token in ["compare", "difference", "versus", "vs"][::-1]):
            return PolyGCategory.SUBJECT_CENTERED

        if any(token in q for token in ["discover", "find", "identify", "list"]):
            return PolyGCategory.OBJECT_DISCOVERY

        if any(token in q for token in ["true", "false", "verify", "validate", "is it"]):
            return PolyGCategory.FACT_CHECK

        if any(token in q for token in ["how", "why", "workflow", "sequence", "nested", "multi-hop"]):
            return PolyGCategory.NESTED

        prompt = (
            "Classify the following user intent into one of: subject_centered, object_discovery, "
            "fact_check, nested. Respond with only the single category token.\n\n"
            f"User query: '{query}'"
        )
        
        llm_response = self.llm.invoke(prompt)
        response_text = llm_response.content if hasattr(llm_response, "content") else str(llm_response)
        normalized = response_text.strip().lower()

        for cat in PolyGCategory:
            if cat.value in normalized:
                return cat

        return PolyGCategory.NESTED

    def route(self, query: str) -> Dict[str, Any]:
        category = self.classify_query(query)
        payload: Dict[str, Any] = {"category": category.value, "query": query}

        if category == PolyGCategory.SUBJECT_CENTERED:
            results = self.neo4j_client.subject_search(query)
            payload.update({"path": "neo4j_subject", "results": results})
            return payload

        if category == PolyGCategory.OBJECT_DISCOVERY:
            primary_target = query.split(" ")[0]
            results = self.neo4j_client.object_discovery(primary_target)
            payload.update({"path": "neo4j_object_discovery", "results": results})
            return payload

        if category == PolyGCategory.FACT_CHECK:
            results = self.neo4j_client.fact_check(query)
            payload.update({"path": "neo4j_fact_check", "results": results})
            return payload

        drift_result = self.drift_engine.run(query) 
        payload.update({"path": "drift_nested", "results": drift_result})
        return payload


def make_router_from_env(env: Optional[Dict[str, str]] = None) -> QueryRouter:
    env = env or os.environ 
    
    neo4j_uri = env.get("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = env.get("NEO4J_USER", "neo4j")
    neo4j_password = env.get("NEO4J_PASSWORD", "password")

    # 🟢 CHANGED: Using local Llama 3 instead of Gemini
    llm = ChatOllama(
        model="llama3", 
        base_url=env.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0.0 # Set to 0.0 for strict, factual logic
    )

    neo4j_client = Neo4jClient(neo4j_uri, neo4j_user, neo4j_password)
    drift_engine = DriftEngine(neo4j_client, llm)

    return QueryRouter(neo4j_client=neo4j_client, drift_engine=drift_engine, llm=llm)