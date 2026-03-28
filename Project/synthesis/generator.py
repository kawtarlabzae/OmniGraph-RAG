from __future__ import annotations
from typing import Any, Dict

from langchain_core.language_models import BaseLanguageModel
from .prompt_builder import build_synthesis_prompt


class ResponseSynthesizer:
    """LLM synthesis engine for combining retrieval payload into final answer."""

    def __init__(self, llm: BaseLanguageModel):
        self.llm = llm

    def synthesize(self, query: str, route_payload: Dict[str, Any]) -> str:
        prompt = build_synthesis_prompt(query, route_payload, extras=route_payload.get("results", {}))
        
        # Use .invoke() to properly call the modern Chat Model
        response = self.llm.invoke(prompt)
        
        # Safely extract text whether it's an AIMessage object or a raw string
        return response.content if hasattr(response, "content") else str(response)