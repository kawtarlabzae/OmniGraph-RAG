import os
from dotenv import load_dotenv

load_dotenv()

from unittest.mock import MagicMock
from fastapi import FastAPI
from pydantic import BaseModel

# 🔴 REMOVED: from langchain_google_genai import ChatGoogleGenerativeAI
# 🟢 ADDED: Ollama import
from langchain_ollama import ChatOllama

# Internal project imports
from core.config import get_settings
from routing.router import OmniGraphRouter
from synthesis.generator import ResponseSynthesizer
from omni_graph_rag.router import QueryRouter
from drift.engine import DriftEngine

app = FastAPI(title="OmniGraph-RAG API")
settings = get_settings()

if os.getenv("MOCK_MODE", "false").lower() == "true":
    print("⚠️ RUNNING IN MOCK MODE")
    mock_client = MagicMock()
    mock_client.subject_search.return_value = [{"node": {"name": "Mocked Innovation Concept"}, "score": 0.9}]
    mock_client.object_discovery.return_value = [{"node": {"name": "Mocked Related Entity"}, "score": 0.8}]
    mock_client.fact_check.return_value = [{"claim": {"claimText": "Mocked Claim Verified"}, "score": 0.95}]
    
    mock_llm = MagicMock()
    mock_llm.return_value = "Mocked LLM response for synthesis."
    
    mock_query_router = QueryRouter(
        neo4j_client=mock_client, 
        drift_engine=DriftEngine(mock_client, mock_llm), 
        llm=mock_llm
    )
    router = OmniGraphRouter(router=mock_query_router)
    synthesizer = ResponseSynthesizer(mock_llm)

else:
    print("🚀 RUNNING IN PRODUCTION MODE (Ollama Engine Active)")
    
    # 🟢 CHANGED: Using local Llama 3 instead of Gemini
    llm = ChatOllama(
        model="llama3",
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0.0
    )
    
    router = OmniGraphRouter()
    synthesizer = ResponseSynthesizer(llm)


class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    query_type: str
    source: str
    final_text: str


@app.post("/query", response_model=QueryResponse)
async def run_query(body: QueryRequest) -> QueryResponse:
    route_data = await router.route(body.query)
    final = synthesizer.synthesize(body.query, route_data)

    return QueryResponse(
        query=body.query,
        query_type=route_data.get("query_type", "UNKNOWN"),
        source=route_data.get("source", "UNKNOWN"),
        final_text=final,
    )