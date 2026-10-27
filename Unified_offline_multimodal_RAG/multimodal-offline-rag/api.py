from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from threading import Lock
from urllib.request import urlopen
import uvicorn

from src.agentic_pipeline import AgenticRAG
from src.ingestion_service import (
    MAX_UPLOAD_BYTES,
    add_documents,
    delete_document,
    import_synthea_zip,
    list_documents,
)

ROOT = Path(__file__).resolve().parent
app = FastAPI(
    title="MediRAG Local API",
    description="Local document search, analysis, and comparison endpoints.",
    version="3.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:4173", "http://127.0.0.1:4173"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)

pipeline: AgenticRAG | None = None
INDEX_LOCK = Lock()


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    mode: Literal["auto", "search", "calculate", "both", "check_interaction", "validate_labs"] = "auto"
    source: str | None = Field(default=None, max_length=255)


class CompareRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    first_source: str
    second_source: str


class FeedbackRequest(BaseModel):
    rating: Literal["helpful", "unhelpful"]
    session_id: str | None = Field(default=None, max_length=100)
    response_id: str | None = Field(default=None, max_length=100)


@app.on_event("startup")
async def startup_event():
    global pipeline
    print("Loading MediRAG pipeline and local vector index...")
    pipeline = AgenticRAG()
    print("MediRAG API is ready.")


def _local_ollama_status() -> dict[str, bool]:
    """Check the local Ollama server and configured model without contacting the internet."""
    try:
        with urlopen("http://127.0.0.1:11434/api/tags", timeout=1.5) as response:
            models = json.loads(response.read().decode("utf-8")).get("models", [])
        names = {str(item.get("name", "")).split(":", 1)[0] for item in models}
        embedding_name = os.getenv("MEDIRAG_EMBEDDING_MODEL", "nomic-embed-text").split(":", 1)[0]
        return {
            "ollama_available": True,
            "llm_model_available": "phi3" in names,
            "embedding_model_available": embedding_name in names,
        }
    except Exception:
        return {"ollama_available": False, "llm_model_available": False, "embedding_model_available": False}


@app.get("/health")
async def health_check():
    status = await run_in_threadpool(_local_ollama_status)
    return {"status": "online", "model_loaded": pipeline is not None, **status}


def _ready() -> AgenticRAG:
    if pipeline is None:
        raise HTTPException(status_code=503, detail="The MediRAG pipeline is not initialized.")
    return pipeline


def _refresh_index() -> None:
    try:
        _ready().reload_index()
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Documents were saved, but the index reload failed: {error}") from error


def _run_index_operation(operation, *args):
    """Run CPU-heavy index work away from the API event loop and serialize writes."""
    with INDEX_LOCK:
        result = operation(*args)
        _refresh_index()
        return result


@app.get("/documents")
async def documents_endpoint():
    return {"documents": list_documents(), "count": len(list_documents())}


@app.post("/documents/upload")
async def upload_documents(files: list[UploadFile] = File(...)):
    if not files or len(files) > 20:
        raise HTTPException(status_code=400, detail="Upload between 1 and 20 supported files at a time.")
    payloads = []
    total = 0
    for file in files:
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        total += len(data)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="The combined upload exceeds 60 MB.")
        payloads.append((file.filename or "", data))
    try:
        result = await run_in_threadpool(_run_index_operation, add_documents, payloads)
    except (ValueError, OSError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return result


@app.post("/documents/import-synthea")
async def import_synthea(file: UploadFile = File(...), patient_limit: int = Form(5)):
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="The Synthea ZIP exceeds 60 MB.")
    try:
        result = await run_in_threadpool(_run_index_operation, import_synthea_zip, data, patient_limit)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return result


@app.post("/documents/reindex")
async def reindex_documents():
    from src.ingestion_service import _rebuild_index
    try:
        count = await run_in_threadpool(_run_index_operation, _rebuild_index)
    except (ValueError, OSError, RuntimeError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"indexed_documents": count}


@app.delete("/documents/{filename}")
async def remove_document(filename: str):
    try:
        result = await run_in_threadpool(_run_index_operation, delete_document, filename)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except (ValueError, OSError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return result


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    print(f"Chat request received mode={request.mode} source_filter={'selected' if request.source else 'all'}")
    try:
        result = await run_in_threadpool(_ready().ask, request.query.strip(), request.mode, request.source)
        print(f"Chat route={result.get('route', 'unknown')} evidence_chunks={len(result.get('sources', []))} source_filter={'selected' if request.source else 'all'}")
        return {
            "answer": result["result"].strip(),
            "route": result.get("route", "search"),
            "sources": result.get("sources", []),
        }
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.post("/compare")
async def compare_endpoint(request: CompareRequest):
    try:
        return await run_in_threadpool(_ready().compare, request.query.strip(), request.first_source, request.second_source)
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error




@app.post("/feedback")
async def feedback_endpoint(request: FeedbackRequest):
    record = {
        "rating": request.rating,
        "session_id": request.session_id,
        "response_id": request.response_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    path = ROOT / "feedback.jsonl"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")
    return {"saved": True}


if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=False)
