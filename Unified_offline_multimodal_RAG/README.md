# MediRAG

MediRAG is an educational, locally run retrieval-augmented generation (RAG) prototype for searching and comparing information in indexed health-record documents. It uses a React/Vite frontend, a FastAPI backend, a local FAISS index, and Ollama for local language and embedding models.

> **Scope:** This is a fictional-data/education prototype, not a clinical decision system. Responses can be incomplete or wrong and must be checked against the cited source passages. It is not certified for HIPAA compliance and does not claim differential privacy or measured clinical accuracy.

## Features

- Search across indexed records, with an optional selected-record filter.
- Combined record search and BMI calculation when a request includes weight and height.
- Compare two selected documents against the same question.
- Add supported local PDF, DOCX, and FHIR JSON documents, or import a small Synthea FHIR ZIP sample through the document library.
- Review returned evidence and source files in the workspace.

## Requirements

- Windows, Python 3.10 or newer, Node.js/npm, and Ollama.
- Ollama models `phi3` and `nomic-embed-text` available locally. Download these models during setup while internet access is available; application inference then runs through the local Ollama service.
- Python packages listed in `multimodal-offline-rag/requirements.txt` and frontend packages listed in `rag-frontend/package.json`.

## Run locally on Windows

Open three PowerShell terminals from the repository folder (`Unified_offline_multimodal_RAG`). Keep Ollama running.

### 1. Backend API

```powershell
cd .\multimodal-offline-rag
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
ollama pull phi3
ollama pull nomic-embed-text
python api.py
```

If the virtual environment and dependencies are already set up, activate it and run `python api.py` instead of repeating installation. The API listens on `http://127.0.0.1:8000`.

### 2. Frontend

```powershell
cd .\rag-frontend
npm install
npm run dev
```

Open the local URL printed by Vite, usually `http://localhost:5173`.

### 3. Check API status

Open `http://127.0.0.1:8000/health`. The API should report that its pipeline is loaded and show whether Ollama and the configured models are available.

## Documents and index

The backend stores uploaded documents and the FAISS index under `multimodal-offline-rag/source_documents` and `multimodal-offline-rag/faissindex`. Add documents through the Document library in the app. Indexing can take time, especially for many or large files, because text is split and embedded locally. Start with a small set of fictional/demo records.

Keep private records, local indexes, feedback logs, and generated backup folders out of public GitHub commits. Review `git status` before committing.

## Configuration

- `OLLAMA_BASE_URL` defaults to `http://127.0.0.1:11434`.
- `MEDIRAG_EMBEDDING_MODEL` defaults to `nomic-embed-text`.
- The chat model is currently configured as `phi3` in the backend.

## Notes

- The app does not send prompts or documents to a hosted AI API; its model requests target the local Ollama service.
- Model/package installation may require internet access. For a fully disconnected machine, install dependencies and download the Ollama models before disconnecting, then transfer the project and required local model files securely.
- The Evaluation/accuracy scoring page has been removed. No accuracy percentage is currently computed.
