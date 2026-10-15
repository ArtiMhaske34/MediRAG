# MediRAG — Local Healthcare RAG Assistant

A privacy-oriented healthcare question-answering project using Retrieval-Augmented Generation (RAG), local language models, and a web interface.

## Overview

MediRAG is a prototype that lets users ask questions about medical documents stored locally. It retrieves relevant document passages and uses a language model to generate an answer with source-document references.

The project combines Python, FastAPI, FAISS, Ollama, and a React frontend.

**Project status:** Local prototype tested with sample PDF documents. This is an educational project, not a certified clinical decision-support system.

## Features

- **Local document retrieval:** Loads medical documents and searches a local FAISS vector index.
- **Retrieval-Augmented Generation:** Uses retrieved context to help answer questions about the documents.
- **Embedding noise experiment:** Applies Laplacian noise to document embeddings using a configurable epsilon value.
- **Local language model:** Integrates with Ollama and the Phi-3 model.
- **FastAPI backend:** Provides an API for submitting questions.
- **React frontend:** Offers a web interface for interacting with the backend.
- **Source references:** Returns retrieved source documents alongside answers.

## Technology Stack

- Python
- FastAPI
- LangChain
- FAISS
- Sentence Transformers
- Ollama / Phi-3
- React
- Differential Privacy-inspired embedding noise

## Architecture

1. **Document ingestion:** Read supported documents and prepare text for retrieval.
2. **Embedding generation:** Convert document text into vectors.
3. **Privacy experiment:** Apply Laplacian noise to document embeddings.
4. **Vector search:** Store and retrieve vectors using FAISS.
5. **Answer generation:** Use the local language model to answer questions based on retrieved context.
6. **Web interface:** Submit questions through the React frontend and display answers.

## Prerequisites

- Python version compatible with the installed dependencies
- Node.js and npm for the frontend
- Ollama installed locally
- The Phi-3 model downloaded through Ollama
- Sufficient RAM and disk space for the model and dependencies

## Setup

### 1. Install Ollama and the model

Install Ollama from its official website, then run:

```bash
ollama pull phi3
```

### 2. Set up the Python environment

From the backend directory:

```bash
python -m venv venv
```

Activate the environment on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 3. Add sample documents

Place only synthetic, non-identifiable, and legally shareable sample documents in `source_documents/`. Do not use real patient records or confidential healthcare data.

### 4. Build the vector index

```bash
python ingest.py
```

This generates a local FAISS index. The generated index should not normally be committed to Git.

### 5. Start the backend

```bash
python api.py
```

Check the API configuration and terminal output for the local server address.

### 6. Start the frontend

Open a separate terminal:

```bash
cd ../rag-frontend
npm install
npm run dev
```

Use the local URL printed by the frontend development server.

## Privacy and Safety Notes

- Keep patient records and confidential data out of public repositories.
- Local processing can reduce external data sharing, but it does not by itself guarantee absolute privacy.
- Adding noise to embeddings is not, by itself, proof of a formal Differential Privacy guarantee. Such a guarantee requires a suitable mechanism, sensitivity analysis, and accounting of the complete data-processing pipeline.
- A local FAISS index should not be described as encrypted unless encryption is actually implemented and verified.
- This prototype is not HIPAA-certified and should not be used to make medical decisions.

## Future Improvements

- Evaluate retrieval quality and answer grounding.
- Add automated tests and error handling.
- Strengthen privacy analysis and document the threat model.
- Improve setup instructions and dependency management.
- Evaluate the agentic routing features separately before describing them as implemented.

