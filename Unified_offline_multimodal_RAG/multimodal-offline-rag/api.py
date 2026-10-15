from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn

# Import your existing pipeline
from src.rag_pipeline import RAGPipeline
from src.agentic_pipeline import AgenticRAG

# Initialize FastAPI app
app = FastAPI(
    title="Offline Multimodal RAG API",
    description="Privacy-preserving local API for document retrieval and generation",
    version="2.0.0"
)

# Enable CORS so our React frontend can communicate with this API safely
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, change this to your React app's local URL (e.g., http://localhost:3000)
    allow_credentials=True,
    
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to hold our pipeline
pipeline = None

# Define the data structures for our API requests and responses
class ChatRequest(BaseModel):
    query: str

class SourceDocument(BaseModel):
    filename: str
    content_snippet: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]

@app.on_event("startup")
async def startup_event():
    """Loads the heavy LLM and FAISS index into memory when the server starts."""
    global pipeline
    print("Initializing RAG Pipeline (Loading LLM and Vector Store)...")
    
    # We removed the try-except block. If this fails, the server will crash
    # and print the exact Python error traceback, making it easy to fix!
    pipeline = AgenticRAG()
    print("✅ Pipeline successfully loaded and ready for requests!")
    
@app.get("/health")
async def health_check():
    """Simple endpoint to verify the server is running."""
    return {"status": "online", "model_loaded": pipeline is not None}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """The main endpoint that receives a question and returns the AI's answer."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="RAG Pipeline is not initialized.")
    
    try:
        # Run the query through your existing LangChain setup
        response = pipeline.ask(request.query)
        
        # Parse the sources to send back to the frontend
        formatted_sources = []
        for doc in response.get('source_documents', []):
            formatted_sources.append(
                SourceDocument(
                    filename=doc.metadata.get('source', 'Unknown'),
                    content_snippet=doc.page_content[:200] + "..." # Send a preview of the text
                )
            )
            
        return ChatResponse(
            answer=response['result'].strip(),
            sources=formatted_sources
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Run the server on port 8000
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)