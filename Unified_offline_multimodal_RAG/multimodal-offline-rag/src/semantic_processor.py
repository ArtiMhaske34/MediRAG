import os
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


class SemanticProcessor:
    def __init__(self, index_dir: str = "faissindex"):
        self.index_dir = index_dir
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=400,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
        print("Connecting to local Ollama embedding model (nomic-embed-text)...")
        self.embeddings = OllamaEmbeddings(
            model=os.getenv("MEDIRAG_EMBEDDING_MODEL", "nomic-embed-text"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            client_kwargs={"timeout": 120},
        )
        self.vectorstore = None

    def chunk_and_embed(self, text_corpus: List[Dict[str, str]], replace: bool = False):
        """
        Takes raw text, chunks it, embeds it, and adds it to the FAISS index.
        text_corpus expected format: [{"text": "...", "source": "file.pdf"}]
        """
        documents = []
        for item in text_corpus:
            # Create Langchain Document objects with metadata
            doc = Document(page_content=item["text"], metadata={"source": item["source"]})
            documents.append(doc)
            
        print(f"Splitting {len(documents)} source texts into semantic chunks...")
        chunks = self.text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks.")

        print("Computing embeddings and building FAISS index...")
        if not replace and os.path.exists(os.path.join(self.index_dir, "index.faiss")):
            # Load existing index and add new chunks
            self.vectorstore = FAISS.load_local(self.index_dir, self.embeddings, allow_dangerous_deserialization=True)
            self.vectorstore.add_documents(chunks)
        else:
            # Create a new index from scratch
            self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
            
        self._save_index()

    def _save_index(self):
        """Persists the FAISS index and metadata to disk[cite: 563]."""
        os.makedirs(self.index_dir, exist_ok=True)
        self.vectorstore.save_local(self.index_dir)
        print(f"Successfully saved FAISS index to {self.index_dir}/")
