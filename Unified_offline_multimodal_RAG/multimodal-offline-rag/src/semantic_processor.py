import os
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.dp_embeddings import DPHuggingFaceEmbeddings

class SemanticProcessor:
    def __init__(self, index_dir: str = "faissindex"):
        self.index_dir = index_dir
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=400,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
        # USE THE DP EMBEDDING CLASS INSTEAD OF THE STANDARD ONE
        print("Loading Differentially Private Embedding Model...")
        self.embeddings = DPHuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            epsilon=1.2 # Adjust this value to test the Privacy-Utility trade-off
        )
        self.vectorstore = None

    def chunk_and_embed(self, text_corpus: List[Dict[str, str]]):
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

        print("Computing embeddings and building FAISS Flat L2 index... [cite: 561-562]")
        if os.path.exists(os.path.join(self.index_dir, "index.faiss")):
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