import numpy as np
from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings

class DPHuggingFaceEmbeddings(HuggingFaceEmbeddings):
    """
    A custom wrapper that injects Differential Privacy (Laplacian noise) 
    into document embeddings before they are stored in the local FAISS index.
    """
    epsilon: float = 1.2  # The privacy budget. Lower = More Noise/Privacy.
    sensitivity: float = 0.02 # Max theoretical L2 distance for normalized vectors

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Applies DP noise to documents before indexing."""
        # 1. Generate the original, clean embeddings
        print(f" -> Generating clean embeddings for {len(texts)} chunks...")
        clean_embeddings = super().embed_documents(texts)
        
        # 2. Inject Laplacian Noise for Data-at-Rest Privacy
        print(f" -> Applying DP-RAG Laplacian Noise (epsilon={self.epsilon})...")
        noisy_embeddings = []
        scale = self.sensitivity / self.epsilon
        
        for emb in clean_embeddings:
            emb_array = np.array(emb)
            # Generate mathematical noise from a Laplace distribution
            noise = np.random.laplace(loc=0.0, scale=scale, size=emb_array.shape)
            noisy_emb = emb_array + noise
            noisy_embeddings.append(noisy_emb.tolist())
            
        return noisy_embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Generates a clean embedding for the user's real-time query.
        Queries do not get noise, as they are not stored permanently.
        """
        return super().embed_query(text)