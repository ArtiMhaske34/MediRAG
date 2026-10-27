import os
from typing import List

# Do not let model startup contact the Hugging Face Hub in this local prototype.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings

class NoisyHuggingFaceEmbeddings(HuggingFaceEmbeddings):
    """Experimental embedding noise; this class does not provide a formal privacy guarantee."""
    noise_scale: float = 0.02 / 1.2

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Adds experimental Laplace noise to document embeddings before indexing."""
        # 1. Generate the original, clean embeddings
        print(f" -> Generating clean embeddings for {len(texts)} chunks...")
        clean_embeddings = super().embed_documents(texts)
        
        # 2. Add experimental noise; this is not a formal privacy guarantee
        print(f" -> Applying experimental embedding noise (scale={self.noise_scale})...")
        noisy_embeddings = []
        scale = self.noise_scale
        
        for emb in clean_embeddings:
            emb_array = np.array(emb)
            # Generate Laplace-distributed experimental noise
            noise = np.random.laplace(loc=0.0, scale=scale, size=emb_array.shape)
            noisy_emb = emb_array + noise
            noisy_embeddings.append(noisy_emb.tolist())
            
        return noisy_embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Generates an unmodified embedding for a real-time query.
        """
        return super().embed_query(text)