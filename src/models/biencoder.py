import numpy as np
from typing import Optional
from sentence_transformers import SentenceTransformer
import torch
import warnings

from .base import BaseRetriever

class BiEncoderRetriever(BaseRetriever):
    """
    Standard Dense Semantic Retriever (Bi-Encoder).
    Uses sentence-transformers to encode corpus documents and queries.
    Performs cosine similarity search using PyTorch operations.
    """
    def __init__(self, corpus: list[dict], model_name: str = "all-MiniLM-L6-v2", batch_size: int = 64):
        super().__init__(corpus)
        
        # Suppress warnings
        warnings.filterwarnings("ignore", category=FutureWarning)
        
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        
        doc_texts = [rec["text"] for rec in corpus]
        print(f"  Encoding {len(doc_texts)} documents with {model_name} on {self.device}...")
        
        # Encode corpus documents. Normalizing embeddings allows us to use simple dot product for cosine similarity.
        self.doc_embeddings = self.model.encode(
            doc_texts, 
            batch_size=batch_size, 
            show_progress_bar=True, 
            convert_to_numpy=True, 
            normalize_embeddings=True
        )
        # Store tensor on device for fast scoring
        self.doc_embeddings_tensor = torch.tensor(self.doc_embeddings, device=self.device, dtype=torch.float32)

    def search(self, query: str, session_history: list[str], top_k: int = 10,
               filter_ids: Optional[set[str]] = None, **kwargs) -> list[tuple[str, float]]:
        
        # We concatenate the recent history + current query to provide some session context 
        # to the stateless bi-encoder.
        context = session_history[-3:]
        if context:
            full_query = " ".join(context + [query])
        else:
            full_query = query
            
        # Encode query
        query_emb = self.model.encode([full_query], convert_to_numpy=True, normalize_embeddings=True)[0]
        query_tensor = torch.tensor(query_emb, device=self.device, dtype=torch.float32)
        
        # Compute cosine similarity (dot product since both are normalized)
        # Matrix-vector multiplication: (N_docs, dim) x (dim,) -> (N_docs,)
        scores = torch.mv(self.doc_embeddings_tensor, query_tensor).cpu().numpy()
        
        # Sort and get top-k
        top_indices = np.argsort(scores)[::-1]
        
        results = []
        for idx in top_indices:
            nid = self.note_ids[idx]
            if filter_ids is None or nid in filter_ids:
                results.append((nid, float(scores[idx])))
            if len(results) >= top_k:
                break
                
        return results

    def __getstate__(self):
        # When pickling, drop the large tensor and model to save space.
        # This allows joblib.dump to work without blowing up memory.
        # In a real system, you'd save/load the embeddings separately.
        state = self.__dict__.copy()
        state["doc_embeddings_tensor"] = None
        state["model"] = None
        return state
        
    def __setstate__(self, state):
        self.__dict__.update(state)
        # Re-initialize the model and tensor when loaded
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)
        if self.doc_embeddings is not None:
            self.doc_embeddings_tensor = torch.tensor(self.doc_embeddings, device=self.device, dtype=torch.float32)

