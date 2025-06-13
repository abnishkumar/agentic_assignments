from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
import numpy as np
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
import os
from dotenv import load_dotenv
load_dotenv()

class RerankerConfig(BaseModel):
    method: str = "bm25"  # "bm25" or "mmr"
    diversity_bias: float = 0.5  # For MMR
    top_k: int = 5
    embedding_model: str =  os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

class Reranker:
    def __init__(self, config: RerankerConfig):
        self.config = config
        self.model = SentenceTransformer(self.config.embedding_model)

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25."""
        return text.lower().split()

    def rerank_bm25(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank documents using BM25."""
        if not documents:
            return []
        tokenized_docs = [self._tokenize(doc['text']) for doc in documents]
        tokenized_query = self._tokenize(query)
        bm25 = BM25Okapi(tokenized_docs)
        scores = bm25.get_scores(tokenized_query)
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:self.config.top_k]]

    def rerank_mmr(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank documents using MMR (Maximal Marginal Relevance)."""
        if not documents:
            return []
        # Get embeddings
        query_embedding = self.model.encode(query)
        doc_embeddings = self.model.encode([doc['text'] for doc in documents])

        # Handle empty embeddings
        if len(doc_embeddings) == 0:
            return []

        # Calculate similarity matrix
        similarity_matrix = np.dot(doc_embeddings, query_embedding)

        # Initialize selected documents
        selected_indices = []
        remaining_indices = list(range(len(documents)))

        # Select first document (most similar to query)
        first_idx = int(np.argmax(similarity_matrix))
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)

        # MMR selection
        while len(selected_indices) < self.config.top_k and remaining_indices:
            mmr_scores = []
            for idx in remaining_indices:
                # Relevance to query
                relevance = similarity_matrix[idx]
                # Diversity penalty
                diversity = 0
                for selected_idx in selected_indices:
                    diversity += np.dot(doc_embeddings[idx], doc_embeddings[selected_idx])
                diversity /= len(selected_indices)
                # MMR score
                mmr_score = self.config.diversity_bias * relevance - (1 - self.config.diversity_bias) * diversity
                mmr_scores.append((idx, mmr_score))
            best_idx = max(mmr_scores, key=lambda x: x[1])[0]
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

        return [documents[i] for i in selected_indices]

    def rerank(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank documents using the specified method."""
        if self.config.method == "bm25":
            return self.rerank_bm25(query, documents)
        elif self.config.method == "mmr":
            return self.rerank_mmr(query, documents)
        else:
            raise ValueError(f"Unknown reranking method: {self.config.method}")