from typing import List, Dict, Any, Optional
import os
from pymongo import MongoClient
from pydantic import BaseModel
import time
from urllib.parse import quote_plus
from dotenv import load_dotenv
from datetime import datetime

# Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

load_dotenv()

class VectorStoreConfig(BaseModel):
    """Configuration for vector store"""
    store_type: str = os.getenv("VECTOR_STORE_TYPE", "mongodb")  # Only mongodb supported here
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    index_type: str = os.getenv("INDEX_TYPE", "flat")
    dimension: Optional[int] = None
    top_k: int = int(os.getenv("TOP_K", "5"))
    similarity_threshold: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
    batch_size: int = int(os.getenv("BATCH_SIZE", "100"))

    # MongoDB specific
    username: Optional[str] = quote_plus(os.getenv("MONGODB_USER", ""))
    password: Optional[str] = quote_plus(os.getenv("MONGODB_PASSWORD", ""))
    mongodb_uri: str = f"mongodb+srv://{username}:{password}@agenticai-cluster.hvvgasu.mongodb.net/?retryWrites=true&w=majority&tls=true"
    mongodb_db: str = os.getenv("MONGODB_DBNAME", "pdf_analysis")

class MongoDBVectorStore:
    """MongoDB-based vector store implementation for Atlas Vector Search"""

    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.client = MongoClient(config.mongodb_uri, tls=True, tlsAllowInvalidCertificates=False)
        self.db = self.client[config.mongodb_db]
        self.collection = self.db[f"vectors_{config.embedding_model.replace('/', '_')}_{config.index_type}"]
        self._create_indices()

    def _create_indices(self) -> None:
        """Create necessary indices in MongoDB (metadata only, NOT for embeddings)"""
        self.collection.create_index([("metadata.filename", 1)])
        self.collection.create_index([("metadata.page", 1)])

    def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]]) -> None:
        """Add texts and embeddings to MongoDB"""
        embeddings = self._get_embeddings(texts)
        documents = [
            {
                "text": text,
                "embedding": embedding,
                "metadata": metadata
            }
            for text, embedding, metadata in zip(texts, embeddings, metadatas)
        ]
        self.collection.insert_many(documents)

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """This method will be patched by the parent VectorStore.
            Generate embeddings for a list of texts using the configured embedding model.
        """
        return self.embeddings.embed_documents(texts)

    def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar texts in MongoDB Atlas using $vectorSearch"""
        query_embedding = self._get_embeddings([query])[0]
        pipeline = [
            {
                "$vectorSearch": {
                    "index": f"vector_index_{self.config.index_type}",  # Must match your Atlas vector index name
                    "queryVector": query_embedding,
                    "path": "embedding",
                    "numCandidates": k * 5,
                    "limit": k
                }
            },
            {
                "$project": {
                    "text": 1,
                    "metadata": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        results = list(self.collection.aggregate(pipeline))
        return [
            {
                "text": doc["text"],
                "metadata": doc.get("metadata", {}),
                "score": doc.get("score", 0.0)
            }
            for doc in results
        ]

    def clear(self) -> None:
        """Clear all vectors from MongoDB"""
        self.collection.delete_many({})

# Placeholder classes for other vector databases
class OpenSearchVectorStore:
    """Placeholder for OpenSearch vector store implementation."""
    def __init__(self, config: VectorStoreConfig):
        raise NotImplementedError("OpenSearchVectorStore is not implemented in this setup.")

class FAISSVectorStore:
    """Placeholder for FAISS vector store implementation."""
    def __init__(self, config: VectorStoreConfig):
        raise NotImplementedError("FAISSVectorStore is not implemented in this setup.")

class VectorStore:
    """Main vector store class that uses the appropriate implementation based on configuration"""
    def __init__(self, config: Optional[VectorStoreConfig] = None):
        self.config = config or VectorStoreConfig()
        self._setup_embeddings()
        self._setup_vector_store()

    def _setup_embeddings(self) -> None:
        if "openai" in self.config.embedding_model.lower():
            self.embeddings = OpenAIEmbeddings(
                model=self.config.embedding_model,
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
        else:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.config.embedding_model,
                model_kwargs={"device": "cuda" if os.getenv("USE_GPU", "false").lower() == "true" else "cpu"}
            )
        if not self.config.dimension:
            self.config.dimension = self._get_model_dimension()

    def _setup_vector_store(self) -> None:
        if self.config.store_type == "mongodb":
            self.store = MongoDBVectorStore(self.config)
        elif self.config.store_type == "opensearch":
            self.store = OpenSearchVectorStore(self.config)
        elif self.config.store_type == "faiss":
            self.store = FAISSVectorStore(self.config)
        else:
            raise ValueError(f"Unsupported vector store type: {self.config.store_type}")

        # Patch embedding method into store
        self.store._get_embeddings = self._get_embeddings

    def _get_model_dimension(self) -> int:
        test_embedding = self.embeddings.embed_query("test")
        return len(test_embedding)

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self.embeddings.embed_documents(texts)

    def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]]) -> None:
        self.store.add_texts(texts, metadatas)

    def similarity_search(self, query: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
        k = k or self.config.top_k
        return self.store.similarity_search(query, k)

    def clear(self) -> None:
        self.store.clear()

    def search(self, query: str, k: int = 5, store: str = "mongodb") -> Dict[str, List[Dict[str, Any]]]:
        results = {}
        stores_to_search = [store]
        for store_type in stores_to_search:
            temp_config = VectorStoreConfig(store_type=store_type)
            temp_store = None
            if store_type == "mongodb":
                temp_store = MongoDBVectorStore(temp_config)
            elif store_type == "opensearch":
                temp_store = OpenSearchVectorStore(temp_config)
            elif store_type == "faiss":
                temp_store = FAISSVectorStore(temp_config)
            if temp_store:
                temp_store._get_embeddings = self._get_embeddings
                start_time = time.time()
                store_results = temp_store.similarity_search(query, k)
                search_time = time.time() - start_time
                results[store_type] = {
                    "results": store_results,
                    "time": search_time
                }
        return results

    def calculate_accuracy(self, query: str, ground_truth: List[str], k: int = 5, store: str = "mongodb") -> Dict[str, Dict[str, float]]:
        results = self.search(query, k=k, store=store)
        accuracy_metrics = {}
        for store_type, result in results.items():
            retrieved_texts = [hit['text'] for hit in result['results']]
            relevant_retrieved = sum(1 for text in retrieved_texts if text in ground_truth)
            precision = relevant_retrieved / len(retrieved_texts) if retrieved_texts else 0.0
            recall = relevant_retrieved / len(ground_truth) if ground_truth else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            accuracy_metrics[store_type] = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "search_time": result['time']
            }
        return accuracy_metrics