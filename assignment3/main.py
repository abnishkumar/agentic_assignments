import os
from typing import List, Dict, Any, Optional
from pdf_processor import PDFProcessor, PDFChunk
from vector_store import VectorStore, VectorStoreConfig
from reranker import Reranker, RerankerConfig
from llm_processor import LLMProcessor, LLMConfig
from docx_renderer import DocxRenderer
from dotenv import load_dotenv
import time
import json
from datetime import datetime
import numpy as np

load_dotenv()

os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
## Langsmith Tracking And Tracing
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"]="true"

class PDFAnalysisSystem:
    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.stores = self._setup_vector_stores()
        self.reranker = Reranker(RerankerConfig(method="mmr"))
        self.llm_processor = LLMProcessor(LLMConfig())
        self.docx_renderer = DocxRenderer()
        
        # Create necessary directories
        os.makedirs("pdfs", exist_ok=True)
        os.makedirs("output", exist_ok=True)
        
        # Load processed files tracking
        self.processed_files = self._load_processed_files()

    def _setup_vector_stores(self) -> Dict[str, VectorStore]:
        """Setup vector stores with different index types and models."""
        stores = {}

        # Get model configuration
        model_type = os.getenv("EMBEDDING_MODEL_TYPE", "local")
        local_model = os.getenv("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        openai_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")

        # Create stores for each index type
        index_types = ["flat", "hnsw", "ivf"]
        for index_type in index_types:
            # Local model store
            local_config = VectorStoreConfig(
                embedding_model=local_model,
                index_type=index_type
            )
            stores[f"{index_type}_local"] = VectorStore(local_config)

            # OpenAI model store if configured
            if model_type == "openai" and os.getenv("OPENAI_API_KEY"):
                openai_config = VectorStoreConfig(
                    embedding_model=openai_model,
                    index_type=index_type
                )
                stores[f"{index_type}_openai"] = VectorStore(openai_config)

        return stores

    def _load_processed_files(self) -> Dict[str, str]:
        """Load tracking of processed files."""
        tracking_file = "output/processed_files.json"
        if os.path.exists(tracking_file):
            with open(tracking_file, "r") as f:
                return json.load(f)
        return {}

    def _save_processed_files(self):
        """Save tracking of processed files."""
        with open("output/processed_files.json", "w") as f:
            json.dump(self.processed_files, f, indent=2)

    def process_directory(self, directory: str = "pdfs") -> List[str]:
        """Process all unprocessed PDFs in a directory."""
        processed_files = []
        
        for filename in os.listdir(directory):
            if filename.endswith('.pdf'):
                file_path = os.path.join(directory, filename)
                if file_path not in self.processed_files:
                    try:
                        self.process_single_file(file_path)
                        processed_files.append(filename)
                    except Exception as e:
                        print(f"Error processing {filename}: {str(e)}")
        
        return processed_files

    def process_single_file(self, file_path: str) -> bool:
        """Process a single PDF file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        print(f"Processing {file_path}...")
        chunks = self.pdf_processor.process_pdf(file_path)
        
        if not chunks:
            print(f"No content extracted from {file_path}")
            return False
        
        # Prepare texts and metadatas
        texts = [chunk.text for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        # Add to vector stores
        for store in self.stores.values():
            store.add_texts(texts, metadatas)
        
        # Update processed files tracking
        self.processed_files[file_path] = datetime.now().isoformat()
        self._save_processed_files()
        
        print(f"Successfully processed {file_path}")
        return True

    def evaluate_retrieval_performance(self, query: str, k: int = 5) -> Dict[str, float]:
        """Evaluate retrieval performance of different stores."""
        results = {}
        
        for store_name, store in self.stores.items():
            start_time = time.time()
            search_results = store.search(query, k=k)
            end_time = time.time()
            
            results[store_name] = {
                "time": end_time - start_time,
                "results": search_results
            }
        
        return results

    def query_documents(self, query: str, k: int = 10, save_output: bool = True) -> Dict[str, Any]:
        """Query documents and generate response."""
        # Evaluate retrieval performance
        performance_results = self.evaluate_retrieval_performance(query, k=k)
        
        # Get best performing store
        best_store = min(performance_results.items(), key=lambda x: x[1]["time"])[0]
        print(f"Best performing store: {best_store}")
        
        # Retrieve and rerank documents
        search_results = self.stores[best_store].search(query, k=k)
        # If search_results is a dict with "results" key:
        if isinstance(search_results, dict) and "results" in search_results:
            docs = search_results["results"]
        else:
            docs = search_results
        # Ensure docs is a list of dicts with 'text' key
        if isinstance(docs, list):
            if len(docs) == 0:
                docs = []
            elif isinstance(docs[0], str):
                docs = [{"text": d} for d in docs]
            elif isinstance(docs[0], dict) and "text" in docs[0]:
                pass  # already correct format
            else:
                # fallback: convert everything to string
                docs = [{"text": str(d)} for d in docs]
        else:
            docs = []
        reranked_docs = self.reranker.rerank(query, docs)
        
        # Generate response
        response = self.llm_processor.generate_response(query, reranked_docs)
        
        # Prepare metadata
        metadata = {
            "Query": query,
            "Timestamp": datetime.now().isoformat(),
            "Best Store": best_store,
            "Retrieval Time": performance_results[best_store]["time"],
            "Reranking Method": self.reranker.config.method,
            "Embedding Model": best_store.split('_')[1]  # local or openai
        }
        
        # Save output if requested
        if save_output:
            output_filename = f"output/query_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
            doc = self.docx_renderer.render(query, reranked_docs, response, metadata)
            self.docx_renderer.save(output_filename)
            print(f"Results saved to {output_filename}")
        
        return {
            "query": query,
            "response": response,
            "context": reranked_docs,
            "metadata": metadata,
            "performance": performance_results
        }

def main():
    # Initialize the system
    system = PDFAnalysisSystem()
    
    while True:
        print("\nPDF Analysis System")
        print("1. Process all unprocessed PDFs in directory")
        print("2. Process a single PDF file")
        print("3. Query documents")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == "1":
            processed = system.process_directory()
            print(f"Processed {len(processed)} new files: {', '.join(processed)}")
        
        elif choice == "2":
            file_path = input("Enter the path to the PDF file: ")
            try:
                if system.process_single_file(file_path):
                    print("File processed successfully")
            except Exception as e:
                print(f"Error: {str(e)}")
        
        elif choice == "3":
            query = input("Enter your query: ")
            save = input("Save results to DOCx? (y/n): ").lower() == 'y'
            results = system.query_documents(query, save_output=save)
            print("\nResponse:", results["response"])
        
        elif choice == "4":
            print("Exiting...")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()