# PDF Processing and Vector Search System

This project implements a comprehensive system for processing PDFs, performing vector search, and generating outputs using LLMs.

## Features

- PDF text, image, and table extraction
- Semantic chunking and embedding
- Vector database storage (MongoDB and OpenSearch)
- Multiple index mechanisms (Flat, HNSW, IVF)
- Retrieval pipeline with performance comparison
- Reranking using BM25 and MMR
- LLM-based output generation
- DOCx rendering

## Prerequisites

- Python 3.13
- MongoDB (remote instance)
- OpenAI API key

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root directory with the following configuration:

```env
# OpenAI API Configuration
OPENAI_API_KEY="sk-4d1b2f3c-5e6f-7g"
OPENAI_EMBEDDING_MODEL=text-embedding-ada-002

# HUGGINGFACE Configuration
HUGGINGFACE_HUB_TOKEN=WBAoGgnLAu
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# GROQ Configuration
GROQ_API_KEY=9kLWGdyb3FYmXg4p2DMJIJOUleDvIcyomjX

# LANGCHAIN Configuration
LANGCHAIN_API_KEY="ae3e3fedf43c294ef4030d99ccc9d_fad5694852"
LANGCHAIN_PROJECT="Agentic2.0"
LANGCHAIN_TRACING_V2="true"

# MongoDB Configuration
MONGODB_USER=agenticai
MONGODB_PASSWORD=password
MONGODB_DBNAME=agenticai
VECTOR_STORE_TYPE=mongodb
# OpenSearch Configuration
OPENSEARCH_HOST=localhost
OPENSEARCH_PORT=9200
OPENSEARCH_USER=admin
OPENSEARCH_PASSWORD=admin

# System Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MODEL_NAME=gpt-3.5-turbo
TEMPERATURE=0.7
MAX_TOKENS=1000
```

### Environment Variables Explained

#### OpenAI API Configuration
- `OPENAI_API_KEY`: Your OpenAI API key for accessing GPT models
  - Required for LLM-based response generation
  - Get your API key from [OpenAI Platform](https://platform.openai.com)

#### MongoDB Configuration
- `MONGODB_URI`: Connection string for MongoDB
  - MongoDB, use format: `mongodb://username:password@host:port/`

## Usage

1. Place your PDF files in the `pdfs` directory

2. Run the main script:
```bash
python main.py
```

3. Use the interactive menu:
   - Option 1: Process all unprocessed PDFs in directory
   - Option 2: Process a single PDF file
   - Option 3: Query documents
   - Option 4: Exit

## Project Structure

- `main.py`: Main execution script with interactive menu
- `pdf_processor.py`: PDF processing and chunking
- `vector_store.py`: Vector database operations
- `retriever.py`: Retrieval pipeline implementation
- `reranker.py`: Reranking implementation
- `llm_processor.py`: LLM integration and output generation
- `docx_renderer.py`: DOCx rendering functionality
- `.env`: Environment configuration file
- `requirements.txt`: Python dependencies
- `pdfs/`: Directory for input PDF files
- `output/`: Directory for generated documents and results

## Output

The system generates two types of output:

1. Query Results:
   - DOCx files with query responses
   - Located in `output/` directory
   - Named with timestamp: `query_result_YYYYMMDD_HHMMSS.docx`

2. System Files:
   - `processed_files.json`: Tracks processed PDFs


## Security Notes

1. Never commit the `.env` file to version control
2. Keep your API keys and credentials secure
3. Use appropriate access controls for MongoDB and OpenSearch
4. Regularly rotate passwords and API keys

## Troubleshooting

1. MongoDB Connection Issues:
   - Check connection string in `.env`
   - Ensure network access for remote instance

  