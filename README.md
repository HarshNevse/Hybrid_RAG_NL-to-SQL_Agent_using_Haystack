# Hybrid_RAG_NL-to-SQL_Agent_using_Haystack
An SQL agent API with a hybrid RAG pipeline using Python, LLMs, and Haystack. The system processes natural language (NLP) by fusing results from both semantic vector search and traditional keyword-based retrieval, maximizing relevance and accuracy when generating complex SQL queries against relational database schemas.

## Features

- **Hybrid Search**: The pipeline combines keyword-based (BM25) and semantic (SentenceTransformers) search to accurately retrieve the most relevant parts of your database schema based on the user's question.
- **Advanced RAG Pipeline**: It utilises Haystack to orchestrate the retrieval of schema context, ranking of results, and the final generation of the SQL query.
- **LLM Integration**: Uses a Large Language Model (Llama-3.3-70b-versatile via Groq's fast API) to generate accurate SQL queries based on the natural language question and the retrieved schema context.
- **Prompt Engineering**: Carefully crafted prompts for accurate SQL generation with strict output formatting
- **Flask REST API**: Easy-to-use endpoints for schema upload and query generation

## Architecture

The system is built around a sophisticated RAG (Retrieval-Augmented Generation) pipeline that processes database schemas and generates SQL queries through the following components:
![download](https://github.com/user-attachments/assets/bff1a227-51a4-4d14-8860-eb9a2d3abd0a)

## Core Concepts

-   **API-First Design**: The system is exposed via a clear Flask API, allowing for easy integration and testing.
-   **Ranking System**: Re-ranks retrieved documents based on query relevance
-   **Document Store**: In-memory storage for processed schema documents with embeddings
-   **Hybrid Retriever**: Combines BM25 and semantic search for optimal context retrieval
-   **Ranking System**: Re-ranks retrieved documents based on query relevance
-   **Prompt Builder**: Constructs specialized prompts for SQL generation
-   **LLM Generator**: Generates SQL queries using Llama 3 model


## Tech Stack

- **Framework**: Haystack AI
- **Vector Store**: In-Memory Document Store
- **Embeddings**: SentenceTransformers (all-MiniLM-L6-v2)
- **Search**: BM25 + Semantic Search (Hybrid)
- **Ranking**: Transformers Similarity Ranker
- **LLM**: Llama 3.3-70B-versatile (via Groq API)
- **Backend**: Flask 
- **Frontend**: Vanilla HTML/CSS/JavaScript

## Prerequisites

- Python 3.8+
- Groq API Key
- Required Python packages (see installation)

## Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd sql-agent-rag
```

2. **Install dependencies**
```bash
pip install flask flask-cors haystack-ai sentence-transformers openai
```

3. **Set up environment variables**
```bash
export GROQ_API_KEY="your-groq-api-key-here"
```

4. **Run the application**
```bash
python app.py
```

The server will start on `http://localhost:5000`

## How to Use

### API Endpoints

#### 1. Upload Schema
Upload your database schema to initialize the RAG pipeline:

```bash
POST /upload_schema
Content-Type: application/json

{
    "schema": "your_sql_schema_text_here"
}
```

**Response:**
```json
{
    "success": true,
    "message": "Schema uploaded successfully"
}
```

#### 2. Generate SQL Query
Ask natural language questions to generate SQL queries:

```bash
POST /ask
Content-Type: application/json

{
    "question": "Show me all employees with duplicate skills grouped by department"
}
```

**Response:**
```json
{
    "sql_query": "SELECT department, skill, COUNT(*) as employee_count FROM employees GROUP BY department, skill HAVING COUNT(*) > 1;",
    "prompt": "full_prompt_sent_to_llm"
}
```

#### 3. Start a New Chat
  
To clear the current schema from memory and start a new session, send a request to the `/clear` endpoint.

  ```bash
    curl -X POST http://127.0.0.1:5000/clear
  ```

## Demo

https://github.com/user-attachments/assets/df09018e-bb73-4320-a4e1-522c32b2b441


## Query Generation Process

1. **Question Analysis**: Processes natural language input to understand intent
2. **Hybrid Retrieval**: Searches schema using both keyword and semantic matching
3. **Context Ranking**: Ranks retrieved schema chunks by relevance
4. **Prompt Construction**: Builds specialized prompts with retrieved context
5. **SQL Generation**: Uses LLM to generate accurate SQL queries
6. **Output Formatting**: Returns clean SQL without additional explanations

## Configuration

### Environment Variables

- `GROQ_API_KEY`: Your Groq API key for LLM access
- `FLASK_ENV`: Set to 'development' for debug mode

### Model Configuration

The system uses the following models by default:
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **LLM**: `llama-3.3-70b-versatile` (via Groq)

## Important Notes

- The system requires a valid Groq API key for LLM functionality
- Schema files should be in standard SQL format
- Generated queries follow the exact schema structure provided

## Author
Developed by Harsh, AI/ML Enthusiast.
