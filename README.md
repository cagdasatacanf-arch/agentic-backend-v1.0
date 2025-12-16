# Agentic AI Backend

A production‑lean FastAPI backend for agentic AI systems with RAG, OpenAI integration, and Qdrant vector storage.

## Features

- FastAPI HTTP API for agent queries and document ingestion.
- OpenAI GPT‑4o for reasoning and generation.
- Qdrant vector database for document retrieval and RAG.
- Docker + Docker Compose for reproducible deployment.
- GitHub Actions CI/CD for automated testing and image builds.
- Clean layered architecture ready for tool expansion.
- Background task support for document indexing.

## Quick Start

1. Clone the repo:
   ```bash
   git clone https://github.com/<you>/agentic-backend.git
   cd agentic-backend
   ```

2. Create and configure env:
   ```bash
   cp .env.example .env
   # Edit .env with your OPENAI_API_KEY and other values
   ```

3. Start the stack:
   ```bash
   docker compose up --build
   ```

4. Use the API:
   - **Health check**: `GET http://localhost:8000/api/v1/health`
   - **Query agent**: `POST http://localhost:8000/api/v1/query`
   - **Ingest docs**: `POST http://localhost:8000/api/v1/docs`
   - **API docs**: `http://localhost:8000/docs`

## Project Structure

```
app/
  config.py           # Pydantic settings
  main.py            # FastAPI app, middleware, exceptions
  rag.py             # RAG + vector DB logic
  api/
    routes_query.py  # /api/v1/query endpoint
    routes_docs.py   # /api/v1/docs endpoint
  services/
    agent_service.py # Agent orchestration & tools
tests/
  test_health.py
  test_query.py
  test_docs.py
docker-compose.yml
Dockerfile
requirements.txt
.env.example
```

## API Contract

### POST /api/v1/query

Query the agent with a question.

**Request:**
```json
{
  "question": "What is this system?",
  "top_k": 5,
  "session_id": null
}
```

**Response:**
```json
{
  "answer": "This is an agentic AI backend...",
  "sources": [
    {
      "id": "doc-123",
      "text": "Relevant passage...",
      "score": 0.87,
      "metadata": {}
    }
  ],
  "usage": {
    "prompt_tokens": 123,
    "completion_tokens": 456,
    "total_tokens": 579
  }
}
```

### POST /api/v1/docs

Ingest a document for RAG.

**Request:**
```json
{
  "text": "Full document content",
  "metadata": {"filename": "doc.pdf"}
}
```

**Response:**
```json
{
  "id": "pending",
  "status": "queued"
}
```

### GET /api/v1/health

Health check.

**Response:**
```json
{
  "status": "ok",
  "version": "1.0.0"
}
```

## Error Codes

All errors follow a consistent schema:

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable message",
    "details": {},
    "correlation_id": "uuid"
  }
}
```

Common codes: `BAD_REQUEST`, `VALIDATION_ERROR`, `UNAUTHENTICATED`, `INTERNAL_ERROR`, `UPSTREAM_ERROR`.

See `error-codes.md` for full list.

## Deployment

### Local development

```bash
docker compose up --build
```

### Production

1. Set `ENVIRONMENT=prod` in `.env`.
2. Set a strong `INTERNAL_API_KEY`.
3. Deploy the repo to a VPS or container platform (Render, Railway, AWS, etc.):
   ```bash
   git clone ...
   cp .env.example .env  # set prod values
   docker compose up -d --build
   ```

## Architecture

See `architecture-diagram.md` for a Mermaid diagram of the system.

## Contributing

Run tests locally:

```bash
pip install -r requirements.txt
pytest
```

Format code (optional):

```bash
black app tests
ruff check app tests
```

## License

MIT
