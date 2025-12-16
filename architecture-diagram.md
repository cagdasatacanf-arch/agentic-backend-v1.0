# System Architecture

```mermaid
flowchart LR
    subgraph Client
        U[User / Frontend]
    end

    subgraph Backend["FastAPI Agent Server"]
        API[FastAPI REST API<br/>/api/v1/query, /api/v1/docs]
        ORCH[Agent Orchestrator<br/>(tools, routing)]
        MEM[Agent Memory & RAG<br/>(Qdrant + embeddings)]
        LLM[LLM API<br/>(OpenAI GPT-4o)]
    end

    subgraph Infra[Runtime]
        DOCS["Docs / PDFs<br/>/data/docs"]
        VDB["Qdrant Vector DB"]
        DOCKER["Docker & Compose"]
    end

    U -->|HTTP JSON| API
    API --> ORCH
    ORCH -->|embed/query| MEM
    MEM --> VDB
    ORCH -->|chat / tools| LLM
    ORCH -->|answer + sources| API
    API -->|JSON response| U

    DOCKER --- API
    DOCKER --- VDB
    DOCS --> MEM
```

## Data Flow

1. User sends question to `/api/v1/query`.
2. Agent orchestrator embeds the question.
3. RAG retrieves relevant docs from Qdrant.
4. LLM receives question + context and generates answer.
5. Answer + sources returned to user.

## Scalability Path

- Single server: current setup (Docker + Compose).
- Multiple replicas: add load balancer + reverse proxy (NGINX, Traefik).
- Background jobs: add Redis + Celery worker for long‑running indexing.
- Monitoring: add Prometheus + Grafana via OpenTelemetry hook.
