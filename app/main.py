import logging
import os
from logging.config import dictConfig
from typing import Dict

from fastapi import FastAPI, Request, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette import status
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from app.config import settings

LOGGING_CONFIG = {
    "version": 1,
    "formatters": {"default": {"format": "%(asctime)s - %(levelname)s - %(message)s"}},
    "handlers": {"console": {"class": "logging.StreamHandler", "formatter": "default"}},
    "loggers": {"app": {"handlers": ["console"], "level": "INFO"}},
}
dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("app")

# --- OpenTelemetry Setup ---
resource = Resource(attributes={
    "service.name": "agentic-backend",
    "service.version": "1.0.0"
})

trace.set_tracer_provider(TracerProvider(resource=resource))
tracer = trace.get_tracer(__name__)

# Export to Jaeger (via OTLP gRPC endpoint)
otlp_exporter = OTLPSpanExporter(endpoint="http://jaeger:4317", insecure=True)
span_processor = BatchSpanProcessor(otlp_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="Agentic AI Prototype", openapi_url="/api/v1/openapi.json")
# Instrument FastAPI
FastAPIInstrumentor.instrument_app(app)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

origins = ["http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = settings.internal_api_key or ""
OTEL_EXPORTER_OTLP_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
ENVIRONMENT = os.getenv("ENVIRONMENT", "local")


class ErrorBody(BaseModel):
    code: str
    message: str
    details: Dict | None = None
    correlation_id: str


def error_response(
    status_code: int,
    code: str,
    message: str,
    details: Dict | None = None,
):
    import uuid

    correlation_id = str(uuid.uuid4())
    logger.error(f"{code} {message} details={details} cid={correlation_id}")
    raise HTTPException(
        status_code=status_code,
        detail=ErrorBody(
            code=code, message=message, details=details, correlation_id=correlation_id
        ).model_dump(),
    )


async def api_key_auth(x_api_key: str = Header(default="")):
    if API_KEY and x_api_key != API_KEY:
        error_response(
            status.HTTP_401_UNAUTHORIZED,
            "UNAUTHENTICATED",
            "Authentication is required for this endpoint.",
        )


@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"{request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"{request.method} {request.url.path} -> {response.status_code}")
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Request body validation failed.",
                "details": exc.errors(),
                "correlation_id": "auto",
            }
        },
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc: StarletteHTTPException):
    if isinstance(exc.detail, dict) and "code" in exc.detail:
        return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": "BAD_REQUEST",
                "message": str(exc.detail),
                "details": {},
                "correlation_id": "auto",
            }
        },
    )


@app.get("/api/v1/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/api/v1/metadata")
def metadata():
    return {
        "models": {
            "chat": settings.openai_chat_model,
            "embedding": settings.openai_embedding_model,
        },
        "vector_store": "qdrant",
        "max_context_tokens": 16000,
    }


# Hide docs on prod if desired
if ENVIRONMENT == "prod":
    app.docs_url = None
    app.redoc_url = None
    app.openapi_url = None





# Routers
from app.api.routes_query import router as query_router  # noqa: E402
from app.api.routes_docs import router as docs_router  # noqa: E402

app.include_router(query_router)
app.include_router(docs_router)
