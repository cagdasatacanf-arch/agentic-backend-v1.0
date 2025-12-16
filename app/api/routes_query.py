from typing import List, Dict

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.main import api_key_auth

router = APIRouter(prefix="/api/v1", tags=["query"])
limiter = Limiter(key_func=get_remote_address)


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    stream: bool = False
    session_id: str | None = None


class Source(BaseModel):
    id: str | None = None
    text: str
    score: float
    metadata: Dict | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    usage: Dict | None = None


@router.post(
    "/query",
    response_model=QueryResponse,
    dependencies=[Depends(api_key_auth)],
)
@limiter.limit("10/minute")
async def query_agent(request: Request, body: QueryRequest):
    from app.services.agent_service import agent_service
    
    answer, docs, usage = await agent_service.answer(
        question=body.question,
        top_k=body.top_k,
        session_id=body.session_id,
    )
    return QueryResponse(
        answer=answer,
        sources=[Source(**d) for d in docs],
        usage=usage,
    )
