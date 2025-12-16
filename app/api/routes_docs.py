from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from app.services.agent_service import agent_service
from app.main import api_key_auth, limiter
from fastapi.requests import Request

router = APIRouter(prefix="/docs", tags=["Docs"])

@router.post("/upload")
@limiter.limit("5/minute")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    _ = Depends(api_key_auth),
):
    """
    Upload a document (PDF, Markdown, Text) to the Knowledge Base.
    """
    try:
        content = await file.read()
        res = await agent_service.ingest_file(
            filename=file.filename,
            content=content,
            content_type=file.content_type or "text/plain"
        )
        
        if "Error" in res:
             raise HTTPException(status_code=400, detail=res)
             
        return {"status": "success", "message": res}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
