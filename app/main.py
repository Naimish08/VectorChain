import uuid
import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks

from app.models.api_models import UploadRequest, UploadResponse, QueryRequest, QueryResponse
from app.services import qa_service

logging.basicConfig(level=logging.INFO)
app = FastAPI(title="Document Q&A API based on Detailed RAG Pipeline")

@app.post("/upload", response_model=UploadResponse)
async def upload_document(request: UploadRequest, background_tasks: BackgroundTasks):
    """
    Endpoint to upload and index a new document.
    """
    document_id = str(uuid.uuid4())
    background_tasks.add_task(qa_service.process_and_index_document, doc_id=document_id, url=request.url)
    return UploadResponse(document_id=document_id, message="Document ingestion started in the background.")

@app.post("/query/{document_id}", response_model=QueryResponse)
async def query_document(document_id: str, request: QueryRequest):
    """
    Endpoint to ask a question about a previously uploaded document.
    """
    result = qa_service.answer_query(document_id, request)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return QueryResponse(**result)

@app.get("/status/{document_id}")
async def get_document_status(document_id: str):
    """Endpoint to check the processing status of a document."""
    status = qa_service.DOCUMENT_STATUS.get(document_id, "not_found")
    return {"document_id": document_id, "status": status}