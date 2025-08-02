import uuid
import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks

from app.models.api_models import UploadRequest, UploadResponse, QueryRequest, QueryResponse
from app.services import qa_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
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

from fastapi import Header
from pydantic import BaseModel
from typing import List
from app.services.phase1.main import main as process_document
from app.services.phase1.semantic_retrieval import process_queries_with_llm

# Request model for /hackrx/run
class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

# Response model for /hackrx/run
class HackRxResponse(BaseModel):
    answers: List[str]

# In-memory document status tracking
DOCUMENT_STATUS = {}

@app.post("/hackrx/run", response_model=HackRxResponse)
async def hackrx_run(request: HackRxRequest, authorization: str = Header(...)):
    """
    Endpoint to process a document and answer questions.
    """
    # Validate bearer token
    expected_token = "c5b88e87ed4a87e4e9425966d9328fd212fb3f4de1e464c58016dc324eeefd75"
    if authorization != f"Bearer {expected_token}":
        raise HTTPException(status_code=401, detail="Invalid or missing authorization token")
    
    document_id = str(uuid.uuid4())
    DOCUMENT_STATUS[document_id] = "processing"
    
    try:
        # Process the document
        logger.info(f"Processing document: {request.documents}")
        process_document(request.documents)
        DOCUMENT_STATUS[document_id] = "completed"
        
        # Process queries with phase2 pipeline
        logger.info(f"Processing {len(request.questions)} questions")
        answers = process_queries_with_llm(request.questions)
        logger.info(f"Generated answers: {answers}")
        
        return HackRxResponse(answers=answers)
    
    except Exception as e:
        DOCUMENT_STATUS[document_id] = f"failed: {str(e)}"
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")