import logging
from fastapi import FastAPI, HTTPException

from app.models.api_models import RequestPayload, ResponsePayload
from app.services import qa_service

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Dynamic Document Q&A Service",
    description="A generic API that can answer questions about any provided PDF document.",
)

@app.post("/hackrx/run", response_model=ResponsePayload)
async def run_pipeline(payload: RequestPayload):
    """
    API endpoint to receive a document URL and questions.
    It delegates all logic to the qa_service.
    """
    try:
        answers = qa_service.process_document_and_answer_questions(payload)
        return ResponsePayload(answers=answers)
    except Exception as e:
        logging.error(f"An internal error occurred during the pipeline: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")