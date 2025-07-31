from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional

# --- API Request/Response Models ---

class UploadRequest(BaseModel):
    url: HttpUrl = Field(..., description="URL of the PDF document to be indexed.")

class UploadResponse(BaseModel):
    document_id: str
    message: str

class QueryRequest(BaseModel):
    question: str = Field(..., description="The question to ask about the document.")

class QueryResponse(BaseModel):
    question: str
    answer: str
    document_id: str

# --- Metadata Schema (Step 3b in your diagram) ---
# This guides the LLM on what structured data to extract from each chunk.
class DocumentMetadata(BaseModel):
    """Schema for metadata to be extracted from a document chunk."""
    plan_name: Optional[str] = Field(description="The specific insurance plan name, like 'A', 'B', if mentioned.")
    benefit_type: Optional[str] = Field(description="The type of benefit, like 'Room Rent', 'NCD', 'Maternity'.")
    section: Optional[str] = Field(description="The section title, like 'Exclusions', 'Definitions'.")