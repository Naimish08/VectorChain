from pydantic import BaseModel, Field, HttpUrl
from typing import List

# Defines the structure of the incoming JSON request.
class RequestPayload(BaseModel):
    documents: HttpUrl = Field(..., description="URL to the PDF document to be processed.")
    questions: List[str] = Field(..., description="A list of questions to answer based on the document.")

# Defines the structure of a single answer object.
class QuestionAnswer(BaseModel):
    question: str
    answer: str

# Defines the overall structure of the JSON response.
class ResponsePayload(BaseModel):
    answers: List[QuestionAnswer]

# This is the GENERIC metadata schema, adaptable to any document type.
class GenericDocumentMetadata(BaseModel):
    """A generic metadata schema for any document chunk."""
    main_topic: str = Field(description="The primary, high-level topic of this text chunk (e.g., 'Contractual Obligations', 'Financial Results', 'Methodology').")
    summary: str = Field(description="A concise one-sentence summary of the chunk's content.")
    key_entities: List[str] = Field(description="A list of important named entities (like people, products, or technical terms) mentioned in the text.")