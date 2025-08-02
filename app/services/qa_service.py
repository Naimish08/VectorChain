import os
import uuid
import logging
from typing import Dict

import requests
from pinecone import Pinecone as PineconeClient

from app.config import settings
from app.models.api_models import QueryRequest, DocumentMetadata

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Global Components Initialization ---
os.environ["GOOGLE_API_KEY"] = settings.google_api_key
os.environ["PINECONE_API_KEY"] = settings.pinecone_api_key

# --- FIX #2: Use a more specific and modern model name ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

pinecone_client = PineconeClient()
index_name = settings.pinecone_index_name

if index_name not in pinecone_client.list_indexes().names():
    logger.info(f"Creating Pinecone index: {index_name}")
    pinecone_client.create_index(name=index_name, dimension=768, metric='cosine')
else:
    logger.info(f"Pinecone index '{index_name}' already exists.")

vectorstore = Pinecone.from_existing_index(index_name, embeddings)
DOCUMENT_STATUS: Dict[str, str] = {}


def process_and_index_document(doc_id: str, url: str):
    """
    Implements the entire Ingestion Pipeline.
    """
    logger.info(f"[{doc_id}] Starting ingestion pipeline for URL: {url}")
    DOCUMENT_STATUS[doc_id] = "processing"
    
    temp_pdf_path = f"temp_{doc_id}.pdf"
    try:
        # Step 1: Document Loader
        response = requests.get(url)
        response.raise_for_status()
        with open(temp_pdf_path, "wb") as f:
            f.write(response.content)
        
        loader = PyPDFLoader(temp_pdf_path)
        # Step 2: Text Chunker
        chunks = loader.load_and_split(RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200))
        logger.info(f"[{doc_id}] PDF split into {len(chunks)} chunks.")

        # --- FIX #1: Use the modern .with_structured_output() method for metadata extraction ---
        # This is the new, recommended way to extract structured data.
        extraction_prompt = ChatPromptTemplate.from_template(
            "Extract the relevant metadata from the following text chunk:\n\n{chunk_text}"
        )
        extraction_chain = extraction_prompt | llm.with_structured_output(DocumentMetadata)

        docs_with_metadata = []
        for chunk in chunks:
            try:
                chunk.metadata['document_id'] = doc_id
                extracted_data = extraction_chain.invoke({"chunk_text": chunk.page_content})
                # Convert the Pydantic model to a dict to update metadata
                chunk.metadata.update(extracted_data.dict())
                docs_with_metadata.append(chunk)
            except Exception as e:
                logger.warning(f"[{doc_id}] Metadata extraction failed for a chunk: {e}. Storing chunk without extra metadata.")
                docs_with_metadata.append(chunk)
        
        # Add all processed chunks to Pinecone in a single batch
        vectorstore.add_documents(docs_with_metadata, namespace=doc_id)
        
        DOCUMENT_STATUS[doc_id] = "ready"
        logger.info(f"[{doc_id}] Successfully indexed in Pinecone under namespace '{doc_id}'.")

    except Exception as e:
        logger.error(f"[{doc_id}] Error during ingestion pipeline: {e}")
        DOCUMENT_STATUS[doc_id] = f"failed: {e}"
    finally:
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)


def answer_query(doc_id: str, request: QueryRequest) -> Dict:
    """
    Implements the Query/Response Pipeline.
    """
    status = DOCUMENT_STATUS.get(doc_id)
    if status is None:
        return {"error": "Document not found. Please upload it first."}
    if status != "ready":
        return {"error": f"Document is not ready for querying. Current status: {status}"}

    logger.info(f"Querying document '{doc_id}' with question: '{request.question}'")

    # The retriever will find relevant documents within the specified namespace
    retriever = vectorstore.as_retriever(
        search_kwargs={'namespace': doc_id}
    )
    
    # This chain combines the retrieved documents and the question to generate the final answer.
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )

    result = qa_chain.invoke({"query": request.question})
    answer = result.get('result', 'Could not find a specific answer in the document.')
    
    return {"question": request.question, "answer": answer, "document_id": doc_id}