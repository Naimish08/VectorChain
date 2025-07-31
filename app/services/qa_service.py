import os
import requests
import logging
from typing import List

from app.config import settings
from app.models.api_models import RequestPayload, GenericDocumentMetadata

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_extraction_chain, RetrievalQA
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize models once when the server starts for efficiency.
os.environ["GOOGLE_API_KEY"] = settings.google_api_key
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0, convert_system_message_to_human=True)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def process_document_and_answer_questions(payload: RequestPayload) -> List[dict]:
    # 1. Download and Chunk PDF
    logger.info(f"Loading document from {payload.documents}")
    temp_pdf_path = "temp_document.pdf"
    try:
        response = requests.get(payload.documents)
        response.raise_for_status()
        with open(temp_pdf_path, "wb") as f:
            f.write(response.content)
        
        loader = PyPDFLoader(temp_pdf_path)
        chunks = loader.load_and_split(RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200))
        logger.info(f"PDF split into {len(chunks)} chunks.")
    finally:
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)

    # 2. Extract Generic Metadata for each chunk
    logger.info("Extracting generic metadata from chunks...")
    extraction_chain = create_extraction_chain(schema=GenericDocumentMetadata.model_json_schema(), llm=llm)
    docs_with_metadata = []
    for chunk in chunks:
        try:
            extracted_data = extraction_chain.invoke(chunk.page_content)['text']
            if extracted_data:
                chunk.metadata.update(extracted_data[0])
            docs_with_metadata.append(chunk)
        except Exception as e:
            logger.warning(f"Metadata extraction failed for a chunk: {e}")
            docs_with_metadata.append(chunk)
    
    # 3. Create In-Memory Vector Store (FAISS) for this request
    logger.info("Creating in-memory FAISS vector store.")
    vectorstore = FAISS.from_documents(docs_with_metadata, embeddings)

    # 4. Configure the Self-Query Retriever and QA Chain
    metadata_field_info = [
        AttributeInfo(name="main_topic", description="The high-level topic of a document segment.", type="string"),
        AttributeInfo(name="key_entities", description="Specific entities mentioned in the text.", type="list[string]"),
    ]
    retriever = SelfQueryRetriever.from_llm(
        llm=llm,
        vectorstore=vectorstore,
        document_contents="A chunk of text from a document.",
        metadata_field_info=metadata_field_info,
    )
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    # 5. Process all questions and collect answers
    logger.info("Processing all questions against the document.")
    answers = []
    for question in payload.questions:
        result = qa_chain.invoke({"query": question})
        answer = result.get('result', 'Could not find a specific answer in the document.')
        answers.append({"question": question, "answer": answer})
        
    return answers