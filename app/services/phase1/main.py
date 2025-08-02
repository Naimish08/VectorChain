import requests
import tempfile
import os
from langchain_core.documents import Document
from typing import List
# They have been moved to the top of the file.
from app.services.phase1.document_loader import load_pdf_from_url, extract_layout_tables
from app.services.phase1.text_chunker import semantic_chunking
from app.services.phase1.create_embeddings import generate_embeddings
from app.services.phase1.into_pinecone import upsert_to_pinecone

def main(url: str) -> List[Document]:
    """Main function to load documents and extract tables from a PDF URL."""
    
    # Download PDF to temporary file
    response = requests.get(url)
    response.raise_for_status()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
        temp_file.write(response.content)
        temp_file_path = temp_file.name
    
    # Load documents
    documents = load_pdf_from_url(url)
    
    # Extract layout-based tables
    layout_tables = extract_layout_tables(temp_file_path)
    
    # Perform semantic chunking
    all_documents = semantic_chunking(documents, layout_tables)
    
    # Generate embeddings
    embedded_docs = generate_embeddings(all_documents)
    
    # Upsert to Pinecone
    upsert_to_pinecone(embedded_docs)
    
    os.unlink(temp_file_path)  # Clean up temporary file
    return all_documents

if __name__ == "__main__":
    # Example usage for testing
    sample_url = "https://example.com/sample.pdf"
    main(sample_url)
