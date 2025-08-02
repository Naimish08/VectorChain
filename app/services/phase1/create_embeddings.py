from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from typing import List, Dict
import os

def generate_embeddings(documents: List[Document]) -> List[Dict]:
    """Generate embeddings for documents using GoogleGenerativeAIEmbeddings."""
    # Initialize embeddings (API key should be in config or env)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=os.getenv("GOOGLE_API_KEY", "AIzaSyCWJJ_Wg9c5CVJO37VDjG5re9YxN8ux4gU")
    )
    
    texts = [doc.page_content for doc in documents]
    
    # Generate embeddings
    vectors = embeddings.embed_documents(texts)
    
    print(f"Generated {len(vectors)} embeddings.")
    print(f"Embedding dimension: {len(vectors[0])}")
    
    # Prepare embedded documents for Pinecone
    embedded_docs = [
        {
            "id": doc.metadata.get("chunk_id", f"doc_{i}"),
            "values": vector,
            "metadata": doc.metadata | {"text": doc.page_content}
        }
        for i, (doc, vector) in enumerate(zip(documents, vectors))
    ]
    
    return embedded_docs