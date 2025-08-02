from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict
from tqdm import tqdm
import os

def upsert_to_pinecone(embedded_docs: List[Dict]) -> None:
    """Upsert embedded documents to Pinecone index."""
    # Initialize Pinecone (API key should be in config or env)
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY", "pcsk_2uzvZY_AKLgp4KmLmtmjrbbZDErZQrEsKLKW7Adduyz1T1UPTaiTcPYtMD4nUvVsPxhEyY"))
    
    index_name = "my-index"
    
    # Check if index exists, delete if it does
    if index_name in pc.list_indexes().names():
        print(f"Deleting existing index: {index_name}")
        pc.delete_index(name=index_name)
        print("Index deleted.")
    
    # Create new index
    pc.create_index(
        name=index_name,
        dimension=768,
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    
    index = pc.Index(index_name)
    
    # Batch upsert vectors
    batch_size = 100
    for i in tqdm(range(0, len(embedded_docs), batch_size)):
        batch = embedded_docs[i:i + batch_size]
        index.upsert(vectors=batch)
    
    print(f"Upserted {len(embedded_docs)} vectors to Pinecone index '{index_name}'.")