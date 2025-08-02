from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone
import os
from typing import List, Tuple
from app.services.phase2.param_tuning import process_top_chunks
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def query_pinecone(queries: List[str]) -> List[Tuple[str, List[dict]]]:
    """Query Pinecone index with natural language queries, returning top 2 chunks."""
    # Initialize embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    # Initialize Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index("my-index")
    
    # Generate embeddings for queries
    query_vectors = embeddings.embed_documents(queries)
    
    # Collect results
    results = []
    
    # Query for each vector
    for query_text, query_vector in zip(queries, query_vectors):
        logger.info(f"Processing query: {query_text}")
        response = index.query(
            vector=query_vector,
            top_k=2,  # Get top 2 matches
            include_metadata=True
        )
        
        matches = response["matches"] if response["matches"] else []
        logger.info(f"Matches found for '{query_text}': {len(matches)}")
        for i, match in enumerate(matches):
            logger.info(f"Match {i+1}: ID={match['id']}, Score={match['score']:.3f}, Text={match['metadata']['text'][:100]}...")
        results.append((query_text, matches))
    
    return results

def process_queries_with_llm(queries: List[str]) -> List[str]:
    """Process queries by retrieving top 2 chunks and sending to phase2 for answer generation."""
    results = query_pinecone(queries)
    answers = []
    
    for query_text, matches in results:
        if matches:
            # Extract top 2 chunks
            chunks = [
                {
                    "id": match["id"],
                    "score": match["score"],
                    "text": match["metadata"]["text"],
                    "metadata": {
                        "type": match["metadata"].get("type"),
                        "page": match["metadata"].get("page") or match["metadata"].get("pages")
                    }
                }
                for match in matches
            ]
            # Send to phase2 for processing
            answer = process_top_chunks(query_text, chunks)
            logger.info(f"Raw LLM answer for '{query_text}': {answer}")
            answers.append(answer)
        else:
            logger.info(f"No matches found for query: {query_text}")
            answers.append("No answer found for this question.")
    
    return answers

def print_query_results(results: List[Tuple[str, List[dict]]]) -> None:
    """Print query results in a formatted manner."""
    for query_text, matches in results:
        print(f"\n🔍 Query: {query_text}")
        
        if matches:
            for match in matches:
                meta = match["metadata"]
                text = meta["text"]
                
                print(f"— ID: {match['id']} (score: {match['score']:.3f})")
                print(f"  Type: {meta.get('type')} | Page: {meta.get('page') or meta.get('pages')}")
                print(f"  Full Chunk:\n{text}\n")
        else:
            print("  No matches found.")