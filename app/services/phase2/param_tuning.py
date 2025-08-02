from typing import List, Dict
import google.generativeai as genai
import os
from time import sleep
import google.api_core.exceptions

def process_top_chunks(query: str, chunks: List[Dict]) -> str:
    """Process top chunks with an LLM to generate a concise yet comprehensive answer."""
    # Initialize LLM
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel("gemini-2.5-flash")  # Changed to match Colab
    
    # Combine chunks into context, truncating each chunk to avoid token overflow
    context = "\n\n".join([chunk["text"][:500] for chunk in chunks]) if chunks else "No relevant information found."
    
    # Create prompt emphasizing relevant, user-friendly answers
    prompt = (
        f"Identify the most relevant information in the context that directly answers the question. "
        f"Provide a concise answer in 1-3 bullet points using simple, user-friendly language, avoiding technical jargon unless necessary. "
        f"Include enough detail to be clear and helpful, but keep each point brief. "
        f"If no direct answer is found, provide the closest related information from the context to assist the user, or state: 'The information may lie beyond the scope of this document.'\n\n"
        f"Question: {query}\n\n"
        f"Context: {context}"
    )
    
    retries = 3
    for attempt in range(retries):
        try:
            # Generate response with moderate temperature for clarity
            response = model.generate_content(
                prompt,
                generation_config={"temperature": 0.3, "max_output_tokens": 300}
            )
            answer = response.text.strip()
            # Preserve line breaks for readability
            return answer if answer else "The information may lie beyond the scope of this document."
        except google.api_core.exceptions.ResourceExhausted:
            if attempt < retries - 1:
                sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                continue
            return "Error: API quota exceeded after retries."
        except Exception as e:
            return f"An error occurred while processing the query: {str(e)}"
    return "An error occurred while processing the query: Max retries reached."