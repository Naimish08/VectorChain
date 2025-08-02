from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List

def semantic_chunking(documents: List[Document], layout_tables: List[Document]) -> List[Document]:
    """Perform semantic chunking on documents and combine with layout-based tables."""
    # Combine all pages into one long string
    full_text = "\n".join([doc.page_content for doc in documents])
    
    # Semantic-aware splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )
    
    # Split into semantic chunks
    split_docs = splitter.create_documents([full_text])
    
    print(f"Generated {len(split_docs)} semantic chunks.")
    
    # Assign chunk IDs to metadata
    for i, doc in enumerate(split_docs):
        doc.metadata.update({
            "chunk_id": f"chunk_{i}"
        })
    
    # Combine semantic text chunks and layout-based tables
    all_documents = split_docs + layout_tables
    
    # Add a 'type' tag for clarity
    for doc in all_documents:
        if doc.metadata.get("is_table", False):
            doc.metadata["type"] = "table"
        else:
            doc.metadata["type"] = "text"
    
    print(f"Total documents: {len(all_documents)}")
    print(f"- Semantic chunks: {len(split_docs)}")
    print(f"- Tables: {len(layout_tables)}")
    
    return all_documents