import fitz  # PyMuPDF
import os
import requests
from langchain_core.documents import Document
from typing import List
import tempfile

def load_pdf_from_url(url: str) -> List[Document]:
    """Load a PDF from a URL and convert each page into a LangChain Document."""
    # Download the PDF
    response = requests.get(url)
    response.raise_for_status()  # Ensure the request was successful
    
    # Save to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
        temp_file.write(response.content)
        temp_file_path = temp_file.name
    
    # Open the PDF
    doc = fitz.open(temp_file_path)
    print(f"Loaded PDF from URL with {len(doc)} pages")
    
    # Convert each page into a LangChain Document with metadata
    documents = []
    pdf_name = os.path.basename(url) or "document.pdf"
    
    for i, page in enumerate(doc):
        text = page.get_text()
        metadata = {
            "source": pdf_name,
            "page": i + 1
        }
        documents.append(Document(page_content=text, metadata=metadata))
    
    doc.close()
    os.unlink(temp_file_path)  # Clean up temporary file
    
    print(f"Created {len(documents)} LangChain Document objects.")
    return documents

def detect_table_blocks(blocks, y_tolerance=5):
    """Group blocks by similar Y-values to detect rows."""
    lines = []
    current_line = []
    last_y = None

    for block in blocks:
        if block['type'] != 0:
            continue  # only text blocks

        for line in block['lines']:
            spans = line['spans']
            y = spans[0]['bbox'][1]

            if last_y is None or abs(y - last_y) <= y_tolerance:
                current_line.append(spans)
            else:
                if len(current_line) > 1:
                    lines.append(current_line)
                current_line = [spans]

            last_y = y

    if len(current_line) > 1:
        lines.append(current_line)

    return lines

def extract_layout_tables(pdf_path: str) -> List[Document]:
    """Extract layout-based tables from a PDF."""
    doc = fitz.open(pdf_path)
    layout_tables = []
    
    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        detected_lines = detect_table_blocks(blocks)
        
        if detected_lines:
            table_text = ""
            for row in detected_lines:
                row_text = " | ".join([span["text"].strip() for group in row for span in group])
                table_text += row_text + "\n"
            
            layout_tables.append(Document(
                page_content=table_text,
                metadata={
                    "page": page_num + 1,
                    "is_table": True,
                    "source": os.path.basename(pdf_path)
                }
            ))
    
    doc.close()
    print(f"Detected {len(layout_tables)} layout-based tables from the PDF.")
    return layout_tables
