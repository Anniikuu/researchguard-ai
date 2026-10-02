import fitz  # PyMuPDF
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class PDFExtractionError(Exception):
    pass

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Extracts text page-by-page from PDF binary data using PyMuPDF.
    Preserves 1-indexed page numbers.
    """
    if not pdf_bytes:
        raise PDFExtractionError("PDF bytes buffer is empty")
    
    pages_data = []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page_idx in range(len(doc)):
            page_num = page_idx + 1
            page = doc[page_idx]
            text = page.get_text("text") or ""
            pages_data.append({
                "page_number": page_num,
                "text": text.strip()
            })
        doc.close()
    except Exception as e:
        logger.error(f"PyMuPDF failed to extract PDF: {e}")
        raise PDFExtractionError(f"Failed to extract PDF text: {str(e)}")
        
    return pages_data

def extract_text_from_pdf_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Extracts text page-by-page from a PDF file on disk.
    """
    try:
        with open(file_path, "rb") as f:
            pdf_bytes = f.read()
        return extract_text_from_pdf_bytes(pdf_bytes)
    except Exception as e:
        logger.error(f"Error reading PDF file {file_path}: {e}")
        raise PDFExtractionError(f"Error reading file {file_path}: {str(e)}")
