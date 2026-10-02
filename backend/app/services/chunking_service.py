from typing import List, Dict, Any
from ..config import settings

def chunk_pages_data(
    pages_data: List[Dict[str, Any]],
    chunk_size: int = None,
    chunk_overlap: int = None
) -> List[Dict[str, Any]]:
    """
    Splits page-by-page extracted text into deterministic chunks.
    Each chunk preserves its page_number and chunk_index.
    """
    if chunk_size is None:
        chunk_size = settings.CHUNK_SIZE
    if chunk_overlap is None:
        chunk_overlap = settings.CHUNK_OVERLAP
        
    chunks = []
    chunk_index = 0
    step = max(1, chunk_size - chunk_overlap)
    
    for page in pages_data:
        page_num = page["page_number"]
        text = page["text"]
        if not text:
            continue
            
        # If page text is smaller than chunk_size, create a single chunk for this page
        if len(text) <= chunk_size:
            chunks.append({
                "chunk_index": chunk_index,
                "page_number": page_num,
                "content": text,
                "token_count": len(text.split())
            })
            chunk_index += 1
        else:
            # Sliding window over characters/words on this page
            start = 0
            while start < len(text):
                end = start + chunk_size
                chunk_text = text[start:end].strip()
                if chunk_text:
                    chunks.append({
                        "chunk_index": chunk_index,
                        "page_number": page_num,
                        "content": chunk_text,
                        "token_count": len(chunk_text.split())
                    })
                    chunk_index += 1
                start += step
                
    return chunks
