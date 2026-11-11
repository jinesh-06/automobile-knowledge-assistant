from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Any
import uuid

class DocumentChunker:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def chunk_pages(self, pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Split page texts into chunks while preserving metadata.
        """
        chunks_data = []
        for page in pages_data:
            text = page["text"]
            if not text:
                continue
                
            chunks = self.text_splitter.split_text(text)
            
            for idx, chunk_text in enumerate(chunks):
                chunk_id = f"{page['document_id']}_p{page['page_number']}_c{idx}"
                chunk_record = {
                    "text": chunk_text,
                    "metadata": {
                        "document_id": page["document_id"],
                        "document_name": page["document_name"],
                        "page_number": page["page_number"],
                        "chunk_id": chunk_id,
                        "document_type": page.get("document_type", "vehicle_manual"),
                        "extraction_method": page.get("extraction_method", "text")
                    }
                }
                chunks_data.append(chunk_record)
                
        return chunks_data
