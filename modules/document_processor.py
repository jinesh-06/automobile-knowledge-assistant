import fitz  # PyMuPDF
from PIL import Image
import io
import re
import hashlib
from typing import List, Dict, Any, Optional
from modules.ocr_processor import OCRProcessor
import logging

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, text_length_threshold: int = 50, ocr_processor: Optional[OCRProcessor] = None):
        self.text_length_threshold = text_length_threshold
        self.ocr_processor = ocr_processor or OCRProcessor()
        self.last_processing_report: Dict[str, Any] = {}

    def clean_text(self, text: str) -> str:
        """
        Clean excess whitespace while preserving numbers, units, and structure.
        """
        if not text:
            return ""
        # Remove consecutive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Remove excessive spaces
        text = re.sub(r'[ \t]{2,}', ' ', text)
        return text.strip()

    def generate_document_id(self, file_content: bytes) -> str:
        """Generate a SHA256 hash for the document to prevent duplicates."""
        return hashlib.sha256(file_content).hexdigest()

    def process_pdf(self, file_content: bytes, document_name: str) -> List[Dict[str, Any]]:
        """
        Process a PDF file and extract text page by page.
        Returns a list of page data dictionaries.
        Guarantees deterministic cleanup of PDF documents and image buffers.
        """
        if not file_content:
            self.last_processing_report = {
                "document_name": document_name,
                "total_pages": 0,
                "extracted_pages": 0,
                "pages_requiring_ocr": 0,
                "failed_pages": [],
                "warnings": ["File content is empty."]
            }
            return []

        document_id = self.generate_document_id(file_content)
        pages_data = []
        failed_pages = []
        pages_requiring_ocr = 0

        doc = None
        total_pages = 0
        try:
            doc = fitz.open(stream=file_content, filetype="pdf")
            total_pages = len(doc)

            for page_idx in range(total_pages):
                page_num = page_idx + 1  # 1-indexed
                page = doc.load_page(page_idx)
                text = page.get_text("text") or ""
                extraction_method = "text"

                # Check if page has insufficient text and needs OCR fallback
                if len(text.strip()) < self.text_length_threshold:
                    pages_requiring_ocr += 1
                    if self.ocr_processor.is_available:
                        pix = None
                        img = None
                        img_bytes_io = None
                        try:
                            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x scale for better OCR
                            img_bytes = pix.tobytes("png")
                            img_bytes_io = io.BytesIO(img_bytes)
                            img = Image.open(img_bytes_io)
                            ocr_text = self.ocr_processor.perform_ocr(img)

                            if ocr_text and ocr_text.strip():
                                text = ocr_text
                                extraction_method = "ocr"
                            else:
                                failed_pages.append({
                                    "page": page_num,
                                    "reason": "OCR returned empty text"
                                })
                        except Exception as e:
                            failed_pages.append({
                                "page": page_num,
                                "reason": f"OCR execution failed: {str(e)}"
                            })
                            logger.warning(f"Failed to OCR page {page_num} of {document_name}: {e}")
                        finally:
                            # Deterministically release image and pixmap memory
                            if img is not None:
                                try:
                                    img.close()
                                except Exception:
                                    pass
                                img = None
                            if img_bytes_io is not None:
                                try:
                                    img_bytes_io.close()
                                except Exception:
                                    pass
                                img_bytes_io = None
                            pix = None
                    else:
                        failed_pages.append({
                            "page": page_num,
                            "reason": "Tesseract OCR is not available"
                        })
                        logger.warning(f"Page {page_num} of '{document_name}' requires OCR, but Tesseract is not available.")

                cleaned_text = self.clean_text(text)
                if cleaned_text:
                    pages_data.append({
                        "document_id": document_id,
                        "document_name": document_name,
                        "page_number": page_num,  # 1-indexed
                        "text": cleaned_text,
                        "document_type": "vehicle_manual",
                        "extraction_method": extraction_method
                    })

        finally:
            if doc is not None:
                try:
                    doc.close()
                except Exception:
                    pass
                doc = None

        # Assemble processing report
        warnings = []
        if failed_pages:
            failed_nums = [p["page"] for p in failed_pages]
            warnings.append(
                f"Page(s) {failed_nums} in '{document_name}' yielded no usable text (either empty or OCR unavailable/failed)."
            )

        self.last_processing_report = {
            "document_name": document_name,
            "total_pages": total_pages,
            "extracted_pages": len(pages_data),
            "pages_requiring_ocr": pages_requiring_ocr,
            "failed_pages": failed_pages,
            "warnings": warnings
        }

        # If every page required OCR and Tesseract was unavailable, raise actionable error
        if total_pages > 0 and len(pages_data) == 0:
            if pages_requiring_ocr == total_pages and not self.ocr_processor.is_available:
                raise RuntimeError(
                    f"Tesseract OCR is required to extract text from scanned document '{document_name}', "
                    f"but Tesseract is not installed or discoverable. "
                    f"Please install Tesseract OCR and ensure it is in your system PATH, or set the TESSERACT_PATH environment variable."
                )
            if failed_pages:
                reasons = "; ".join(f"p.{p['page']}: {p['reason']}" for p in failed_pages[:3])
                raise ValueError(f"No text could be extracted from '{document_name}'. Failures: {reasons}")
            raise ValueError(f"No text could be extracted from '{document_name}'. Document contains no readable content.")

        return pages_data
