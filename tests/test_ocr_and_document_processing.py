import unittest
from unittest.mock import MagicMock, patch
import fitz
import io
import os
import shutil
import tempfile
from PIL import Image

from modules.ocr_processor import OCRProcessor
from modules.document_processor import DocumentProcessor
from modules.document_manager import DocumentManager


def create_test_pdf(pages_text: list) -> bytes:
    """
    Create an in-memory PDF with specified text per page.
    If page text is empty string, creates a page with no selectable text.
    """
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        if text:
            page.insert_text((50, 50), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


class TestOCRAndDocumentProcessing(unittest.TestCase):
    """
    Comprehensive tests for OCR reliability, Tesseract discovery, scanned-PDF handling,
    deterministic resource cleanup, and error/warning propagation.
    """

    def setUp(self):
        self.mock_ocr = MagicMock(spec=OCRProcessor)
        self.mock_ocr.is_available = True
        self.mock_ocr.unavailable_reason = ""
        self.mock_ocr.perform_ocr = MagicMock(return_value="")

    # 1. Digital PDF text extraction without OCR
    def test_digital_pdf_text_extraction_without_ocr(self):
        """Digital PDF with sufficient text extracts directly via PyMuPDF without invoking OCR."""
        text_p1 = "This is a detailed automobile manual page with more than fifty characters of text."
        text_p2 = "Another digital page describing brake caliper maintenance and fluid inspection."
        pdf_bytes = create_test_pdf([text_p1, text_p2])

        processor = DocumentProcessor(ocr_processor=self.mock_ocr)
        pages_data = processor.process_pdf(pdf_bytes, "digital_manual.pdf")

        self.assertEqual(len(pages_data), 2)
        self.assertEqual(pages_data[0]["page_number"], 1)
        self.assertEqual(pages_data[0]["extraction_method"], "text")
        self.assertIn("automobile manual page", pages_data[0]["text"])
        self.assertEqual(pages_data[1]["page_number"], 2)
        self.assertEqual(pages_data[1]["extraction_method"], "text")
        self.assertIn("brake caliper", pages_data[1]["text"])

        # OCR must not be invoked for pages with sufficient digital text
        self.mock_ocr.perform_ocr.assert_not_called()

    # 2. Scanned PDF with mocked successful OCR
    def test_scanned_pdf_with_mocked_successful_ocr(self):
        """Pages with insufficient text trigger OCR and record extraction_method='ocr'."""
        pdf_bytes = create_test_pdf([""])  # Blank page (<50 chars)
        self.mock_ocr.perform_ocr.return_value = "Recovered technical manual specifications via OCR."

        processor = DocumentProcessor(ocr_processor=self.mock_ocr)
        pages_data = processor.process_pdf(pdf_bytes, "scanned_manual.pdf")

        self.assertEqual(len(pages_data), 1)
        self.assertEqual(pages_data[0]["page_number"], 1)
        self.assertEqual(pages_data[0]["extraction_method"], "ocr")
        self.assertEqual(pages_data[0]["text"], "Recovered technical manual specifications via OCR.")
        self.mock_ocr.perform_ocr.assert_called_once()

    # 3. Scanned PDF with Tesseract unavailable
    def test_scanned_pdf_tesseract_unavailable_raises_actionable_error(self):
        """When all pages require OCR and Tesseract is unavailable, raise actionable error with setup guidance."""
        pdf_bytes = create_test_pdf(["", ""])  # 2 scanned pages
        self.mock_ocr.is_available = False
        self.mock_ocr.unavailable_reason = "Tesseract is not installed or not in PATH."

        processor = DocumentProcessor(ocr_processor=self.mock_ocr)
        with self.assertRaises(RuntimeError) as ctx:
            processor.process_pdf(pdf_bytes, "scanned_doc.pdf")

        error_msg = str(ctx.exception)
        self.assertIn("scanned_doc.pdf", error_msg)
        self.assertIn("Tesseract OCR is required", error_msg)
        self.assertIn("TESSERACT_PATH", error_msg)

    # 4. Tesseract discovery through an environment variable
    def test_tesseract_discovery_through_environment_variable(self):
        """Verify OCRProcessor discovers and validates Tesseract from TESSERACT_PATH."""
        fake_path = r"C:\fake\path\tesseract.exe" if os.name == "nt" else "/fake/path/tesseract"
        with patch.dict(os.environ, {"TESSERACT_PATH": fake_path}), \
             patch("os.path.isfile", return_value=True), \
             patch("pytesseract.get_tesseract_version", return_value="5.3.3"):
            ocr = OCRProcessor()
            self.assertTrue(ocr.is_available)
            self.assertEqual(ocr.executable_path, fake_path)
            self.assertEqual(ocr.tesseract_version, "5.3.3")

    # 5. Mixed digital and scanned PDF pages
    def test_mixed_digital_and_scanned_pdf_pages(self):
        """Mixed PDFs extract digital text directly and use OCR for scanned pages."""
        digital_text = "Page one has ample digital text describing the engine oil viscosity specifications."
        pdf_bytes = create_test_pdf([digital_text, ""])  # Page 1 digital, Page 2 scanned

        self.mock_ocr.perform_ocr.return_value = "Page two tire pressure table recovered by OCR."

        processor = DocumentProcessor(ocr_processor=self.mock_ocr)
        pages_data = processor.process_pdf(pdf_bytes, "mixed_manual.pdf")

        self.assertEqual(len(pages_data), 2)
        self.assertEqual(pages_data[0]["page_number"], 1)
        self.assertEqual(pages_data[0]["extraction_method"], "text")
        self.assertIn("engine oil viscosity", pages_data[0]["text"])

        self.assertEqual(pages_data[1]["page_number"], 2)
        self.assertEqual(pages_data[1]["extraction_method"], "ocr")
        self.assertIn("tire pressure table", pages_data[1]["text"])

    # 6. Mixed PDF where OCR fails on one page but other pages succeed
    def test_mixed_pdf_ocr_fails_on_one_page_preserves_successful_page(self):
        """Failure on one scanned page does not discard successfully extracted pages."""
        p1_text = "Page one has valid digital text regarding transmission fluid maintenance intervals."
        p3_text = "Page three has valid digital text regarding brake pad thickness measurements."
        pdf_bytes = create_test_pdf([p1_text, "", p3_text])  # Page 2 is scanned

        self.mock_ocr.perform_ocr.side_effect = RuntimeError("OCR engine crashed on page 2")

        processor = DocumentProcessor(ocr_processor=self.mock_ocr)
        pages_data = processor.process_pdf(pdf_bytes, "mixed_failing.pdf")

        # Pages 1 and 3 must be preserved
        self.assertEqual(len(pages_data), 2)
        self.assertEqual(pages_data[0]["page_number"], 1)
        self.assertEqual(pages_data[1]["page_number"], 3)

        # Failed page recorded in report
        report = processor.last_processing_report
        self.assertEqual(len(report["failed_pages"]), 1)
        self.assertEqual(report["failed_pages"][0]["page"], 2)
        self.assertIn("crashed", report["failed_pages"][0]["reason"])

    # 7. OCR returns empty text
    def test_ocr_returns_empty_text_no_empty_chunks(self):
        """Pages where OCR produces empty text are discarded without creating empty chunks."""
        p1_text = "Page one has valid digital text with more than fifty characters to pass threshold."
        pdf_bytes = create_test_pdf([p1_text, ""])

        self.mock_ocr.perform_ocr.return_value = "   \n  "  # Empty OCR

        processor = DocumentProcessor(ocr_processor=self.mock_ocr)
        pages_data = processor.process_pdf(pdf_bytes, "empty_ocr.pdf")

        self.assertEqual(len(pages_data), 1)
        self.assertEqual(pages_data[0]["page_number"], 1)
        report = processor.last_processing_report
        self.assertEqual(len(report["failed_pages"]), 1)
        self.assertEqual(report["failed_pages"][0]["page"], 2)

    # 8. OCR raises an exception on a page
    def test_ocr_raises_exception_handled_safely(self):
        """Page-level OCR exception is caught, logged, and does not terminate the pipeline."""
        p1_text = "Page one has valid digital text that exceeds the fifty character threshold."
        pdf_bytes = create_test_pdf([p1_text, ""])

        self.mock_ocr.perform_ocr.side_effect = Exception("Unreadable corrupted image")

        processor = DocumentProcessor(ocr_processor=self.mock_ocr)
        pages_data = processor.process_pdf(pdf_bytes, "exception_ocr.pdf")

        self.assertEqual(len(pages_data), 1)
        self.assertEqual(pages_data[0]["page_number"], 1)
        self.assertEqual(processor.last_processing_report["failed_pages"][0]["page"], 2)

    # 9. All pages produce no usable text
    def test_all_pages_produce_no_usable_text_fails_explicitly(self):
        """When no pages produce usable text, fail explicitly with descriptive ValueError."""
        pdf_bytes = create_test_pdf(["", ""])
        self.mock_ocr.perform_ocr.return_value = ""

        processor = DocumentProcessor(ocr_processor=self.mock_ocr)
        with self.assertRaises(ValueError) as ctx:
            processor.process_pdf(pdf_bytes, "blank_document.pdf")
        self.assertIn("No text could be extracted", str(ctx.exception))

    # 10. Correct 1-indexed page numbers and extraction-method metadata
    def test_correct_1_indexed_page_numbers_and_metadata(self):
        """Verify page numbers start at 1 and metadata fields match schema."""
        pdf_bytes = create_test_pdf([
            "Page one text with more than fifty characters in automobile documentation.",
            "Page two text with more than fifty characters in automobile documentation.",
            "Page three text with more than fifty characters in automobile documentation."
        ])

        processor = DocumentProcessor(ocr_processor=self.mock_ocr)
        pages_data = processor.process_pdf(pdf_bytes, "car_specs.pdf")

        self.assertEqual(len(pages_data), 3)
        for idx, page in enumerate(pages_data):
            self.assertEqual(page["page_number"], idx + 1)
            self.assertEqual(page["document_name"], "car_specs.pdf")
            self.assertEqual(page["extraction_method"], "text")
            self.assertEqual(page["document_type"], "vehicle_manual")
            self.assertTrue(len(page["document_id"]) > 0)

    # 11. PDF document closure after successful processing
    def test_pdf_document_closed_after_successful_processing(self):
        """PyMuPDF document is closed deterministically on success."""
        pdf_bytes = create_test_pdf(["Valid digital text with more than fifty characters."])
        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 1
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Valid digital text with more than fifty characters."
        mock_doc.load_page.return_value = mock_page

        with patch("fitz.open", return_value=mock_doc):
            processor = DocumentProcessor(ocr_processor=self.mock_ocr)
            pages = processor.process_pdf(pdf_bytes, "test.pdf")
            self.assertEqual(len(pages), 1)
            mock_doc.close.assert_called_once()

    # 12. PDF document closure when extraction raises an exception
    def test_pdf_document_closed_when_extraction_raises_exception(self):
        """PyMuPDF document is closed deterministically even when an exception is raised."""
        pdf_bytes = create_test_pdf(["Valid text."])
        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 1
        mock_doc.load_page.side_effect = RuntimeError("Fatal PyMuPDF read error")

        with patch("fitz.open", return_value=mock_doc):
            processor = DocumentProcessor(ocr_processor=self.mock_ocr)
            with self.assertRaises(RuntimeError):
                processor.process_pdf(pdf_bytes, "corrupt.pdf")
            mock_doc.close.assert_called_once()

    # 13. Image resource cleanup on success and exception
    def test_image_resource_cleanup_on_success_and_exception(self):
        """PIL Image handles are closed deterministically in OCR path on both success and exception."""
        pdf_bytes = create_test_pdf([""])  # Scanned page

        mock_image = MagicMock(spec=Image.Image)
        mock_image.close = MagicMock()

        # Case A: Success
        self.mock_ocr.perform_ocr.return_value = "OCR Output"
        with patch("PIL.Image.open", return_value=mock_image):
            processor = DocumentProcessor(ocr_processor=self.mock_ocr)
            processor.process_pdf(pdf_bytes, "cleanup_test.pdf")
            mock_image.close.assert_called()

        mock_image.close.reset_mock()

        # Case B: Exception during OCR
        self.mock_ocr.perform_ocr.side_effect = Exception("OCR error")
        with patch("PIL.Image.open", return_value=mock_image):
            processor = DocumentProcessor(ocr_processor=self.mock_ocr)
            with self.assertRaises(ValueError):
                processor.process_pdf(pdf_bytes, "cleanup_test2.pdf")
            mock_image.close.assert_called()

    # 14. User-facing error and warning propagation
    def test_user_facing_error_and_warning_propagation(self):
        """DocumentManager surfaces actionable errors and returns warnings for partial extractions."""
        test_dir = tempfile.mkdtemp()
        try:
            mock_chroma = MagicMock()
            mock_chroma.get_all_documents_metadata.return_value = []
            mock_embed = MagicMock()
            mock_embed.get_embeddings.return_value = [[0.1] * 384]

            doc_manager = DocumentManager(chroma_store=mock_chroma, embedding_manager=mock_embed)
            doc_manager.documents_dir = test_dir

            # Sub-test A: Scanned PDF without Tesseract propagates actionable error
            scanned_pdf = create_test_pdf([""])
            doc_manager.doc_processor.ocr_processor.is_available = False

            with self.assertRaises(RuntimeError) as ctx:
                doc_manager.process_and_index_document(scanned_pdf, "scanned_doc.pdf")
            self.assertIn("Tesseract OCR is required", str(ctx.exception))
            self.assertIn("scanned_doc.pdf", str(ctx.exception))

            # Sub-test B: Partial success (1 digital page, 1 failed OCR page) returns partial_success status and warnings
            p1_text = "Page one has valid digital text that exceeds the fifty character threshold."
            mixed_pdf = create_test_pdf([p1_text, ""])

            doc_manager.doc_processor.ocr_processor.is_available = True
            doc_manager.doc_processor.ocr_processor.perform_ocr = MagicMock(side_effect=Exception("OCR fail"))

            result = doc_manager.process_and_index_document(mixed_pdf, "mixed_doc.pdf")
            self.assertEqual(result["status"], "partial_success")
            self.assertEqual(result["pages"], 1)
            self.assertEqual(result["total_pages"], 2)
            self.assertTrue(len(result["warnings"]) > 0)
            self.assertTrue(len(result["failed_pages"]) > 0)

        finally:
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)


if __name__ == "__main__":
    unittest.main()
