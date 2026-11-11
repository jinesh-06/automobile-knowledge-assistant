import unittest
import os
import shutil
import tempfile
from unittest.mock import MagicMock

from modules.document_manager import DocumentManager

class TestFilenameSanitization(unittest.TestCase):
    """
    Unit tests for filename validation, sanitization, and directory confinement in DocumentManager.
    """

    def setUp(self):
        # Create an isolated temporary directory for testing file operations
        self.test_dir = tempfile.mkdtemp()
        self.mock_chroma = MagicMock()
        self.mock_embeddings = MagicMock()
        
        self.doc_manager = DocumentManager(self.mock_chroma, self.mock_embeddings)
        self.doc_manager.documents_dir = self.test_dir

    def tearDown(self):
        # Clean up temporary directory
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_normal_pdf_filenames(self):
        """Test standard legitimate PDF filenames."""
        valid_names = [
            "manual.pdf",
            "owner_manual_2023.pdf",
            "spec-sheet-v1.pdf",
            "maintenance_guide.PDF"
        ]
        for name in valid_names:
            clean = self.doc_manager.sanitize_and_validate_filename(name)
            self.assertEqual(clean, name.strip())

    def test_valid_filenames_with_spaces_and_parentheses(self):
        """Test legitimate PDF filenames containing spaces and special valid characters."""
        valid_names = [
            "2024 Toyota Camry Manual.pdf",
            "Honda Civic Service Guide (Part 1).pdf",
            "BMW 3 Series - 2021.pdf"
        ]
        for name in valid_names:
            clean = self.doc_manager.sanitize_and_validate_filename(name)
            self.assertEqual(clean, name.strip())

    def test_directory_traversal_attempts_rejected(self):
        """Test that relative traversal paths are strictly rejected."""
        traversal_attempts = [
            "../../evil.pdf",
            "..\\..\\evil.pdf",
            "sub/../../evil.pdf",
            "../etc/passwd.pdf",
            "..\\windows\\system32\\evil.pdf"
        ]
        for attempt in traversal_attempts:
            with self.assertRaises(ValueError, msg=f"Should reject: {attempt}"):
                self.doc_manager.sanitize_and_validate_filename(attempt)

    def test_absolute_paths_rejected(self):
        """Test that POSIX and Windows absolute paths are rejected."""
        absolute_paths = [
            "/etc/shadow.pdf",
            "/tmp/malicious.pdf",
            "\\Windows\\System32\\cmd.pdf"
        ]
        for path in absolute_paths:
            with self.assertRaises(ValueError, msg=f"Should reject: {path}"):
                self.doc_manager.sanitize_and_validate_filename(path)

    def test_windows_drive_letter_paths_rejected(self):
        """Test that Windows drive letter paths are rejected."""
        drive_paths = [
            "C:\\evil.pdf",
            "C:/evil.pdf",
            "D:something.pdf",
            "E:\\Documents\\manual.pdf"
        ]
        for path in drive_paths:
            with self.assertRaises(ValueError, msg=f"Should reject: {path}"):
                self.doc_manager.sanitize_and_validate_filename(path)

    def test_unexpected_extensions_rejected(self):
        """Test that non-PDF extensions and executable extensions are rejected."""
        invalid_extensions = [
            "manual.txt",
            "script.py",
            "payload.exe",
            "exploit.pdf.exe",
            "manual.doc",
            "config.json",
            "no_extension"
        ]
        for name in invalid_extensions:
            with self.assertRaises(ValueError, msg=f"Should reject: {name}"):
                self.doc_manager.sanitize_and_validate_filename(name)

    def test_blank_or_invalid_names_rejected(self):
        """Test that empty, blank, or non-string names are rejected."""
        invalid_names = ["", "   ", None, 123, ".pdf", "  .pdf  "]
        for name in invalid_names:
            with self.assertRaises(ValueError):
                self.doc_manager.sanitize_and_validate_filename(name)

    def test_delete_document_rejects_traversal(self):
        """Test that delete_document applies filename sanitization and does not delete external files."""
        # Create an external decoy file
        parent_dir = os.path.dirname(self.test_dir)
        canary_file = os.path.join(parent_dir, "canary_test_delete.txt")
        with open(canary_file, "w") as f:
            f.write("important data")

        try:
            with self.assertRaises(ValueError):
                self.doc_manager.delete_document("../canary_test_delete.txt")
            
            # Ensure canary file was NOT deleted
            self.assertTrue(os.path.exists(canary_file))
        finally:
            if os.path.exists(canary_file):
                os.remove(canary_file)

    def test_rejected_upload_does_not_modify_filesystem(self):
        """Verify that rejected uploads create no files on the filesystem."""
        initial_files = os.listdir(self.test_dir)
        with self.assertRaises(ValueError):
            self.doc_manager.process_and_index_document(
                file_content=b"%PDF-1.4 mock",
                document_name="../../escape.pdf"
            )
        self.assertEqual(os.listdir(self.test_dir), initial_files)

if __name__ == "__main__":
    unittest.main()
