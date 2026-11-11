import unittest
import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch

from modules.document_manager import DocumentManager

class TestReindexingSafety(unittest.TestCase):
    """
    Regression tests for document re-indexing safety in DocumentManager.
    Verifies that existing indexed chunks and disk files are preserved whenever
    extraction, chunking, or embedding fails during a replacement operation.
    Uses isolated temporary storage and mocks to ensure data/chroma_db is never touched.
    """

    def setUp(self):
        self.test_docs_dir = tempfile.mkdtemp()
        
        # Mock ChromaStore and its underlying Collection
        self.mock_chroma_store = MagicMock()
        self.mock_collection = MagicMock()
        self.mock_chroma_store.collection = self.mock_collection
        
        # Mock EmbeddingManager
        self.mock_embedding_manager = MagicMock()

        self.doc_manager = DocumentManager(
            chroma_store=self.mock_chroma_store,
            embedding_manager=self.mock_embedding_manager
        )
        self.doc_manager.documents_dir = self.test_docs_dir

        # Seed initial state with an existing document
        self.doc_name = "test_manual.pdf"
        self.initial_file_path = os.path.join(self.test_docs_dir, self.doc_name)
        with open(self.initial_file_path, "wb") as f:
            f.write(b"%PDF-1.4 initial content")

        # Initial collection metadata contains the existing document
        self.mock_chroma_store.get_all_documents_metadata.return_value = [
            {"name": self.doc_name, "pages": 2, "chunks": 4}
        ]
        self.mock_collection.get.return_value = {
            "ids": ["old_hash_p1_c0", "old_hash_p1_c1", "old_hash_p2_c0", "old_hash_p2_c1"]
        }

    def tearDown(self):
        if os.path.exists(self.test_docs_dir):
            shutil.rmtree(self.test_docs_dir)

    def test_reindex_pdf_extraction_failure_preserves_old_data(self):
        """If PDF text extraction fails, old indexed chunks and file must remain untouched."""
        # Simulate extraction failure
        self.doc_manager.doc_processor.process_pdf = MagicMock(return_value=[])

        bad_file_bytes = b"corrupted pdf bytes"
        with self.assertRaises(ValueError) as ctx:
            self.doc_manager.process_and_index_document(
                file_content=bad_file_bytes,
                document_name=self.doc_name,
                replace_existing=True
            )

        self.assertIn("No text could be extracted", str(ctx.exception))
        # Ensure ChromaStore deletion was NEVER called
        self.mock_chroma_store.delete_document.assert_not_called()
        self.mock_collection.delete.assert_not_called()
        self.mock_chroma_store.add_chunks.assert_not_called()

        # Ensure physical file content on disk remains the original content
        with open(self.initial_file_path, "rb") as f:
            self.assertEqual(f.read(), b"%PDF-1.4 initial content")

    def test_reindex_chunking_failure_preserves_old_data(self):
        """If chunking yields zero chunks, old indexed chunks and file must remain untouched."""
        self.doc_manager.doc_processor.process_pdf = MagicMock(return_value=[
            {"document_id": "newhash", "document_name": self.doc_name, "page_number": 1, "text": "Valid text"}
        ])
        # Simulate chunker failure
        self.doc_manager.chunker.chunk_pages = MagicMock(return_value=[])

        with self.assertRaises(ValueError) as ctx:
            self.doc_manager.process_and_index_document(
                file_content=b"%PDF-1.4 valid bytes",
                document_name=self.doc_name,
                replace_existing=True
            )

        self.assertIn("Document yielded no chunks", str(ctx.exception))
        self.mock_chroma_store.delete_document.assert_not_called()
        self.mock_collection.delete.assert_not_called()
        self.mock_chroma_store.add_chunks.assert_not_called()

        with open(self.initial_file_path, "rb") as f:
            self.assertEqual(f.read(), b"%PDF-1.4 initial content")

    def test_reindex_embedding_failure_preserves_old_data(self):
        """If embedding generation fails, old indexed chunks and file must remain untouched."""
        self.doc_manager.doc_processor.process_pdf = MagicMock(return_value=[
            {"document_id": "newhash", "document_name": self.doc_name, "page_number": 1, "text": "Valid text"}
        ])
        self.doc_manager.chunker.chunk_pages = MagicMock(return_value=[
            {"text": "Chunk text 1", "metadata": {"chunk_id": "newhash_p1_c0", "document_name": self.doc_name}}
        ])
        # Simulate embedding failure (empty list returned)
        self.mock_embedding_manager.get_embeddings.return_value = []

        with self.assertRaises(ValueError) as ctx:
            self.doc_manager.process_and_index_document(
                file_content=b"%PDF-1.4 valid bytes",
                document_name=self.doc_name,
                replace_existing=True
            )

        self.assertIn("Failed to generate embeddings", str(ctx.exception))
        self.mock_chroma_store.delete_document.assert_not_called()
        self.mock_collection.delete.assert_not_called()
        self.mock_chroma_store.add_chunks.assert_not_called()

        with open(self.initial_file_path, "rb") as f:
            self.assertEqual(f.read(), b"%PDF-1.4 initial content")

    def test_successful_reindex_updates_chunks_and_purges_stale(self):
        """On successful re-indexing, new chunks are added and old stale chunk IDs deleted."""
        new_chunks = [
            {"text": "New chunk 1", "metadata": {"chunk_id": "newhash_p1_c0", "document_name": self.doc_name}},
            {"text": "New chunk 2", "metadata": {"chunk_id": "newhash_p1_c1", "document_name": self.doc_name}}
        ]
        new_embeddings = [[0.1] * 384, [0.2] * 384]

        self.doc_manager.doc_processor.process_pdf = MagicMock(return_value=[
            {"document_id": "newhash", "document_name": self.doc_name, "page_number": 1, "text": "New text"}
        ])
        self.doc_manager.chunker.chunk_pages = MagicMock(return_value=new_chunks)
        self.mock_embedding_manager.get_embeddings.return_value = new_embeddings

        result = self.doc_manager.process_and_index_document(
            file_content=b"%PDF-1.4 updated content",
            document_name=self.doc_name,
            replace_existing=True
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["chunks"], 2)

        # 1. New chunks added
        self.mock_chroma_store.add_chunks.assert_called_once_with(new_chunks, new_embeddings)

        # 2. Stale chunks from previous version purged by ID
        self.mock_collection.delete.assert_called_once()
        deleted_ids = self.mock_collection.delete.call_args[1]["ids"]
        self.assertEqual(
            set(deleted_ids),
            {"old_hash_p1_c0", "old_hash_p1_c1", "old_hash_p2_c0", "old_hash_p2_c1"}
        )

        # 3. Disk file updated to new content
        with open(self.initial_file_path, "rb") as f:
            self.assertEqual(f.read(), b"%PDF-1.4 updated content")

    def test_reindex_without_replace_flag_raises_error(self):
        """Attempting to re-index an existing document without replace_existing=True raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            self.doc_manager.process_and_index_document(
                file_content=b"%PDF-1.4 new",
                document_name=self.doc_name,
                replace_existing=False
            )
        self.assertIn("already exists", str(ctx.exception))
        self.mock_chroma_store.add_chunks.assert_not_called()

    def test_reindex_partial_extraction_rejected_preserves_old_data_and_file(self):
        """Partially extracted replacement is rejected, preserving existing ChromaDB chunks and original disk file."""
        # Simulate partial extraction: 1 page extracted out of 2 total pages
        self.doc_manager.doc_processor.process_pdf = MagicMock(return_value=[
            {"document_id": "newhash", "document_name": self.doc_name, "page_number": 1, "text": "Extracted page 1"}
        ])
        self.doc_manager.doc_processor.last_processing_report = {
            "document_name": self.doc_name,
            "total_pages": 2,
            "extracted_pages": 1,
            "pages_requiring_ocr": 1,
            "failed_pages": [{"page": 2, "reason": "OCR engine crashed"}],
            "warnings": ["Page(s) [2] yielded no usable text."]
        }

        with self.assertRaises(ValueError) as ctx:
            self.doc_manager.process_and_index_document(
                file_content=b"%PDF-1.4 partially unreadable replacement",
                document_name=self.doc_name,
                replace_existing=True
            )

        err_msg = str(ctx.exception)
        self.assertIn(f"Cannot replace existing document '{self.doc_name}'", err_msg)
        self.assertIn("extraction succeeded for only 1 of 2 page(s)", err_msg)
        self.assertIn("Existing indexed document and physical file were preserved", err_msg)
        self.assertIn("failed page(s): [2]", err_msg)

        # ChromaDB mutations must NOT have occurred
        self.mock_chroma_store.add_chunks.assert_not_called()
        self.mock_collection.delete.assert_not_called()

        # Physical file on disk must retain original content
        with open(self.initial_file_path, "rb") as f:
            self.assertEqual(f.read(), b"%PDF-1.4 initial content")

    def test_reindex_chromadb_insertion_failure_preserves_old_data_and_file(self):
        """If ChromaDB chunk insertion fails, existing stale chunks are not deleted and disk file is untouched."""
        new_chunks = [
            {"text": "Valid new chunk", "metadata": {"chunk_id": "newhash_p1_c0", "document_name": self.doc_name}}
        ]
        self.doc_manager.doc_processor.process_pdf = MagicMock(return_value=[
            {"document_id": "newhash", "document_name": self.doc_name, "page_number": 1, "text": "Valid text"}
        ])
        self.doc_manager.doc_processor.last_processing_report = {
            "document_name": self.doc_name,
            "total_pages": 1,
            "extracted_pages": 1,
            "failed_pages": [],
            "warnings": []
        }
        self.doc_manager.chunker.chunk_pages = MagicMock(return_value=new_chunks)
        self.mock_embedding_manager.get_embeddings.return_value = [[0.1] * 384]

        # Simulate ChromaDB insertion failure
        self.mock_chroma_store.add_chunks.side_effect = RuntimeError("ChromaDB connection dropped during insertion")

        with self.assertRaises(RuntimeError) as ctx:
            self.doc_manager.process_and_index_document(
                file_content=b"%PDF-1.4 replacement",
                document_name=self.doc_name,
                replace_existing=True
            )

        self.assertIn("ChromaDB connection dropped", str(ctx.exception))
        # Stale chunk deletion must NOT be called
        self.mock_collection.delete.assert_not_called()
        # Physical file must remain the original content
        with open(self.initial_file_path, "rb") as f:
            self.assertEqual(f.read(), b"%PDF-1.4 initial content")

    def test_new_document_partial_extraction_allowed_with_warnings(self):
        """A new document with partial extraction is permitted to index with partial_success and warnings."""
        new_doc_name = "brand_new_manual.pdf"
        new_chunks = [
            {"text": "Chunk from page 1", "metadata": {"chunk_id": "newhash_p1_c0", "document_name": new_doc_name}}
        ]
        self.doc_manager.doc_processor.process_pdf = MagicMock(return_value=[
            {"document_id": "newhash", "document_name": new_doc_name, "page_number": 1, "text": "Extracted page 1"}
        ])
        self.doc_manager.doc_processor.last_processing_report = {
            "document_name": new_doc_name,
            "total_pages": 2,
            "extracted_pages": 1,
            "pages_requiring_ocr": 1,
            "failed_pages": [{"page": 2, "reason": "OCR engine crashed"}],
            "warnings": ["Page(s) [2] yielded no usable text."]
        }
        self.doc_manager.chunker.chunk_pages = MagicMock(return_value=new_chunks)
        self.mock_embedding_manager.get_embeddings.return_value = [[0.1] * 384]

        result = self.doc_manager.process_and_index_document(
            file_content=b"%PDF-1.4 new document bytes",
            document_name=new_doc_name,
            replace_existing=False
        )

        self.assertEqual(result["status"], "partial_success")
        self.assertEqual(result["pages"], 1)
        self.assertEqual(result["total_pages"], 2)
        self.assertEqual(result["chunks"], 1)
        self.assertTrue(len(result["warnings"]) > 0)
        self.assertTrue(len(result["failed_pages"]) > 0)

        # Chunks were added to ChromaDB
        self.mock_chroma_store.add_chunks.assert_called_once_with(new_chunks, [[0.1] * 384])

        # Physical file was written to disk
        new_file_path = os.path.join(self.test_docs_dir, new_doc_name)
        self.assertTrue(os.path.exists(new_file_path))
        with open(new_file_path, "rb") as f:
            self.assertEqual(f.read(), b"%PDF-1.4 new document bytes")

    def test_reindex_chunk_discovery_failure_aborts_and_preserves_data(self):
        """If ChromaDB fails during existing chunk discovery, abort immediately preserving DB and disk."""
        self.doc_manager.doc_processor.process_pdf = MagicMock(return_value=[
            {"document_id": "newhash", "document_name": self.doc_name, "page_number": 1, "text": "New text"}
        ])
        self.doc_manager.chunker.chunk_pages = MagicMock(return_value=[
            {"text": "New chunk", "metadata": {"chunk_id": "newhash_p1_c0", "document_name": self.doc_name}}
        ])
        self.mock_embedding_manager.get_embeddings.return_value = [[0.1] * 384]

        # Simulate discovery failure in ChromaDB
        self.mock_collection.get.side_effect = RuntimeError("ChromaDB connection dropped during discovery")

        temp_path = os.path.join(self.test_docs_dir, f".{self.doc_name}.tmp")

        with self.assertRaises(RuntimeError) as ctx:
            self.doc_manager.process_and_index_document(
                file_content=b"%PDF-1.4 updated",
                document_name=self.doc_name,
                replace_existing=True
            )

        self.assertIn("Failed to retrieve existing chunk IDs", str(ctx.exception))
        # Ensure no chunks were added or deleted
        self.mock_chroma_store.add_chunks.assert_not_called()
        self.mock_collection.delete.assert_not_called()

        # Ensure physical file remains old content
        with open(self.initial_file_path, "rb") as f:
            self.assertEqual(f.read(), b"%PDF-1.4 initial content")

        # Ensure temporary file was not created or was cleaned up
        self.assertFalse(os.path.exists(temp_path))

        # Verify retry succeeds once discovery is operational
        self.mock_collection.get.side_effect = None
        self.mock_collection.get.return_value = {"ids": ["old_hash_p1_c0"]}
        retry_result = self.doc_manager.process_and_index_document(
            file_content=b"%PDF-1.4 updated",
            document_name=self.doc_name,
            replace_existing=True
        )
        self.assertEqual(retry_result["status"], "success")
        with open(self.initial_file_path, "rb") as f:
            self.assertEqual(f.read(), b"%PDF-1.4 updated")

    def test_reindex_temp_file_write_failure_aborts_and_preserves_data(self):
        """If writing the temporary file fails (e.g. disk full), abort before ChromaDB mutation."""
        self.doc_manager.doc_processor.process_pdf = MagicMock(return_value=[
            {"document_id": "newhash", "document_name": self.doc_name, "page_number": 1, "text": "New text"}
        ])
        self.doc_manager.chunker.chunk_pages = MagicMock(return_value=[
            {"text": "New chunk", "metadata": {"chunk_id": "newhash_p1_c0", "document_name": self.doc_name}}
        ])
        self.mock_embedding_manager.get_embeddings.return_value = [[0.1] * 384]

        temp_path = os.path.join(self.test_docs_dir, f".{self.doc_name}.tmp")

        real_open = open
        def faulty_open(path, mode="r", *args, **kwargs):
            if f".{self.doc_name}.tmp" in str(path):
                raise IOError("No space left on device")
            return real_open(path, mode, *args, **kwargs)

        with patch("builtins.open", side_effect=faulty_open):
            with self.assertRaises(IOError) as ctx:
                self.doc_manager.process_and_index_document(
                    file_content=b"%PDF-1.4 updated",
                    document_name=self.doc_name,
                    replace_existing=True
                )
            self.assertIn("Failed to create temporary file", str(ctx.exception))

        # ChromaDB must not have been mutated
        self.mock_chroma_store.add_chunks.assert_not_called()
        self.mock_collection.delete.assert_not_called()

        # Disk file remains untouched
        with open(self.initial_file_path, "rb") as f:
            self.assertEqual(f.read(), b"%PDF-1.4 initial content")

        # Temp file does not exist
        self.assertFalse(os.path.exists(temp_path))

        # Retry without open failure completes successfully
        retry_result = self.doc_manager.process_and_index_document(
            file_content=b"%PDF-1.4 updated",
            document_name=self.doc_name,
            replace_existing=True
        )
        self.assertEqual(retry_result["status"], "success")
        with open(self.initial_file_path, "rb") as f:
            self.assertEqual(f.read(), b"%PDF-1.4 updated")

    def test_reindex_os_replace_permission_error_rolls_back_new_chunks(self):
        """If os.replace raises PermissionError, clean temp file and roll back newly added chunks."""
        new_chunks = [
            {"text": "New chunk 1", "metadata": {"chunk_id": "newhash_p1_c0", "document_name": self.doc_name}},
            {"text": "New chunk 2", "metadata": {"chunk_id": "newhash_p1_c1", "document_name": self.doc_name}}
        ]
        self.doc_manager.doc_processor.process_pdf = MagicMock(return_value=[
            {"document_id": "newhash", "document_name": self.doc_name, "page_number": 1, "text": "New text"}
        ])
        self.doc_manager.chunker.chunk_pages = MagicMock(return_value=new_chunks)
        self.mock_embedding_manager.get_embeddings.return_value = [[0.1] * 384, [0.2] * 384]

        temp_path = os.path.join(self.test_docs_dir, f".{self.doc_name}.tmp")

        with patch("os.replace", side_effect=PermissionError("[WinError 32] File is locked by another process")):
            with self.assertRaises(RuntimeError) as ctx:
                self.doc_manager.process_and_index_document(
                    file_content=b"%PDF-1.4 updated",
                    document_name=self.doc_name,
                    replace_existing=True
                )
            self.assertIn("Failed to replace physical document file", str(ctx.exception))
            self.assertIn("Newly inserted chunks were rolled back", str(ctx.exception))

        # Chunks were added
        self.mock_chroma_store.add_chunks.assert_called_once()
        # Rollback was called to delete the new chunk IDs
        self.mock_collection.delete.assert_called_once()
        deleted_ids = self.mock_collection.delete.call_args[1]["ids"]
        self.assertEqual(set(deleted_ids), {"newhash_p1_c0", "newhash_p1_c1"})

        # Physical file remains original content
        with open(self.initial_file_path, "rb") as f:
            self.assertEqual(f.read(), b"%PDF-1.4 initial content")

        # Temp file cleaned up
        self.assertFalse(os.path.exists(temp_path))

        # Retry once lock is released succeeds
        self.mock_collection.delete.reset_mock()
        self.mock_chroma_store.add_chunks.reset_mock()
        retry_result = self.doc_manager.process_and_index_document(
            file_content=b"%PDF-1.4 updated",
            document_name=self.doc_name,
            replace_existing=True
        )
        self.assertEqual(retry_result["status"], "success")
        with open(self.initial_file_path, "rb") as f:
            self.assertEqual(f.read(), b"%PDF-1.4 updated")

    def test_reindex_stale_deletion_first_batch_failure_reports_partial_failure(self):
        """If stale-chunk deletion fails on the first batch, report partial_failure with recovery guidance."""
        new_chunks = [
            {"text": "New chunk 1", "metadata": {"chunk_id": "newhash_p1_c0", "document_name": self.doc_name}}
        ]
        self.doc_manager.doc_processor.process_pdf = MagicMock(return_value=[
            {"document_id": "newhash", "document_name": self.doc_name, "page_number": 1, "text": "New text"}
        ])
        self.doc_manager.chunker.chunk_pages = MagicMock(return_value=new_chunks)
        self.mock_embedding_manager.get_embeddings.return_value = [[0.1] * 384]

        # Simulate deletion failure on batch 1
        self.mock_collection.delete.side_effect = RuntimeError("ChromaDB delete timeout")

        result = self.doc_manager.process_and_index_document(
            file_content=b"%PDF-1.4 updated",
            document_name=self.doc_name,
            replace_existing=True
        )

        # Status must be partial_failure, not success
        self.assertEqual(result["status"], "partial_failure")
        self.assertIn("ChromaDB delete timeout", result["error"])
        self.assertTrue(result["failed_stale_chunks"] > 0)
        self.assertIn("retry uploading", result["recovery_message"].lower())

        # Physical file was updated to new content
        with open(self.initial_file_path, "rb") as f:
            self.assertEqual(f.read(), b"%PDF-1.4 updated")

        # New chunks were added
        self.mock_chroma_store.add_chunks.assert_called_once()
        # Old chunks were attempted to delete
        self.mock_collection.delete.assert_called_once()

        # Temp file cleaned up
        temp_path = os.path.join(self.test_docs_dir, f".{self.doc_name}.tmp")
        self.assertFalse(os.path.exists(temp_path))

        # Retry cleans up stale chunks
        self.mock_collection.delete.side_effect = None
        self.mock_collection.delete.reset_mock()
        self.mock_collection.get.return_value = {
            "ids": ["old_hash_p1_c0", "old_hash_p1_c1", "old_hash_p2_c0", "old_hash_p2_c1", "newhash_p1_c0"]
        }
        retry_result = self.doc_manager.process_and_index_document(
            file_content=b"%PDF-1.4 updated",
            document_name=self.doc_name,
            replace_existing=True
        )
        self.assertEqual(retry_result["status"], "success")
        # Stale chunks were purged on retry
        self.mock_collection.delete.assert_called_once()
        purged = self.mock_collection.delete.call_args[1]["ids"]
        self.assertNotIn("newhash_p1_c0", purged)

    def test_reindex_stale_deletion_multi_batch_partial_failure_reports_remaining_count(self):
        """If batch 1 of stale deletion succeeds but batch 2 fails, report remaining un-deleted count."""
        # Setup 150 old chunks
        old_ids = [f"old_chunk_{i}" for i in range(150)]
        self.mock_collection.get.return_value = {"ids": old_ids}

        new_chunks = [
            {"text": "New chunk 1", "metadata": {"chunk_id": "newhash_p1_c0", "document_name": self.doc_name}}
        ]
        self.doc_manager.doc_processor.process_pdf = MagicMock(return_value=[
            {"document_id": "newhash", "document_name": self.doc_name, "page_number": 1, "text": "New text"}
        ])
        self.doc_manager.chunker.chunk_pages = MagicMock(return_value=new_chunks)
        self.mock_embedding_manager.get_embeddings.return_value = [[0.1] * 384]

        # First batch (100) succeeds, second batch (50) fails
        delete_calls = []
        def delete_side_effect(ids=None):
            delete_calls.append(ids)
            if len(delete_calls) == 2:
                raise RuntimeError("Batch 2 ChromaDB connection reset")

        self.mock_collection.delete.side_effect = delete_side_effect

        result = self.doc_manager.process_and_index_document(
            file_content=b"%PDF-1.4 updated",
            document_name=self.doc_name,
            replace_existing=True
        )

        self.assertEqual(result["status"], "partial_failure")
        self.assertEqual(result["failed_stale_chunks"], 50)
        self.assertIn("50 stale chunk(s)", result["recovery_message"])
        self.assertIn("Batch 2 ChromaDB connection reset", result["error"])

        # Batch 1 was processed
        self.assertEqual(len(delete_calls[0]), 100)

        # File is new version
        with open(self.initial_file_path, "rb") as f:
            self.assertEqual(f.read(), b"%PDF-1.4 updated")

        # Retry cleans up the remaining 50
        self.mock_collection.delete.side_effect = None
        self.mock_collection.get.return_value = {"ids": old_ids[100:] + ["newhash_p1_c0"]}
        retry_result = self.doc_manager.process_and_index_document(
            file_content=b"%PDF-1.4 updated",
            document_name=self.doc_name,
            replace_existing=True
        )
        self.assertEqual(retry_result["status"], "success")

    def test_reindex_file_replace_failure_after_insertion_preserves_old_file_and_rolls_back(self):
        """New chunk insertion succeeds, but file replacement fails: old file is preserved and new chunks rolled back."""
        new_chunks = [
            {"text": "New chunk 1", "metadata": {"chunk_id": "newhash_p1_c0", "document_name": self.doc_name}}
        ]
        self.doc_manager.doc_processor.process_pdf = MagicMock(return_value=[
            {"document_id": "newhash", "document_name": self.doc_name, "page_number": 1, "text": "New text"}
        ])
        self.doc_manager.chunker.chunk_pages = MagicMock(return_value=new_chunks)
        self.mock_embedding_manager.get_embeddings.return_value = [[0.1] * 384]

        temp_path = os.path.join(self.test_docs_dir, f".{self.doc_name}.tmp")

        with patch("os.replace", side_effect=OSError("Cross-device link or disk write failure")):
            with self.assertRaises(RuntimeError) as ctx:
                self.doc_manager.process_and_index_document(
                    file_content=b"%PDF-1.4 updated",
                    document_name=self.doc_name,
                    replace_existing=True
                )
            self.assertIn("Failed to replace physical document file", str(ctx.exception))

        # Physical file is STILL old version
        with open(self.initial_file_path, "rb") as f:
            self.assertEqual(f.read(), b"%PDF-1.4 initial content")

        # Rollback deleted the new chunk
        self.mock_collection.delete.assert_called_once_with(ids=["newhash_p1_c0"])

        # Temp file cleaned up
        self.assertFalse(os.path.exists(temp_path))

    def test_reindex_identical_content_rollback_does_not_delete_valid_chunks(self):
        """When replacing with identical content, a file replacement failure does NOT delete valid existing chunks."""
        # Chunk IDs match existing IDs exactly because content is identical
        identical_chunk_ids = ["old_hash_p1_c0", "old_hash_p1_c1", "old_hash_p2_c0", "old_hash_p2_c1"]
        identical_chunks = [
            {"text": f"Chunk {i}", "metadata": {"chunk_id": cid, "document_name": self.doc_name}}
            for i, cid in enumerate(identical_chunk_ids)
        ]

        self.doc_manager.doc_processor.process_pdf = MagicMock(return_value=[
            {"document_id": "old_hash", "document_name": self.doc_name, "page_number": 1, "text": "Initial text"}
        ])
        self.doc_manager.chunker.chunk_pages = MagicMock(return_value=identical_chunks)
        self.mock_embedding_manager.get_embeddings.return_value = [[0.1] * 384] * 4

        with patch("os.replace", side_effect=PermissionError("Locked file")):
            with self.assertRaises(RuntimeError) as ctx:
                self.doc_manager.process_and_index_document(
                    file_content=b"%PDF-1.4 initial content",
                    document_name=self.doc_name,
                    replace_existing=True
                )
            self.assertIn("Failed to replace physical document file", str(ctx.exception))

        # Because all new_chunk_ids overlap with old_chunk_ids, rollback_ids was empty!
        # collection.delete must NEVER have been called!
        self.mock_collection.delete.assert_not_called()

        # Old file remains intact
        with open(self.initial_file_path, "rb") as f:
            self.assertEqual(f.read(), b"%PDF-1.4 initial content")

    def test_reindex_file_replace_and_rollback_failure_preserves_old_file_and_reports_both_errors(self):
        """
        When replacing an existing document, if chunk insertion succeeds but os.replace fails
        AND subsequent rollback deletion of newly inserted chunks also fails:
        1. The original physical PDF remains unchanged.
        2. The caller receives a clear RuntimeError describing both the file-replacement failure
           and the rollback failure.
        3. The operation never reports success.
        4. Identifies that newly inserted chunks remain in ChromaDB (orphaned) when rollback fails.
        5. Existing chunk IDs that overlap between old and new content are not incorrectly deleted.
        6. Isolated mocks ensure production ChromaDB data is never mutated.
        """
        # Define new chunks: one overlapping chunk ID ("old_hash_p1_c0") and one brand-new chunk ID ("new_hash_p1_c1")
        # In setUp, existing IDs are: ["old_hash_p1_c0", "old_hash_p1_c1", "old_hash_p2_c0", "old_hash_p2_c1"]
        new_chunks = [
            {"text": "Overlapping chunk content", "metadata": {"chunk_id": "old_hash_p1_c0", "document_name": self.doc_name}},
            {"text": "Brand new chunk content", "metadata": {"chunk_id": "new_hash_p1_c1", "document_name": self.doc_name}}
        ]
        self.doc_manager.doc_processor.process_pdf = MagicMock(return_value=[
            {"document_id": "new_hash", "document_name": self.doc_name, "page_number": 1, "text": "New text"}
        ])
        self.doc_manager.chunker.chunk_pages = MagicMock(return_value=new_chunks)
        self.mock_embedding_manager.get_embeddings.return_value = [[0.1] * 384, [0.2] * 384]

        temp_path = os.path.join(self.test_docs_dir, f".{self.doc_name}.tmp")

        # Force os.replace to fail, and force collection.delete (during rollback) to fail as well
        replace_exc = OSError("Disk write error during atomic file replace")
        rollback_exc = RuntimeError("ChromaDB delete failed during rollback")

        with patch("os.replace", side_effect=replace_exc):
            self.mock_collection.delete.side_effect = rollback_exc

            with self.assertRaises(RuntimeError) as ctx:
                self.doc_manager.process_and_index_document(
                    file_content=b"%PDF-1.4 updated content",
                    document_name=self.doc_name,
                    replace_existing=True
                )

        err_msg = str(ctx.exception)

        # Acceptance Criteria:
        # 1. Caller receives a clear RuntimeError describing both file-replacement failure and rollback failure
        self.assertIn("Failed to replace physical file", err_msg)
        self.assertIn("Disk write error during atomic file replace", err_msg)
        self.assertIn("rollback of newly inserted chunks encountered an error", err_msg)
        self.assertIn("ChromaDB delete failed during rollback", err_msg)
        self.assertIn("Retrying replacement will restore consistency", err_msg)

        # 2. Original physical PDF remains unchanged
        with open(self.initial_file_path, "rb") as f:
            self.assertEqual(f.read(), b"%PDF-1.4 initial content")

        # 3. Temporary file was cleaned up
        self.assertFalse(os.path.exists(temp_path))

        # 4. New chunks were added to ChromaDB prior to the replace failure
        self.mock_chroma_store.add_chunks.assert_called_once_with(
            new_chunks, [[0.1] * 384, [0.2] * 384]
        )

        # 5. Rollback deletion was attempted ONLY for the non-overlapping new chunk ID
        # Overlapping existing chunk ("old_hash_p1_c0") was NOT included in rollback delete call
        self.mock_collection.delete.assert_called_once()
        attempted_rollback_ids = self.mock_collection.delete.call_args[1]["ids"]
        self.assertEqual(attempted_rollback_ids, ["new_hash_p1_c1"])
        self.assertNotIn("old_hash_p1_c0", attempted_rollback_ids)

        # 6. Because rollback deletion failed with RuntimeError, the newly inserted chunks
        # remain in ChromaDB (orphaned) until a subsequent retry restores consistency.
        # Verify recovery on retry once file replace and ChromaDB are restored:
        self.mock_collection.delete.side_effect = None
        self.mock_collection.delete.reset_mock()
        self.mock_chroma_store.add_chunks.reset_mock()
        self.mock_collection.get.return_value = {
            "ids": ["old_hash_p1_c0", "old_hash_p1_c1", "old_hash_p2_c0", "old_hash_p2_c1", "new_hash_p1_c1"]
        }
        retry_result = self.doc_manager.process_and_index_document(
            file_content=b"%PDF-1.4 updated content",
            document_name=self.doc_name,
            replace_existing=True
        )
        self.assertEqual(retry_result["status"], "success")
        with open(self.initial_file_path, "rb") as f:
            self.assertEqual(f.read(), b"%PDF-1.4 updated content")


if __name__ == "__main__":
    unittest.main()

