import os
import re
from typing import List, Dict, Any
from modules.document_processor import DocumentProcessor
from modules.chunker import DocumentChunker
from modules.embeddings import EmbeddingManager
from modules.chroma_store import ChromaStore
import logging

logger = logging.getLogger(__name__)

class DocumentManager:
    def __init__(self, chroma_store: ChromaStore, embedding_manager: EmbeddingManager):
        self.chroma_store = chroma_store
        self.embedding_manager = embedding_manager
        self.doc_processor = DocumentProcessor()
        self.chunker = DocumentChunker()
        
        self.documents_dir = "data/documents"
        os.makedirs(self.documents_dir, exist_ok=True)

    def sanitize_and_validate_filename(self, document_name: str) -> str:
        """
        Validate and sanitize document_name.
        Rejects directory traversal, drive letters, absolute paths, invalid characters,
        and non-PDF extensions. Confines file operations to self.documents_dir.
        """
        if not document_name or not isinstance(document_name, str):
            raise ValueError("Document name must be a non-empty string.")

        clean_name = document_name.strip()
        if not clean_name:
            raise ValueError("Document name cannot be blank.")

        # Disallow directory separators and traversal tokens
        if "/" in clean_name or "\\" in clean_name or ".." in clean_name:
            raise ValueError(f"Invalid document name '{document_name}': path traversal or separators are not permitted.")

        # Disallow drive-letter prefixes on Windows (e.g., 'C:' or 'D:')
        if re.match(r'^[a-zA-Z]:', clean_name):
            raise ValueError(f"Invalid document name '{document_name}': drive-letter paths are not permitted.")

        # Disallow illegal filename characters on Windows: < > : " / \ | ? * and control chars
        if re.search(r'[<>:"/\\|?*\x00-\x1f]', clean_name):
            raise ValueError(f"Invalid document name '{document_name}': contains illegal filesystem characters.")

        # Validate file extension (must be .pdf)
        base, ext = os.path.splitext(clean_name)
        if ext.lower() != ".pdf":
            raise ValueError(f"Invalid file extension '{ext}'. Only .pdf files are supported.")

        # Ensure filename base is not empty (e.g. '.pdf')
        if not base.strip():
            raise ValueError(f"Invalid document name '{document_name}': missing filename base.")

        # Resolve path and verify confinement strictly within documents_dir
        base_dir = os.path.abspath(self.documents_dir)
        target_path = os.path.abspath(os.path.join(base_dir, clean_name))

        if os.path.dirname(target_path) != base_dir:
            raise ValueError(f"Security error: target path '{target_path}' escapes document directory '{base_dir}'.")

        return clean_name

    def process_and_index_document(self, file_content: bytes, document_name: str, replace_existing: bool = False):
        """
        Full pipeline: validate, extract text, chunk, embed, and store.
        Ensures existing index data is preserved if extraction or embedding fails.
        """
        # Validate filename and target confinement
        clean_name = self.sanitize_and_validate_filename(document_name)
        file_path = os.path.join(self.documents_dir, clean_name)

        if not file_content or not isinstance(file_content, bytes):
            raise ValueError("File content must be non-empty bytes.")

        # Check if document already exists in ChromaDB or on disk
        existing_docs = [d["name"] for d in self.chroma_store.get_all_documents_metadata()]
        is_existing = (clean_name in existing_docs) or os.path.exists(file_path)

        if is_existing and not replace_existing:
            raise ValueError(f"Document {clean_name} already exists.")

        # Step 1: Extract text from PDF (in-memory; does not alter DB or disk)
        pages_data = self.doc_processor.process_pdf(file_content, clean_name)
        if not pages_data:
            raise ValueError("No text could be extracted from the document.")

        # Check for partial extraction safety on document replacement
        report = getattr(self.doc_processor, "last_processing_report", {})
        failed_pages = report.get("failed_pages", [])
        total_pages = report.get("total_pages", len(pages_data))
        is_partial = bool(failed_pages) or (total_pages > len(pages_data))

        if is_existing and is_partial:
            failed_nums = [
                p["page"] if isinstance(p, dict) and "page" in p else p
                for p in failed_pages
            ]
            failed_info = f" (failed page(s): {failed_nums})" if failed_nums else ""
            raise ValueError(
                f"Cannot replace existing document '{clean_name}': extraction succeeded for only "
                f"{len(pages_data)} of {total_pages} page(s){failed_info}. "
                f"Existing indexed document and physical file were preserved."
            )

        # Step 2: Chunk pages (in-memory)
        chunks_data = self.chunker.chunk_pages(pages_data)
        if not chunks_data:
            raise ValueError("Document yielded no chunks after processing.")

        # Step 3: Generate embeddings (in-memory)
        texts_to_embed = [chunk["text"] for chunk in chunks_data]
        embeddings = self.embedding_manager.get_embeddings(texts_to_embed)
        if not embeddings or len(embeddings) != len(chunks_data):
            raise ValueError("Failed to generate embeddings for document chunks.")

        # Step 4: Discover existing chunk IDs if replacing
        # Fail fast: if retrieval of existing chunk IDs fails, abort immediately
        # before touching ChromaDB or writing any files to disk.
        old_chunk_ids = []
        if is_existing:
            try:
                existing_records = self.chroma_store.collection.get(
                    where={"document_name": clean_name},
                    include=[]
                )
                old_chunk_ids = existing_records.get("ids", [])
            except Exception as e:
                logger.error(f"Failed to fetch existing chunk IDs for {clean_name}: {e}")
                raise RuntimeError(
                    f"Failed to retrieve existing chunk IDs for document '{clean_name}': {e}. "
                    f"Replacement operation aborted to prevent data inconsistency."
                ) from e

        # Step 5: Prepare temporary file on disk before modifying ChromaDB
        temp_path = os.path.join(self.documents_dir, f".{clean_name}.tmp")
        try:
            with open(temp_path, "wb") as f:
                f.write(file_content)
        except Exception as e:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            logger.error(f"Failed to write temporary file for {clean_name}: {e}")
            raise IOError(
                f"Failed to create temporary file for '{clean_name}': {e}. "
                f"Replacement aborted before modifying database or existing file."
            ) from e

        # Step 6: Insert / upsert new chunks into ChromaDB
        new_chunk_ids = {chunk["metadata"]["chunk_id"] for chunk in chunks_data}
        try:
            self.chroma_store.add_chunks(chunks_data, embeddings)
        except Exception as e:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            logger.error(f"Failed to insert new chunks for {clean_name}: {e}")
            raise

        # Step 7: Atomically replace the physical file on disk
        try:
            os.replace(temp_path, file_path)
        except Exception as e:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            logger.error(f"Failed to replace physical file for {clean_name}: {e}")

            # Roll back newly inserted chunks that do NOT overlap with pre-existing IDs.
            # Never delete valid existing chunks (e.g. if replacing with identical content).
            rollback_ids = [cid for cid in new_chunk_ids if cid not in old_chunk_ids]
            rollback_error = None
            if rollback_ids:
                try:
                    for i in range(0, len(rollback_ids), 100):
                        self.chroma_store.collection.delete(ids=rollback_ids[i:i+100])
                    logger.info(f"Successfully rolled back {len(rollback_ids)} new chunk(s) for {clean_name}.")
                except Exception as rb_err:
                    rollback_error = rb_err
                    logger.error(f"Failed to roll back new chunks for {clean_name}: {rb_err}")

            if rollback_error:
                raise RuntimeError(
                    f"Failed to replace physical file '{clean_name}': {e}. "
                    f"Additionally, rollback of newly inserted chunks encountered an error: {rollback_error}. "
                    f"Existing document file remains preserved on disk. Retrying replacement will restore consistency."
                ) from e
            else:
                raise RuntimeError(
                    f"Failed to replace physical document file '{clean_name}': {e}. "
                    f"Newly inserted chunks were rolled back and existing document was preserved."
                ) from e

        # Step 8: Purge stale chunks from previous version if any
        stale_deletion_error = None
        remaining_stale_ids = []
        if old_chunk_ids:
            stale_ids = [cid for cid in old_chunk_ids if cid not in new_chunk_ids]
            if stale_ids:
                try:
                    for i in range(0, len(stale_ids), 100):
                        batch = stale_ids[i:i+100]
                        try:
                            self.chroma_store.collection.delete(ids=batch)
                        except Exception as batch_err:
                            remaining_stale_ids = stale_ids[i:]
                            raise batch_err
                except Exception as e:
                    stale_deletion_error = str(e)
                    logger.error(
                        f"Error removing stale chunks for {clean_name}: {e}. "
                        f"{len(remaining_stale_ids)} stale chunk(s) could not be removed."
                    )

        # Step 9: Assemble report and return status
        report = getattr(self.doc_processor, "last_processing_report", {})
        warnings = list(report.get("warnings", []))
        failed_pages = report.get("failed_pages", [])
        total_pages = report.get("total_pages", len(pages_data))
        has_partial = bool(failed_pages) or (total_pages > len(pages_data))

        if stale_deletion_error:
            recovery_msg = (
                f"Physical file was updated and {len(chunks_data)} new chunks were indexed, "
                f"but {len(remaining_stale_ids)} stale chunk(s) from the previous version could not be removed "
                f"due to a database error: {stale_deletion_error}. "
                f"To resolve this, retry uploading '{clean_name}' with replacement enabled."
            )
            warnings.append(recovery_msg)
            return {
                "status": "partial_failure",
                "document_name": clean_name,
                "pages": len(pages_data),
                "total_pages": total_pages,
                "chunks": len(chunks_data),
                "warnings": warnings,
                "failed_pages": failed_pages,
                "error": stale_deletion_error,
                "failed_stale_chunks": len(remaining_stale_ids),
                "recovery_message": recovery_msg
            }

        return {
            "status": "partial_success" if has_partial else "success",
            "document_name": clean_name,
            "pages": len(pages_data),
            "total_pages": total_pages,
            "chunks": len(chunks_data),
            "warnings": warnings,
            "failed_pages": failed_pages
        }

    def delete_document(self, document_name: str):
        """
        Delete a document from ChromaDB and the file system after validation.
        """
        clean_name = self.sanitize_and_validate_filename(document_name)
        self.chroma_store.delete_document(clean_name)
        file_path = os.path.join(self.documents_dir, clean_name)
        if os.path.exists(file_path):
            os.remove(file_path)
