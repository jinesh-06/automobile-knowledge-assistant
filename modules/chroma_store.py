import chromadb
from chromadb.config import Settings
import os
from typing import List, Dict, Any

class ChromaStore:
    def __init__(self, persist_directory: str = "data/chroma_db"):
        self.persist_directory = persist_directory
        os.makedirs(self.persist_directory, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection_name = "automobile_knowledge"
        
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks_data: List[Dict[str, Any]], embeddings: List[List[float]]):
        """
        Add chunks and their embeddings to ChromaDB.
        """
        if not chunks_data:
            return
            
        ids = [chunk["metadata"]["chunk_id"] for chunk in chunks_data]
        documents = [chunk["text"] for chunk in chunks_data]
        metadatas = [chunk["metadata"] for chunk in chunks_data]
        
        # Batch insert if large
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            self.collection.upsert(
                ids=ids[i:i+batch_size],
                embeddings=embeddings[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size],
                documents=documents[i:i+batch_size]
            )

    def delete_document(self, document_name: str):
        """
        Delete all chunks associated with a document_name.
        """
        self.collection.delete(
            where={"document_name": document_name}
        )

    def query(self, query_embedding: List[float], top_k: int = 5, filter_document_name: str = None) -> List[Dict[str, Any]]:
        """
        Query the collection for similar chunks.
        """
        where_clause = None
        if filter_document_name and filter_document_name != "All Documents":
            where_clause = {"document_name": filter_document_name}
            
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_clause,
            include=["documents", "metadatas", "distances"]
        )
        
        formatted_results = []
        if results['ids'] and results['ids'][0]:
            for idx in range(len(results['ids'][0])):
                # Distance in cosine space (lower is more similar) -> Similarity = 1 - distance
                similarity = 1.0 - results['distances'][0][idx]
                formatted_results.append({
                    "id": results['ids'][0][idx],
                    "text": results['documents'][0][idx],
                    "metadata": results['metadatas'][0][idx],
                    "relevance": round(similarity, 3)
                })
        return formatted_results

    def get_all_documents_metadata(self) -> List[Dict[str, Any]]:
        """
        Return unique document names and page counts.
        """
        # Fetching a small chunk just to get metadatas, this is a simplified approach
        # For large scale, we should maintain a separate metadata table, but we will aggregate here
        try:
            results = self.collection.get(include=['metadatas'])
            metadatas = results['metadatas']
            doc_info = {}
            for meta in metadatas:
                doc_name = meta['document_name']
                page_num = meta['page_number']
                if doc_name not in doc_info:
                    doc_info[doc_name] = {"pages": set(), "chunks": 0}
                doc_info[doc_name]["pages"].add(page_num)
                doc_info[doc_name]["chunks"] += 1
                
            return [
                {
                    "name": name,
                    "pages": len(info["pages"]),
                    "chunks": info["chunks"]
                }
                for name, info in doc_info.items()
            ]
        except Exception:
            return []

    def get_collection_stats(self) -> Dict[str, int]:
        """
        Return stats for the dashboard.
        """
        try:
            docs = self.get_all_documents_metadata()
            total_documents = len(docs)
            total_pages = sum(d["pages"] for d in docs)
            total_chunks = self.collection.count()
            return {
                "documents": total_documents,
                "pages": total_pages,
                "chunks": total_chunks,
                "vectors": total_chunks
            }
        except Exception:
            return {"documents": 0, "pages": 0, "chunks": 0, "vectors": 0}
