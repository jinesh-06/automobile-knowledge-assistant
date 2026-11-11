from typing import List, Dict, Any
from modules.chroma_store import ChromaStore
from modules.embeddings import EmbeddingManager

class Retriever:
    def __init__(self, chroma_store: ChromaStore, embedding_manager: EmbeddingManager, default_threshold: float = 0.25):
        self.chroma_store = chroma_store
        self.embedding_manager = embedding_manager
        self.default_threshold = default_threshold
        
    def retrieve(self, query: str, top_k: int = 5, filter_document_name: str = None, threshold: float = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks from ChromaDB for a given query.
        Discards chunks with relevance < threshold.
        """
        effective_threshold = threshold if threshold is not None else self.default_threshold
        query_embedding = self.embedding_manager.get_query_embedding(query)
        if not query_embedding:
            return []
            
        # Query extra candidates to account for threshold and deduplication
        candidate_count = max(top_k * 2, 10)
        results = self.chroma_store.query(
            query_embedding=query_embedding,
            top_k=candidate_count,
            filter_document_name=filter_document_name
        )
        
        # Filter by threshold and deduplicate by text content
        seen_chunks = set()
        filtered_results = []
        for res in results:
            chunk_sig = " ".join(res["text"].split()[:40])
            if chunk_sig not in seen_chunks and res["relevance"] >= effective_threshold:
                seen_chunks.add(chunk_sig)
                filtered_results.append(res)
            if len(filtered_results) >= top_k:
                break
        
        return filtered_results
