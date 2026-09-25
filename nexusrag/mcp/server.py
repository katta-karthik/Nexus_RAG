import sys
import json
from typing import Dict, Any, List, Optional
from nexusrag.indexing.vectorstore import VectorStoreManager
from nexusrag.indexing.embeddings import EmbeddingManager, EmbeddingProvider
from nexusrag.retrieval.semantic import SemanticRetriever
from nexusrag.retrieval.keyword import BM25KeywordRetriever
from nexusrag.retrieval.hybrid import HybridRetriever


class NexusRAGMCPServer:
    """
    Lightweight MCP server exposing document operations to external agents:
    - search_documents(query, strategy, top_k)
    - list_documents()
    - get_document_chunks(doc_id, limit)
    """

    def __init__(self, persist_dir: str = "./chroma_db"):
        self.vectorstore = VectorStoreManager(persist_directory=persist_dir)
        self.embedder = EmbeddingManager(provider=EmbeddingProvider.LOCAL)
        self.semantic_retriever = SemanticRetriever(self.vectorstore, self.embedder)
        
        # Ingest existing chunks into BM25
        all_chunks = self.vectorstore.get_chunks_for_document(limit=500)
        self.keyword_retriever = BM25KeywordRetriever(all_chunks)
        self.hybrid_retriever = HybridRetriever(self.semantic_retriever, self.keyword_retriever)

    def list_documents(self) -> Dict[str, Any]:
        """Lists all indexed documents and vector store stats."""
        stats = self.vectorstore.get_stats()
        return {
            "total_documents": stats["unique_documents"],
            "documents": stats["document_names"],
            "total_vectors": stats["total_vectors"],
            "status": stats["status"],
        }

    def search_documents(
        self, query: str, strategy: str = "hybrid", top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Searches documents using semantic, keyword, or hybrid retrieval."""
        if strategy == "semantic":
            return self.semantic_retriever.retrieve(query, top_k=top_k)
        elif strategy == "keyword":
            return self.keyword_retriever.retrieve(query, top_k=top_k)
        else:
            res = self.hybrid_retriever.retrieve(query, top_k=top_k)
            return res.merged_candidates

    def get_document_chunks(
        self, filename: Optional[str] = None, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Retrieves raw chunks for a document."""
        return self.vectorstore.get_chunks_for_document(filename=filename, limit=limit)

    def handle_request(self, request_json: str) -> str:
        """Simple JSON-RPC 2.0 dispatch."""
        try:
            req = json.loads(request_json)
            method = req.get("method")
            params = req.get("params", {})
            req_id = req.get("id")

            if method == "list_documents":
                result = self.list_documents()
            elif method == "search_documents":
                result = self.search_documents(
                    query=params.get("query", ""),
                    strategy=params.get("strategy", "hybrid"),
                    top_k=params.get("top_k", 5),
                )
            elif method == "get_document_chunks":
                result = self.get_document_chunks(
                    filename=params.get("filename"),
                    limit=params.get("limit", 20),
                )
            else:
                return json.dumps(
                    {"jsonrpc": "2.0", "error": {"code": -32601, "message": f"Method {method} not found"}, "id": req_id}
                )

            return json.dumps({"jsonrpc": "2.0", "result": result, "id": req_id})
        except Exception as e:
            return json.dumps({"jsonrpc": "2.0", "error": {"code": -32000, "message": str(e)}, "id": None})


if __name__ == "__main__":
    server = NexusRAGMCPServer()
    print("NexusRAG MCP Server initialized in stdio mode. Send JSON-RPC requests via stdin.")
    for line in sys.stdin:
        if not line.strip():
            continue
        response = server.handle_request(line)
        sys.stdout.write(response + "\n")
        sys.stdout.flush()
