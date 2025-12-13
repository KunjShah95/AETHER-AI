"""
Smart RAG (Retrieval-Augmented Generation) for AetherAI.

This module provides:
- Local knowledge base indexing
- Semantic search over documents
- Project-aware context retrieval
- Conversation memory with embeddings
- Multi-modal document support (text, code, markdown)
"""

import os
import json
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Try importing embedding libraries
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    EMBEDDINGS_AVAILABLE = False


@dataclass
class Document:
    """Represents a document in the knowledge base."""
    doc_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    chunks: List[str] = field(default_factory=list)
    chunk_embeddings: List[List[float]] = field(default_factory=list)


@dataclass
class SearchResult:
    """Represents a search result."""
    doc_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_index: int = 0


class SmartRAG:
    """Smart RAG system for AetherAI knowledge base."""
    
    def __init__(self, base_dir: Optional[str] = None, 
                 model_name: str = "all-MiniLM-L6-v2"):
        """Initialize Smart RAG.
        
        Args:
            base_dir: Directory for storing index.
            model_name: Sentence transformer model name.
        """
        self.base_dir = base_dir or os.path.join(
            os.getenv('HOME') or os.getenv('USERPROFILE') or os.path.expanduser('~'),
            '.nexus', 'knowledge'
        )
        os.makedirs(self.base_dir, exist_ok=True)
        
        self.index_file = os.path.join(self.base_dir, 'index.json')
        self.embeddings_file = os.path.join(self.base_dir, 'embeddings.npz')
        
        self.model_name = model_name
        self.model = None
        self.documents: Dict[str, Document] = {}
        self.embeddings_cache: Dict[str, List[float]] = {}
        
        # Chunk settings
        self.chunk_size = 500  # Characters per chunk
        self.chunk_overlap = 100  # Overlap between chunks
        
        # Load existing index
        self._load_index()
    
    def _load_model(self):
        """Lazy load the embedding model."""
        if self.model is None and EMBEDDINGS_AVAILABLE:
            try:
                self.model = SentenceTransformer(self.model_name)
            except Exception as e:
                print(f"Warning: Could not load embedding model: {e}")
    
    def _load_index(self):
        """Load index from disk."""
        try:
            if os.path.exists(self.index_file):
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for doc_id, doc_data in data.get('documents', {}).items():
                    self.documents[doc_id] = Document(
                        doc_id=doc_id,
                        content=doc_data.get('content', ''),
                        metadata=doc_data.get('metadata', {}),
                        chunks=doc_data.get('chunks', [])
                    )
            
            # Load embeddings if numpy available
            if NUMPY_AVAILABLE and os.path.exists(self.embeddings_file):
                data = np.load(self.embeddings_file, allow_pickle=True)
                self.embeddings_cache = dict(data['embeddings'].item())
        except Exception as e:
            print(f"Warning: Could not load index: {e}")
    
    def _save_index(self):
        """Save index to disk."""
        try:
            data = {
                'documents': {
                    doc_id: {
                        'content': doc.content,
                        'metadata': doc.metadata,
                        'chunks': doc.chunks
                    }
                    for doc_id, doc in self.documents.items()
                },
                'updated_at': datetime.now().isoformat()
            }
            
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
            
            # Save embeddings
            if NUMPY_AVAILABLE and self.embeddings_cache:
                np.savez(self.embeddings_file, embeddings=self.embeddings_cache)
                
        except Exception as e:
            print(f"Warning: Could not save index: {e}")
    
    # =========================================================================
    # Document Management
    # =========================================================================
    
    def add_document(self, content: str, doc_id: Optional[str] = None,
                     metadata: Optional[Dict] = None) -> str:
        """Add a document to the knowledge base.
        
        Args:
            content: Document content.
            doc_id: Optional document ID.
            metadata: Optional metadata.
            
        Returns:
            Document ID.
        """
        # Generate ID if not provided
        if not doc_id:
            doc_id = hashlib.md5(content[:1000].encode()).hexdigest()[:12]
        
        # Create chunks
        chunks = self._chunk_text(content)
        
        # Create document
        doc = Document(
            doc_id=doc_id,
            content=content,
            metadata=metadata or {},
            chunks=chunks
        )
        
        # Generate embeddings
        if EMBEDDINGS_AVAILABLE and self.model is None:
            self._load_model()
        
        if self.model:
            try:
                # Embed chunks
                chunk_embeddings = self.model.encode(chunks).tolist()
                
                # Store in cache
                for i, emb in enumerate(chunk_embeddings):
                    self.embeddings_cache[f"{doc_id}_{i}"] = emb
                    
            except Exception as e:
                print(f"Warning: Could not generate embeddings: {e}")
        
        self.documents[doc_id] = doc
        self._save_index()
        
        return doc_id
    
    def add_file(self, filepath: str) -> Optional[str]:
        """Add a file to the knowledge base.
        
        Args:
            filepath: Path to file.
            
        Returns:
            Document ID or None.
        """
        try:
            path = Path(filepath)
            if not path.exists():
                return None
            
            content = path.read_text(encoding='utf-8')
            
            metadata = {
                'source': 'file',
                'filepath': str(path.absolute()),
                'filename': path.name,
                'extension': path.suffix,
                'size': path.stat().st_size,
                'added_at': datetime.now().isoformat()
            }
            
            return self.add_document(content, doc_id=path.stem, metadata=metadata)
            
        except Exception as e:
            print(f"Error adding file: {e}")
            return None
    
    def add_directory(self, directory: str, 
                      extensions: List[str] = None) -> List[str]:
        """Add all matching files from a directory.
        
        Args:
            directory: Directory path.
            extensions: File extensions to include.
            
        Returns:
            List of added document IDs.
        """
        extensions = extensions or ['.py', '.md', '.txt', '.json', '.yaml', '.yml']
        added = []
        
        path = Path(directory)
        if not path.is_dir():
            return added
        
        for ext in extensions:
            for filepath in path.rglob(f"*{ext}"):
                # Skip hidden files and common vendor directories
                if any(p.startswith('.') or p in ['node_modules', 'venv', '__pycache__'] 
                       for p in filepath.parts):
                    continue
                
                doc_id = self.add_file(str(filepath))
                if doc_id:
                    added.append(doc_id)
        
        return added
    
    def remove_document(self, doc_id: str) -> bool:
        """Remove a document from the knowledge base.
        
        Args:
            doc_id: Document ID.
            
        Returns:
            True if removed.
        """
        if doc_id in self.documents:
            # Remove embeddings
            doc = self.documents[doc_id]
            for i in range(len(doc.chunks)):
                key = f"{doc_id}_{i}"
                if key in self.embeddings_cache:
                    del self.embeddings_cache[key]
            
            del self.documents[doc_id]
            self._save_index()
            return True
        return False
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks.
        
        Args:
            text: Text to chunk.
            
        Returns:
            List of chunks.
        """
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = text.rfind('.', start, end)
                last_newline = text.rfind('\n', start, end)
                break_point = max(last_period, last_newline)
                
                if break_point > start + self.chunk_size // 2:
                    end = break_point + 1
            
            chunks.append(text[start:end].strip())
            start = end - self.chunk_overlap
        
        return [c for c in chunks if c]
    
    # =========================================================================
    # Search
    # =========================================================================
    
    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Search the knowledge base.
        
        Args:
            query: Search query.
            top_k: Number of results to return.
            
        Returns:
            List of search results.
        """
        if not self.documents:
            return []
        
        if EMBEDDINGS_AVAILABLE and self.embeddings_cache:
            return self._semantic_search(query, top_k)
        else:
            return self._keyword_search(query, top_k)
    
    def _semantic_search(self, query: str, top_k: int) -> List[SearchResult]:
        """Perform semantic search using embeddings.
        
        Args:
            query: Search query.
            top_k: Number of results.
            
        Returns:
            Sorted search results.
        """
        if not EMBEDDINGS_AVAILABLE or not NUMPY_AVAILABLE:
            return self._keyword_search(query, top_k)
        
        if self.model is None:
            self._load_model()
        
        if not self.model:
            return self._keyword_search(query, top_k)
        
        try:
            # Encode query
            query_embedding = self.model.encode([query])[0]
            
            results = []
            
            for doc_id, doc in self.documents.items():
                for i, chunk in enumerate(doc.chunks):
                    key = f"{doc_id}_{i}"
                    if key not in self.embeddings_cache:
                        continue
                    
                    chunk_embedding = np.array(self.embeddings_cache[key])
                    
                    # Cosine similarity
                    similarity = np.dot(query_embedding, chunk_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
                    )
                    
                    results.append(SearchResult(
                        doc_id=doc_id,
                        content=chunk,
                        score=float(similarity),
                        metadata=doc.metadata,
                        chunk_index=i
                    ))
            
            # Sort by score
            results.sort(key=lambda x: x.score, reverse=True)
            
            return results[:top_k]
            
        except Exception as e:
            print(f"Semantic search error: {e}")
            return self._keyword_search(query, top_k)
    
    def _keyword_search(self, query: str, top_k: int) -> List[SearchResult]:
        """Perform keyword-based search.
        
        Args:
            query: Search query.
            top_k: Number of results.
            
        Returns:
            Sorted search results.
        """
        query_words = set(query.lower().split())
        results = []
        
        for doc_id, doc in self.documents.items():
            for i, chunk in enumerate(doc.chunks):
                chunk_lower = chunk.lower()
                
                # Calculate simple TF-based score
                score = sum(1 for word in query_words if word in chunk_lower)
                
                if score > 0:
                    results.append(SearchResult(
                        doc_id=doc_id,
                        content=chunk,
                        score=score / len(query_words),
                        metadata=doc.metadata,
                        chunk_index=i
                    ))
        
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
    
    def get_context_for_query(self, query: str, max_tokens: int = 2000) -> str:
        """Get relevant context for a query.
        
        Args:
            query: Query to find context for.
            max_tokens: Maximum context length (rough estimate).
            
        Returns:
            Relevant context string.
        """
        results = self.search(query, top_k=5)
        
        if not results:
            return ""
        
        context_parts = []
        current_length = 0
        
        for result in results:
            chunk_length = len(result.content.split())
            
            if current_length + chunk_length > max_tokens:
                break
            
            source = result.metadata.get('filename', result.doc_id)
            context_parts.append(f"[From: {source}]\n{result.content}")
            current_length += chunk_length
        
        return "\n\n---\n\n".join(context_parts)
    
    # =========================================================================
    # Conversation Memory
    # =========================================================================
    
    def add_conversation(self, user_message: str, ai_response: str,
                         session_id: str = "default") -> str:
        """Add a conversation to memory.
        
        Args:
            user_message: User's message.
            ai_response: AI's response.
            session_id: Session identifier.
            
        Returns:
            Document ID.
        """
        content = f"User: {user_message}\n\nAssistant: {ai_response}"
        
        metadata = {
            'type': 'conversation',
            'session_id': session_id,
            'timestamp': datetime.now().isoformat()
        }
        
        doc_id = f"conv_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return self.add_document(content, doc_id=doc_id, metadata=metadata)
    
    def get_relevant_memories(self, query: str, 
                              session_id: Optional[str] = None,
                              top_k: int = 3) -> List[SearchResult]:
        """Get relevant conversation memories.
        
        Args:
            query: Query to search for.
            session_id: Filter by session.
            top_k: Number of results.
            
        Returns:
            Relevant memories.
        """
        results = self.search(query, top_k=top_k * 2)
        
        # Filter to conversations
        memories = [
            r for r in results 
            if r.metadata.get('type') == 'conversation'
        ]
        
        # Filter by session if specified
        if session_id:
            memories = [
                m for m in memories 
                if m.metadata.get('session_id') == session_id
            ]
        
        return memories[:top_k]
    
    # =========================================================================
    # Utility
    # =========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics.
        
        Returns:
            Statistics dictionary.
        """
        total_chunks = sum(len(doc.chunks) for doc in self.documents.values())
        total_chars = sum(len(doc.content) for doc in self.documents.values())
        
        return {
            "documents": len(self.documents),
            "chunks": total_chunks,
            "total_characters": total_chars,
            "embeddings_cached": len(self.embeddings_cache),
            "model": self.model_name if self.model else "None (keyword search only)",
            "index_path": self.index_file
        }
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents.
        
        Returns:
            List of document info.
        """
        return [
            {
                "doc_id": doc_id,
                "chunks": len(doc.chunks),
                "characters": len(doc.content),
                **doc.metadata
            }
            for doc_id, doc in self.documents.items()
        ]
    
    def format_stats(self) -> str:
        """Format stats for display.
        
        Returns:
            Formatted string.
        """
        stats = self.get_stats()
        
        lines = [
            "📚 **Knowledge Base Stats:**",
            f"  📄 Documents: {stats['documents']}",
            f"  📑 Chunks: {stats['chunks']}",
            f"  📊 Characters: {stats['total_characters']:,}",
            f"  🧠 Embeddings: {stats['embeddings_cached']}",
            f"  🤖 Model: {stats['model']}",
        ]
        
        return "\n".join(lines)
    
    def clear(self):
        """Clear all documents."""
        self.documents = {}
        self.embeddings_cache = {}
        self._save_index()
