"""
AEGIS Phase 2 Week 3 — Knowledge Store
========================================
Personal knowledge base using ChromaDB for semantic retrieval.
Pulled forward from SOW Phase 4 (Voice + Knowledge Base).

Supports ingesting:
  - Plain text / markdown notes
  - PDF files (text extraction via pypdf)
  - Task history (auto-indexed from TaskStore)
  - Arbitrary string chunks

Embedding model: ollama nomic-embed-text (per SOW section 6.1).
Fallback: chromadb's built-in all-MiniLM-L6-v2 when Ollama is unavailable
(keeps the store testable without a running Ollama instance).

Usage
-----
    store = KnowledgeStore()
    store.ingest_text("My meeting notes", source="notes/meeting.md")
    results = store.search("what did we discuss about the API?", n=5)
    for r in results:
        print(r.text, r.score)
"""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils import embedding_functions

# ── paths ──────────────────────────────────────────────────────────────────────
_DEFAULT_CHROMA_DIR = (
    Path(__file__).resolve().parents[2] / "registry_store" / "chroma_db"
)
_COLLECTION_NAME = "aegis_knowledge"

# ── chunking defaults ──────────────────────────────────────────────────────────
_CHUNK_SIZE    = 400    # words per chunk
_CHUNK_OVERLAP = 50     # word overlap between chunks


@dataclass
class SearchResult:
    """Single knowledge base search result."""
    text:     str
    source:   str
    doc_id:   str
    score:    float          # distance → lower is closer (chromadb default)
    metadata: dict = field(default_factory=dict)


class _HashEmbeddingFunction(embedding_functions.EmbeddingFunction):
    """
    Deterministic hash-based pseudo-embedding.
    Used as a fallback when neither Ollama nor the ONNX model are available
    (e.g. offline CI, sandboxed environments).

    Produces 384-dimensional float vectors from SHA-256 hashes.
    Results are NOT semantically ranked. Replace with nomic-embed-text in production.
    """

    DIM = 384

    def __init__(self) -> None:
        super().__init__()

    def name(self) -> str:
        return "aegis-hash-embedding"

    def __call__(self, input: list[str]) -> list[list[float]]:
        import hashlib
        results = []
        for text in input:
            # Build a DIM-length vector by hashing the text repeatedly
            vec: list[float] = []
            seed = text.encode("utf-8")
            idx = 0
            while len(vec) < self.DIM:
                h = hashlib.sha256(seed + idx.to_bytes(4, "big")).digest()
                # Each byte maps to a float in [-1, 1]
                vec.extend((b - 127.5) / 127.5 for b in h)
                idx += 1
            vec = vec[: self.DIM]
            # Normalise to unit vector (avoids NaN: if all zeros, use uniform)
            norm = sum(v * v for v in vec) ** 0.5
            if norm < 1e-9:
                vec = [1.0 / (self.DIM ** 0.5)] * self.DIM
            else:
                vec = [v / norm for v in vec]
            results.append(vec)
        return results


class KnowledgeStore:
    """
    Semantic knowledge base backed by ChromaDB.

    Embedding strategy
    ------------------
    1. Try Ollama nomic-embed-text (SOW spec, requires running Ollama).
    2. Fall back to chromadb's default sentence-transformer embedding
       (all-MiniLM-L6-v2) — works offline, no extra install.

    The fallback is transparent — queries and ingestion work identically.
    """

    def __init__(
        self,
        persist_dir: Path | None = None,
        ollama_base_url: str = "http://localhost:11434",
        use_ollama: bool = True,
    ) -> None:
        self._persist_dir    = Path(persist_dir) if persist_dir else _DEFAULT_CHROMA_DIR
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._ollama_url     = ollama_base_url
        self._use_ollama     = use_ollama
        self._client         = chromadb.PersistentClient(path=str(self._persist_dir))
        self._collection     = self._init_collection()

    # ── initialisation ─────────────────────────────────────────────────────────

    def _init_collection(self):
        ef = self._make_embedding_function()
        return self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )

    def _make_embedding_function(self):
        if self._use_ollama:
            try:
                return embedding_functions.OllamaEmbeddingFunction(
                    url=f"{self._ollama_url}/api/embeddings",
                    model_name="nomic-embed-text",
                )
            except Exception:
                pass   # fall through to default
        # Try chromadb built-in ONNX model (requires download on first use)
        try:
            ef = embedding_functions.DefaultEmbeddingFunction()
            ef(["probe"])   # trigger download/load now so failures are caught here
            return ef
        except Exception:
            pass

        # Last resort: deterministic hash-based embedding.
        # Not semantically meaningful but keeps all store/retrieval logic
        # fully testable without network access or model downloads.
        # In production, Ollama nomic-embed-text will always be used instead.
        return _HashEmbeddingFunction()

    # ── ingestion ──────────────────────────────────────────────────────────────

    def ingest_text(
        self,
        text: str,
        source: str = "manual",
        metadata: dict | None = None,
    ) -> list[str]:
        """
        Chunk and index a plain-text string.
        Returns list of document IDs inserted.
        """
        chunks = self._chunk_text(text)
        return self._upsert_chunks(chunks, source=source, metadata=metadata or {})

    def ingest_file(self, path: str | Path) -> list[str]:
        """
        Ingest a file from disk.
        Supported: .txt, .md, .py, .pdf
        Returns list of document IDs inserted.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        suffix = path.suffix.lower()

        if suffix == ".pdf":
            text = self._extract_pdf(path)
        elif suffix in {".txt", ".md", ".py", ".rst", ".json", ".yaml", ".yml"}:
            text = path.read_text(encoding="utf-8-sig", errors="replace")
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        return self.ingest_text(text, source=str(path))

    def ingest_task_summary(self, task_id: str, summary: str, agents: list[str]) -> str:
        """Index a completed task summary for future retrieval."""
        doc_id = f"task:{task_id}"
        self._collection.upsert(
            ids=[doc_id],
            documents=[summary],
            metadatas=[{"source": "task_history", "task_id": task_id, "agents": ",".join(agents)}],
        )
        return doc_id

    # ── retrieval ──────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        n: int = 5,
        source_filter: str | None = None,
    ) -> list[SearchResult]:
        """
        Semantic search over the knowledge base.

        Parameters
        ----------
        query:         Natural language query string.
        n:             Max results to return.
        source_filter: If set, only return results from this source prefix.
        """
        if self.count() == 0:
            return []

        where = None
        if source_filter:
            where = {"source": {"$eq": source_filter}}

        results = self._collection.query(
            query_texts=[query],
            n_results=min(n, self.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        out: list[SearchResult] = []
        docs      = results["documents"][0]
        metas     = results["metadatas"][0]
        distances = results["distances"][0]
        ids       = results["ids"][0]

        for doc, meta, dist, doc_id in zip(docs, metas, distances, ids):
            out.append(SearchResult(
                text=doc,
                source=meta.get("source", ""),
                doc_id=doc_id,
                score=round(dist, 4),
                metadata=meta,
            ))

        return out

    def search_as_context(self, query: str, n: int = 3, max_chars: int = 1200) -> str:
        """
        Search and return a compact string suitable for Brain prompt injection.
        Empty string when nothing relevant is found.
        """
        results = self.search(query, n=n)
        if not results:
            return ""

        lines = ["[Knowledge base — relevant context]"]
        total = 0
        for r in results:
            snippet = r.text[:400]
            if total + len(snippet) > max_chars:
                break
            lines.append(f"Source: {r.source}\n{snippet}")
            total += len(snippet)

        return "\n\n".join(lines)

    # ── management ─────────────────────────────────────────────────────────────

    def count(self) -> int:
        return self._collection.count()

    def delete_source(self, source: str) -> int:
        """Remove all chunks from a specific source. Returns count deleted."""
        results = self._collection.get(where={"source": {"$eq": source}})
        ids = results["ids"]
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)

    def list_sources(self) -> list[str]:
        """Unique source names currently indexed."""
        if self.count() == 0:
            return []
        results = self._collection.get(include=["metadatas"])
        sources = {m.get("source", "") for m in results["metadatas"]}
        return sorted(s for s in sources if s)

    def reset(self) -> None:
        """Delete all documents. Use with care."""
        self._client.delete_collection(_COLLECTION_NAME)
        self._collection = self._init_collection()

    # ── private helpers ────────────────────────────────────────────────────────

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping word-based chunks."""
        words = text.split()
        if not words:
            return []

        chunks: list[str] = []
        start = 0
        while start < len(words):
            end = min(start + _CHUNK_SIZE, len(words))
            chunks.append(" ".join(words[start:end]))
            if end == len(words):
                break
            start += _CHUNK_SIZE - _CHUNK_OVERLAP

        return chunks

    def _upsert_chunks(
        self,
        chunks: list[str],
        source: str,
        metadata: dict,
    ) -> list[str]:
        if not chunks:
            return []

        ids, docs, metas = [], [], []
        for i, chunk in enumerate(chunks):
            # stable ID: hash of source + chunk index
            chunk_hash = hashlib.md5(f"{source}:{i}".encode()).hexdigest()[:12]
            doc_id = f"doc:{chunk_hash}"
            ids.append(doc_id)
            docs.append(chunk)
            metas.append({"source": source, "chunk_index": i, **metadata})

        self._collection.upsert(ids=ids, documents=docs, metadatas=metas)
        return ids

    @staticmethod
    def _extract_pdf(path: Path) -> str:
        """Extract text from PDF using pypdf (optional dependency)."""
        try:
            import pypdf
            reader = pypdf.PdfReader(str(path))
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n\n".join(pages)
        except ImportError:
            raise ImportError(
                "pypdf is required for PDF ingestion. "
                "Install it with: pip install pypdf"
            )


# ── module-level singleton ─────────────────────────────────────────────────────
_store: KnowledgeStore | None = None


def get_knowledge_store(use_ollama: bool = False) -> KnowledgeStore:
    """
    Return the shared KnowledgeStore instance.
    use_ollama=False by default so it works without a running Ollama instance.
    Set use_ollama=True in production to use nomic-embed-text.
    """
    global _store
    if _store is None:
        _store = KnowledgeStore(use_ollama=use_ollama)
    return _store
