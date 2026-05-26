from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from typing import Iterable, List, Optional, Sequence, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings


EXAMPLES_DIR = Path(__file__).parent / "data" / "examples"
CACHE_DIR = Path(".cache/faiss_index")

SUPPORTED_SUFFIXES = {".md", ".txt", ".json"}

# Chunk size / overlap for splitting long example articles before embedding.
# Smaller chunks → more precise retrieval; 800 tokens is a good balance.
_CHUNK_SIZE = 800
_CHUNK_OVERLAP = 100


def _source_fingerprint(directories: List[Path]) -> str:
    """Compute an MD5 hash of all source file mtimes + sizes to detect changes."""
    parts = []
    for d in directories:
        for path in sorted(d.rglob("*")):
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
                stat = path.stat()
                parts.append(f"{path}:{stat.st_mtime}:{stat.st_size}")
    return hashlib.md5("\n".join(parts).encode()).hexdigest()


@dataclass
class RAGPipeline:
    """Simple RAG pipeline using local examples for grounding content.

    Maintains separate vectorstores for text (writing/editing) and images (image design).
    Indexes are persisted to .cache/faiss_index/ and only rebuilt when source files change.
    """

    embeddings: Embeddings
    text_vectorstore: FAISS
    image_vectorstore: FAISS

    @classmethod
    def from_examples(cls, examples_dir: Optional[Path] = None) -> "RAGPipeline":
        directory = examples_dir or EXAMPLES_DIR
        text_dir = directory / "text"
        image_dir = directory / "images"

        text_dir.mkdir(parents=True, exist_ok=True)
        image_dir.mkdir(parents=True, exist_ok=True)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        embeddings = OpenAIEmbeddings()
        fingerprint = _source_fingerprint([text_dir, image_dir])
        fingerprint_file = CACHE_DIR / "fingerprint.txt"
        text_index_path = CACHE_DIR / "text"
        image_index_path = CACHE_DIR / "image"

        cached_fingerprint = (
            fingerprint_file.read_text(encoding="utf-8").strip()
            if fingerprint_file.exists()
            else None
        )

        if (
            cached_fingerprint == fingerprint
            and text_index_path.exists()
            and image_index_path.exists()
        ):
            print("[RAG] loading persisted FAISS indexes from cache...")
            text_vectorstore = FAISS.load_local(
                str(text_index_path), embeddings, allow_dangerous_deserialization=True
            )
            image_vectorstore = FAISS.load_local(
                str(image_index_path), embeddings, allow_dangerous_deserialization=True
            )
        else:
            print("[RAG] source files changed (or first run) — rebuilding FAISS indexes...")
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=_CHUNK_SIZE,
                chunk_overlap=_CHUNK_OVERLAP,
                length_function=len,
            )

            raw_text_docs = _load_documents([text_dir])
            raw_image_docs = _load_documents([image_dir])

            text_chunks = splitter.split_documents(raw_text_docs) if raw_text_docs else []
            image_chunks = splitter.split_documents(raw_image_docs) if raw_image_docs else []

            text_vectorstore = (
                FAISS.from_documents(text_chunks, embeddings)
                if text_chunks
                else FAISS.from_texts([""], embeddings)
            )
            image_vectorstore = (
                FAISS.from_documents(image_chunks, embeddings)
                if image_chunks
                else FAISS.from_texts([""], embeddings)
            )

            text_vectorstore.save_local(str(text_index_path))
            image_vectorstore.save_local(str(image_index_path))
            fingerprint_file.write_text(fingerprint, encoding="utf-8")
            print(
                f"[RAG] indexes saved — {len(text_chunks)} text chunks, "
                f"{len(image_chunks)} image chunks"
            )

        return cls(
            embeddings=embeddings,
            text_vectorstore=text_vectorstore,
            image_vectorstore=image_vectorstore,
        )

    def retrieve_text(self, query: str, k: int = 4) -> List[Document]:
        """Retrieve text examples for writing and editing."""
        return self.text_vectorstore.similarity_search(query, k=k)

    def retrieve_images(
        self,
        query: str,
        k: int = 4,
        avoid_patterns: Optional[Sequence[str]] = None,
        fetch_k: int = 12,
    ) -> List[Document]:
        """Retrieve image examples for visual design with optional anti-repetition reranking."""
        patterns = [
            p.strip().lower()
            for p in (avoid_patterns or [])
            if isinstance(p, str) and p.strip()
        ]
        if not patterns:
            return self.image_vectorstore.similarity_search(query, k=k)

        candidates: List[Tuple[Document, float]] = self.image_vectorstore.similarity_search_with_score(
            query,
            k=max(k, fetch_k),
        )
        if not candidates:
            return []

        def diversity_penalty(doc: Document) -> int:
            text = (doc.page_content or "").lower()
            return sum(1 for pattern in patterns if pattern in text)

        ranked = sorted(candidates, key=lambda pair: (diversity_penalty(pair[0]), pair[1]))
        return [doc for doc, _score in ranked[:k]]

    def as_text_context(self, query: str, k: int = 4) -> str:
        """Get text examples as context string for writing/editing agents."""
        results = self.retrieve_text(query, k=k)
        if not results:
            return ""
        return "\n\n".join(doc.page_content for doc in results)

    def as_image_context(
        self,
        query: str,
        k: int = 4,
        avoid_patterns: Optional[Sequence[str]] = None,
    ) -> str:
        """Get image examples as context string for image agent."""
        results = self.retrieve_images(query, k=k, avoid_patterns=avoid_patterns)
        if not results:
            return ""
        return "\n\n".join(doc.page_content for doc in results)

    def as_image_context_with_sources(
        self,
        query: str,
        k: int = 4,
        avoid_patterns: Optional[Sequence[str]] = None,
    ) -> Tuple[str, List[str]]:
        """Return image context and contributing source paths in a single retrieval call."""
        results = self.retrieve_images(query, k=k, avoid_patterns=avoid_patterns)
        if not results:
            return "", []
        context = "\n\n".join(doc.page_content for doc in results)
        sources = [
            str(doc.metadata.get("source", ""))
            for doc in results
            if isinstance(doc.metadata, dict) and doc.metadata.get("source")
        ]
        return context, sources

def _load_documents(directories: Iterable[Path]) -> List[Document]:
    documents: List[Document] = []
    for directory in directories:
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            text = _read_text(path)
            if not text:
                continue
            documents.append(Document(page_content=text, metadata={"source": str(path)}))
    return documents


def _read_text(path: Path) -> str:
    if path.suffix.lower() in {".md", ".txt"}:
        return path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return ""
    return ""

