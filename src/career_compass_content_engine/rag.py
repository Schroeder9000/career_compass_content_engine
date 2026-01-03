from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from typing import Iterable, List

from langchain.docstore.document import Document

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings


EXAMPLES_DIR = Path(__file__).parent / "data" / "examples"

SUPPORTED_SUFFIXES = {".md", ".txt", ".json"}



@dataclass
class RAGPipeline:
    """Simple RAG pipeline using local examples for grounding content."""

    embeddings: Embeddings
    vectorstore: FAISS

    @classmethod
    def from_examples(cls, examples_dir: Path | None = None) -> "RAGPipeline":
        directory = examples_dir or EXAMPLES_DIR
        text_dir = directory / "text"

        image_dir = directory / "images"

        text_dir.mkdir(parents=True, exist_ok=True)
        image_dir.mkdir(parents=True, exist_ok=True)

        documents = _load_documents([text_dir, image_dir])

        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(documents, embeddings) if documents else FAISS.from_texts([""], embeddings)
        return cls(embeddings=embeddings, vectorstore=vectorstore)

    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        return self.vectorstore.similarity_search(query, k=k)

    def as_context(self, query: str, k: int = 4) -> str:
        results = self.retrieve(query, k=k)
        if not results:
            return ""
        return "\n\n".join(doc.page_content for doc in results)

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

