from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from langchain.docstore.document import Document
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings


EXAMPLES_DIR = Path(__file__).parent / "data" / "examples"


@dataclass
class RAGPipeline:
    """Simple RAG pipeline using local examples for grounding content."""

    embeddings: Embeddings
    vectorstore: FAISS

    @classmethod
    def from_examples(cls, examples_dir: Path | None = None) -> "RAGPipeline":
        directory = examples_dir or EXAMPLES_DIR
        text_dir = directory / "text"
        if not text_dir.exists():
            text_dir.mkdir(parents=True, exist_ok=True)
        documents = _load_documents(text_dir)
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


def _load_documents(text_dir: Path) -> List[Document]:
    documents: List[Document] = []
    for path in sorted(text_dir.glob("*.md")):
        loader = TextLoader(path)
        documents.extend(loader.load())
    return documents
