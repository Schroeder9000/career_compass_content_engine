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
    """Simple RAG pipeline using local examples for grounding content.
    
    Maintains separate vectorstores for text (writing/editing) and images (image design).
    """

    embeddings: Embeddings
    text_vectorstore: FAISS
    image_vectorstore: FAISS

    @classmethod
    def from_examples(cls, examples_dir: Path | None = None) -> "RAGPipeline":
        directory = examples_dir or EXAMPLES_DIR
        text_dir = directory / "text"
        image_dir = directory / "images"

        text_dir.mkdir(parents=True, exist_ok=True)
        image_dir.mkdir(parents=True, exist_ok=True)

        # Load text and image documents separately
        text_documents = _load_documents([text_dir])
        image_documents = _load_documents([image_dir])

        embeddings = OpenAIEmbeddings()
        
        # Create separate vectorstores
        text_vectorstore = FAISS.from_documents(text_documents, embeddings) if text_documents else FAISS.from_texts([""], embeddings)
        image_vectorstore = FAISS.from_documents(image_documents, embeddings) if image_documents else FAISS.from_texts([""], embeddings)
        
        return cls(embeddings=embeddings, text_vectorstore=text_vectorstore, image_vectorstore=image_vectorstore)

    def retrieve_text(self, query: str, k: int = 4) -> List[Document]:
        """Retrieve text examples for writing and editing."""
        return self.text_vectorstore.similarity_search(query, k=k)

    def retrieve_images(self, query: str, k: int = 4) -> List[Document]:
        """Retrieve image examples for visual design."""
        return self.image_vectorstore.similarity_search(query, k=k)

    def as_text_context(self, query: str, k: int = 4) -> str:
        """Get text examples as context string for writing/editing agents."""
        results = self.retrieve_text(query, k=k)
        if not results:
            return ""
        return "\n\n".join(doc.page_content for doc in results)

    def as_image_context(self, query: str, k: int = 4) -> str:
        """Get image examples as context string for image agent."""
        results = self.retrieve_images(query, k=k)
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

