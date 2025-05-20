import os
import io
import json
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from openai import AsyncOpenAI

from app.retrieval.loaders import Loader


class Embedding:
    def __init__(self) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, add_start_index=True
        )
        self.client = AsyncOpenAI()

    def split_document(self, document: str) -> List[Document]:
        splitted_document = self._splitter.split_documents(document)
        return splitted_document

    def add_addtional_data_to_docs(
        self, docs: List[Document], file_id: str, file_name: str, meta: dict
    ) -> List[Document]:
        for doc in docs:
            doc.metadata["file_id"] = file_id
            doc.metadata["file_name"] = file_name

            for k, v in meta.items():
                doc.metadata[k] = v
        return docs

    def loader(self, filename: str, file_content_type: str, file_path: str):
        loader = Loader(PDF_EXTRACT_IMAGES=True)
        return loader.load(filename, file_content_type, file_path)

    def loader_url(self, url: str):
        loader = Loader()
        return loader.load_url(url)

    def split_text(self, text: str) -> List[str]:
        return self._splitter.split_text(text)

    async def embed_query(self, query: str) -> List[float]:
        """Embed a single query string."""
        response = await self.client.embeddings.create(
            input=query, model="text-embedding-3-small"
        )
        return response.data[0].embedding

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts concurrently."""
        response = await self.client.embeddings.create(
            input=texts, model="text-embedding-3-small"
        )
        return [d.embedding for d in response.data]


embedding_service = Embedding()
