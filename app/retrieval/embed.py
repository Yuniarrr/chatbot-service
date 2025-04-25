import os
import io
import json
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer


from app.retrieval.loaders import Loader


class Embedding:
    def __init__(self) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, add_start_index=True
        )

    def split_document(self, document: str) -> List[Document]:
        splitted_document = self._splitter.split_documents(document)
        return splitted_document

    def add_addtional_data_to_docs(
        self, docs: List[Document], file_id: str, file_name: str
    ) -> List[Document]:
        for doc in docs:
            doc.metadata["file_id"] = file_id
            doc.metadata["file_name"] = file_name
        return docs

    def loader(self, filename: str, file_content_type: str, file_path: str):
        loader = Loader(PDF_EXTRACT_IMAGES=True)
        return loader.load(filename, file_content_type, file_path)

    def split_text(self, text: str) -> List[str]:
        return self._splitter.split_text(text)


embedding_service = Embedding()
