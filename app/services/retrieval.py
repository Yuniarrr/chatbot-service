import json
import logging
import uuid

from datetime import datetime
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import Optional

from app.models.files import FileStatus, ProcessFileForm, FileUpdateModel
from app.core.logger import SRC_LOG_LEVELS
from app.services.file import file_service
from app.services.uploader import uploader_service
from app.retrieval.loaders import Loader
from app.core.exceptions import RetrievalException
from app.core.constants import ERROR_MESSAGES
from app.env import (
    RAG_EMBEDDING_ENGINE,
    RAG_EMBEDDING_MODEL,
    RAG_EMBEDDING_BATCH_SIZE,
    RAG_OLLAMA_API_KEY,
    RAG_OLLAMA_BASE_URL,
    DEVICE_TYPE,
    RAG_EMBEDDING_MODEL_TRUST_REMOTE_CODE,
)
from app.retrieval.vector import VECTOR_DB_CLIENT
from app.retrieval.utils import get_embedding_function, get_model_path

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["SERVICE"])


class RetrievalService:
    def __init__(self):
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, add_start_index=True
        )

    async def process_file(self, form_data: ProcessFileForm):
        print("In process file service")
        file = await file_service.get_file_by_id(form_data.file_id)

        collection_name = form_data.collection_name

        if collection_name is None:
            collection_name = f"file-{file.id}"

        print("collection_name di bawah form")
        print(collection_name)

        # if form_data.collection_name:
        #     log.info("TODO: collection")
        #     result = VECTOR_DB_CLIENT.query(
        #         collection_name=collection_name, filter={"file_id": file.id}
        #     )

        #     if result is not None and len(result.ids[0]) > 0:
        #         docs = [
        #             Document(
        #                 page_content=result.documents[0][idx],
        #                 metadata=result.metadatas[0][idx],
        #             )
        #             for idx, id in enumerate(result.ids[0])
        #         ]
        #     else:
        #         docs = [
        #             Document(
        #                 page_content=file.data.get("content", ""),
        #                 metadata={
        #                     **file.meta,
        #                     "name": file.file_name,
        #                     "created_by": file.user_id,
        #                     "file_id": file.id,
        #                     "source": file.file_name,
        #                 },
        #             )
        #         ]

        #     text_content = file.data.get("content", "")
        # else:
        file_path = file.file_path

        print("file")
        print(file)

        if file_path:
            print("di filepath")
            print("file:", file)
            print("file_path:", file_path)
            file_path = uploader_service.get_file_from_local(file_path)
            # engine=request.app.state.config.CONTENT_EXTRACTION_ENGINE,
            print("loader")
            loader = Loader(PDF_EXTRACT_IMAGES=True)
            print("docs")
            docs = loader.load(file.file_name, file.meta.get("content_type"), file_path)
            print("docs arr")
            docs = [
                Document(
                    page_content=doc.page_content,
                    metadata={
                        **doc.metadata,
                        "name": file.file_name,
                        "created_by": file.user_id,
                        "file_id": file.id,
                        "source": file.file_name,
                    },
                )
                for doc in docs
            ]
        else:
            print("di else")
            docs = [
                Document(
                    page_content=file.data.get("content", ""),
                    metadata={
                        **file.meta,
                        "name": file.filename,
                        "created_by": file.user_id,
                        "file_id": file.id,
                        "source": file.filename,
                    },
                )
            ]

        print("atas text_content")
        text_content = " ".join([doc.page_content for doc in docs])

        log.debug(f"text_content: {text_content}")

        await file_service.update_file_by_id(
            file.id,
            FileUpdateModel(
                **{
                    "data": {"content": text_content},
                }
            ),
        )

        print("collection_name do atas save docs")
        print(collection_name)

        try:
            result = await self.save_docs_to_vector_db(
                docs=docs,
                collection_name=collection_name,
                metadata={
                    "file_id": file.id,
                    "name": file.file_name,
                    "hash": hash,
                },
                add=(True if form_data.collection_name else False),
            )

            if result:
                await file_service.update_file_by_id(
                    file.id,
                    FileUpdateModel(
                        **{
                            "meta": {"collection_name": collection_name},
                            "status": FileStatus.SUCCESS,
                        }
                    ),
                )

                return {
                    "status": True,
                    "collection_name": collection_name,
                    "filename": file.file_name,
                    "content": text_content,
                }
        except Exception as e:
            print(f"Error process file: {e}")
            log.error(f"Error process file: {e}")
            raise RetrievalException(str(e))

    async def save_docs_to_vector_db(
        self,
        docs,
        collection_name,
        metadata: Optional[dict] = None,
        overwrite: bool = False,
        split: bool = True,
        add: bool = False,
    ):
        print("collection_name in save docs to vector")
        print(collection_name)

        def _get_docs_info(docs: list[Document]) -> str:
            docs_info = set()

            # Trying to select relevant metadata identifying the document.
            for doc in docs:
                metadata = getattr(doc, "metadata", {})
                doc_name = metadata.get("name", "")
                if not doc_name:
                    doc_name = metadata.get("title", "")
                if not doc_name:
                    doc_name = metadata.get("source", "")
                if doc_name:
                    docs_info.add(doc_name)

            return ", ".join(docs_info)

        log.info(
            f"save_docs_to_vector_db: document {_get_docs_info(docs)} {collection_name}"
        )

        docs = self._splitter.split_documents(docs)

        if len(docs) == 0:
            raise ValueError(ERROR_MESSAGES.EMPTY_CONTENT)

        texts = [doc.page_content for doc in docs]

        metadatas = [
            {
                **doc.metadata,
                **(metadata if metadata else {}),
                "embedding_config": json.dumps(
                    {
                        "engine": RAG_EMBEDDING_ENGINE,
                        "model": RAG_EMBEDDING_MODEL,
                    }
                ),
            }
            for doc in docs
        ]

        for metadata in metadatas:
            for key, value in metadata.items():
                if isinstance(value, datetime):
                    metadata[key] = str(value)

        try:
            if VECTOR_DB_CLIENT.has_collection(collection_name=collection_name):
                log.info(f"collection {collection_name} already exists")

                if overwrite:
                    VECTOR_DB_CLIENT.delete_collection(collection_name=collection_name)
                    log.info(f"deleting existing collection {collection_name}")
                elif add is False:
                    log.info(
                        f"collection {collection_name} already exists, overwrite is False and add is False"
                    )
                    return True

            log.info(f"adding to collection {collection_name}")

            embedding_function = get_embedding_function(
                RAG_EMBEDDING_ENGINE,
                RAG_EMBEDDING_MODEL,
                self.get_ef(
                    RAG_EMBEDDING_ENGINE,
                    RAG_EMBEDDING_MODEL,
                ),
                RAG_OLLAMA_BASE_URL,
                RAG_OLLAMA_API_KEY,
                RAG_EMBEDDING_BATCH_SIZE,
            )

            embeddings = embedding_function(
                list(map(lambda x: x.replace("\n", " "), texts))
            )

            items = [
                {
                    "id": str(uuid.uuid4()),
                    "text": text,
                    "vector": embeddings[idx],
                    "metadata": metadatas[idx],
                }
                for idx, text in enumerate(texts)
            ]

            VECTOR_DB_CLIENT.insert(
                collection_name=collection_name,
                items=items,
            )

            return True
        except Exception as e:
            print(f"Error save docs to vector db: {e}")
            log.error(f"Error save docs to vector db: {e}")
            raise RetrievalException(str(e))

    def get_ef(
        self,
        engine: str,
        embedding_model: str,
        auto_update: bool = False,
    ):
        ef = None
        if embedding_model and engine == "":
            from sentence_transformers import SentenceTransformer

            try:
                ef = SentenceTransformer(
                    get_model_path(embedding_model, auto_update),
                    device=DEVICE_TYPE,
                    trust_remote_code=RAG_EMBEDDING_MODEL_TRUST_REMOTE_CODE,
                )
            except Exception as e:
                log.debug(f"Error loading SentenceTransformer: {e}")

        return ef


retrieval_service = RetrievalService()
