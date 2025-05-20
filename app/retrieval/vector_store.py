import asyncio
import json
import logging
import os
import sqlalchemy as sa
import sentence_transformers
from typing import List, Optional

from langchain_postgres import PGVector
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.asyncio import create_async_engine
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_openai import ChatOpenAI
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.retrievers import BaseRetriever

from app.retrieval.model.colbert import ColBERT
from app.retrieval.rerank import (
    AsyncCrossEncoderReranker,
    AsyncReranker,
    ManualRerankRetriever,
    RerankCompressor,
    Reranker,
)
from app.core.database import pgvector_session_manager
from app.core.exceptions import DatabaseException
from app.env import PGVECTOR_DB_URL, VECTOR_TABLE_NAME, SENTENCE_TRANSFORMERS_HOME
from app.core.logger import SRC_LOG_LEVELS

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["RAG"])

Langchain_Embedding = sa.Table(
    "langchain_pg_embedding",
    sa.MetaData(),
    sa.Column(
        "id",
        sa.String,
        primary_key=True,
        server_default=sa.func.uuid_generate_v4(),
    ),
    sa.Column("collection_id", postgresql.UUID(as_uuid=True), nullable=True),
    sa.Column("embedding", postgresql.ARRAY(sa.Float), nullable=True),
    sa.Column("document", sa.String, nullable=True),
    sa.Column("cmetadata", postgresql.JSONB, nullable=True),
)


class VectorStore:
    def __init__(self):
        self._table_name = VECTOR_TABLE_NAME
        self._embedding_model = None
        self._vector_store = None
        self._k = 8

    def initialize_embedding_model(
        self, sentence_transformers_home: Optional[str] = None
    ):
        if not sentence_transformers_home:
            sentence_transformers_home = SENTENCE_TRANSFORMERS_HOME
        print("Initialize embedding model...")
        print(sentence_transformers_home)
        self._embedding_model = HuggingFaceEmbeddings(
            model_name=sentence_transformers_home
        )
        print("Successfully initialize embedding model")

    def initialize_pg_vector(
        self, collection_name: Optional[str] = VECTOR_TABLE_NAME
    ) -> PGVector:
        log.info("Initializing PGVector")

        if self._embedding_model == None:
            print("Embedding model not initialize")
            return self.initialize_embedding_model()

        engine = create_async_engine("postgresql+psycopg://" + PGVECTOR_DB_URL)

        vector_store = PGVector(
            embeddings=self._embedding_model,
            collection_name=collection_name,
            # connection="postgresql+psycopg://" + PGVECTOR_DB_URL,
            connection=engine,
            use_jsonb=True,
        )

        self._vector_store = vector_store

        return vector_store

    async def add_vectostore(
        self, docs: List[Document], collection_name: Optional[str] = None
    ):
        """
        Add documents to the vector store.
        Args:
            docs: List of documents to embed and store.
        """
        print(f"Add vector store to {collection_name}")

        if self._vector_store is None or collection_name != self._table_name:
            vector_store = self.initialize_pg_vector(collection_name)
        else:
            vector_store = self._vector_store

        print("vector_store:", vector_store)

        return await vector_store.aadd_documents(docs)

    def similarity_search(
        self, query: str, collection_name: Optional[str] = None
    ) -> List[Document]:
        """Searches and returns movies.

        Args:
        query: The user query to search for related items

        Returns:
        List[Document]: A list of Documents
        """
        log.info("Performing similarity search")

        if collection_name == self._table_name:
            self.initialize_pg_vector(self._table_name)
        else:
            self.initialize_pg_vector(collection_name)

        return self._vector_store.similarity_search(query, k=self._k)

    async def async_similarity_search(
        self, query: str, collection_name: Optional[str] = None
    ) -> List[Document]:
        """Searches and returns movies.

        Args:
        query: The user query to search for related items
        Returns:
        List[Document]: A list of Documents
        """
        log.info("Performing similarity search")

        if collection_name == self._table_name:
            # self.initialize_pg_vector(self._table_name)
            vector_store = self.initialize_pg_vector(self._table_name)
        else:
            # self.initialize_pg_vector(collection_name)
            vector_store = self.initialize_pg_vector(collection_name)

        return await vector_store.asimilarity_search(query, k=self._k)

    async def similarity_search_with_score(
        self, query: str, collection_name: Optional[str] = None
    ) -> List[Document]:
        """Searches and returns movies.

        Args:
        query: The user query to search for related items

        Returns:
        List[Document]: A list of Documents
        """
        log.info("Performing similarity search")

        if collection_name == self._table_name:
            # self.initialize_pg_vector(self._table_name)
            vector_store = self._vector_store
        else:
            # self.initialize_pg_vector(collection_name)
            vector_store = self.initialize_pg_vector(collection_name)

            # Chain.retrieve,
        return vector_store.asimilarity_search_with_score(query, k=self._k)

    def get_retriever(self, collection_name: Optional[str] = None):
        log.info("Getting retriever")

        if collection_name == self._table_name:
            self.initialize_pg_vector(self._table_name)
        else:
            self.initialize_pg_vector(collection_name)

        return self._vector_store.as_retriever(search_kwargs={"k": self._k})

    async def delete_by_ids(
        self, ids: List[str], collection_name: Optional[str] = None
    ):
        """Load all local embeddings to the vector store."""
        log.info("Deleting documents from vector store")

        if collection_name == self._table_name:
            # self.initialize_pg_vector(self._table_name)
            vector_store = self._vector_store
        else:
            # self.initialize_pg_vector(collection_name)
            vector_store = self.initialize_pg_vector(collection_name)

        await vector_store.adelete(ids=ids)

    async def update_metadata_by_file_id(self, file_id: str, metadata: dict):
        try:
            sql = sa.text(
                """
                UPDATE langchain_pg_embedding
                SET cmetadata = cmetadata || :cmetadata
                WHERE cmetadata->>'file_id' = :file_id
                """
            )

            async with pgvector_session_manager.session() as db:
                async with db.begin():
                    await db.execute(
                        sql, {"file_id": file_id, "cmetadata": json.dumps(metadata)}
                    )
        except Exception as e:
            log.error(f"Error updating metadata for file_id={file_id}: {e}")
            raise DatabaseException(str(e))

    def rec_as_dict(self, rec):
        return dict(zip(Langchain_Embedding.columns.keys(), rec))

    async def get_vector_ids(self, file_id: str):
        try:
            sql = sa.text(
                """
                SELECT id
                FROM langchain_pg_embedding
                WHERE cmetadata->>'file_id' = :file_id
            """
            )

            async with pgvector_session_manager.session() as db:
                result = await db.execute(sql, {"file_id": file_id})

                return [*map(self.rec_as_dict, result)]
        except Exception as e:
            log.error(f"Error get vector id: {e}")
            raise DatabaseException(str(e))

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def get_self_query_retriever(
        self, collection_name: Optional[str] = None, query: str = None
    ):
        """Returns a SelfQueryRetriever that uses metadata filtering."""

        if collection_name == self._table_name:
            vector_store = self._vector_store
        else:
            vector_store = self.initialize_pg_vector(collection_name)

        metadata_field_info = [
            AttributeInfo(
                name="file_name", description="Nama file dokumen", type="string"
            ),
            AttributeInfo(
                name="document_type",
                description="Tipe dokumen, misal jadwal, kalendar akademik, skem, mata kuliah, kerja praktik, cuti bersama, artikel ilmiah, mbkm, pkm, tugas akhir, pengumuman, dan akreditasi",
                type="string",
            ),
            # AttributeInfo(
            #     name="tahun_ajaran",
            #     description="Tahun ajaran dokumen (jika ada)",
            #     type="string",
            # ),
            AttributeInfo(
                name="topik",
                description="Topik terkait dokumen seperti, akreditasi, penelitian, dan lain lain (jika ada)",
                type="string",
            ),
        ]

        llm = ChatOpenAI(temperature=0, model_name="gpt-4o")

        retriever = SelfQueryRetriever.from_llm(
            llm=llm,
            vectorstore=vector_store,
            document_contents="Dokumen berisi informasi akademik seperti jadwal kuliah, mata kuliah, silabus, dan pengumuman. "
            "Dokumen dengan tipe 'jadwal' biasanya berisi detail waktu, kelas, dan tempat pengajaran mata kuliah. "
            "Dokumen tipe 'mata kuliah' lebih ke deskripsi umum mata kuliah tersebut.",
            metadata_field_info=metadata_field_info,
            search_kwargs={"k": self._k},
            verbose=True,
        )

        return retriever

    async def get_hybrid_retriever(
        self,
        collection_name: str,
        # llm: ChatOpenAI,
    ) -> BaseRetriever:
        vector_store = self.initialize_pg_vector(collection_name)
        all_docs = await self.get_all_documents(collection_name=collection_name)

        texts = [doc.page_content for doc in all_docs]
        metadatas = [doc.metadata for doc in all_docs]

        bm25_retriever = BM25Retriever.from_texts(texts=texts, metadatas=metadatas)
        bm25_retriever.k = self._k

        vector_retriever = vector_store.as_retriever(search_kwargs={"k": self._k})

        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever], weights=[0.5, 0.5]
        )

        # reranker = Reranker()
        reranker = sentence_transformers.CrossEncoder(
            "BAAI/bge-reranker-base",
            trust_remote_code=True,
        )

        # reranker = AsyncReranker()
        compressor = RerankCompressor(
            embedding_function=self._embedding_model.embed_query,
            top_n=self._k,
            reranking_function=reranker,
            r_score=0.3,
        )

        return ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=ensemble_retriever
        )

    async def get_all_documents(self, collection_name: str) -> list[Document]:
        try:
            sql = sa.text(
                """
                SELECT e.document, e.cmetadata
                FROM langchain_pg_embedding e
                JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                WHERE c.name = :collection_name
                """
            )

            async with pgvector_session_manager.session() as db:
                result = await db.execute(sql, {"collection_name": collection_name})
                rows = result.fetchall()

                documents = [
                    Document(page_content=row[0], metadata=row[1]) for row in rows
                ]

                return documents

        except Exception as e:
            log.error(f"Error get_all_documents: {e}")
            raise DatabaseException(str(e))

    async def retrieve_with_rerank(
        self,
        query: str,
        collection_name: str,
    ) -> list[Document]:
        vector_store = self.initialize_pg_vector(collection_name)
        all_docs = await self.get_all_documents(collection_name=collection_name)

        print(f"[INFO] Total dokumen dalam koleksi: {len(all_docs)}")

        if not all_docs:
            print("[WARN] Koleksi kosong.")
            return []

        texts = [doc.page_content for doc in all_docs]
        metadatas = [doc.metadata for doc in all_docs]

        bm25_retriever = BM25Retriever.from_texts(texts=texts, metadatas=metadatas)
        bm25_retriever.k = self._k

        vector_retriever = vector_store.as_retriever(search_kwargs={"k": self._k})

        # Combine
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever], weights=[0.5, 0.5]
        )

        # Run initial hybrid retrieval
        initial_docs = await ensemble_retriever.ainvoke(query)
        print(f"[INFO] Jumlah dokumen hasil hybrid retrieval: {len(initial_docs)}")
        if not initial_docs:
            print("[WARN] Tidak ada dokumen ditemukan dari hybrid retriever.")
            return []

        # Rerank
        # reranker = AsyncCrossEncoderReranker(device="cpu")
        # reranked_docs = await reranker.rerank(
        #     query=query, documents=initial_docs, top_n=self._k
        # )

        # return reranked_docs

        reranker = AsyncCrossEncoderReranker(device="cpu")
        reranked_texts = await reranker.rerank(
            query=query,
            documents=[doc.page_content for doc in initial_docs],
            top_n=self._k,
        )

        reranked_docs = [
            doc for doc in initial_docs if doc.page_content in reranked_texts
        ]

        return reranked_docs


vector_store_service = VectorStore()
