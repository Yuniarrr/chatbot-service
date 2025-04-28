import logging
import os
import sqlalchemy as sa

from typing import List, Optional

from langchain_postgres import PGVector
from langchain_core.documents import Document
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from sqlalchemy.dialects import postgresql

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

    def initialize_embedding_model(self):
        print("Initialize embedding model...")
        self._embedding_model = HuggingFaceEmbeddings(
            model_name=SENTENCE_TRANSFORMERS_HOME
        )
        print("Successfully initialize embedding model")

    def initialize_pg_vector(self, collection_name: Optional[str] = VECTOR_TABLE_NAME):
        log.info("Initializing PGVector")

        if self._embedding_model == None:
            print("Embedding model not initialize")
            return

        self._vector_store = PGVector(
            embeddings=self._embedding_model,
            collection_name=collection_name,
            connection="postgresql+psycopg://" + PGVECTOR_DB_URL,
            use_jsonb=True,
        )

    def add_vectostore(
        self, docs: List[Document], collection_name: Optional[str] = None
    ):
        """
        Add documents to the vector store.
        Args:
            docs: List of documents to embed and store.
        """
        print("Adding documents to vector store")

        if collection_name == self._table_name:
            self.initialize_pg_vector(self._table_name)
        else:
            self.initialize_pg_vector(collection_name)

        return self._vector_store.add_documents(docs)

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

        # return self._vector_store.as_retriever(
        #     search_type="similarity", search_kwargs={"k": self._k}
        # ).invoke(query)
        return self._vector_store.similarity_search(query, k=self._k)

    def sim_search(
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

        return self._vector_store.similarity_search(query=query)

    def get_retriever(self, collection_name: Optional[str] = None):
        log.info("Getting retriever")

        if collection_name == self._table_name:
            self.initialize_pg_vector(self._table_name)
        else:
            self.initialize_pg_vector(collection_name)

        return self._vector_store.as_retriever(search_kwargs={"k": self._k})

    def delete_by_ids(self, ids: List[str], collection_name: Optional[str] = None):
        """Load all local embeddings to the vector store."""
        log.info("Deleting documents from vector store")

        if collection_name == self._table_name:
            self.initialize_pg_vector(self._table_name)
        else:
            self.initialize_pg_vector(collection_name)

        self._vector_store.delete(ids=ids)

    def rec_as_dict(self, rec):
        return dict(zip(Langchain_Embedding.columns.keys(), rec))

    async def get_vector_ids(self, file_id: str):
        print("Get vector id")

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


vector_store_service = VectorStore()
