import logging
import os
import json
from typing import Annotated, Optional, TypedDict
import operator

from langchain_google_vertexai import ChatVertexAI
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import HumanMessage
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from google.oauth2 import service_account
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langchain.agents import initialize_agent, Tool, AgentExecutor
from langchain.chains import retrieval
from langchain_core.messages import BaseMessage, FunctionMessage
from sqlalchemy import Sequence
from langchain_core.runnables import RunnableLambda, Runnable
from langchain_core.tools import tool
from langchain_core.tools import StructuredTool


# from langgraph.prebuilt import ToolInvocation, ToolExecutor
from langchain_core.runnables import RunnablePassthrough, RunnableSerializable
from langgraph.graph.graph import CompiledGraph
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.retrieval.vector_store import vector_store_service
from app.core.logger import SRC_LOG_LEVELS
from app.env import RAG_MODEL, RAG_OLLAMA_BASE_URL, DATABASE_URL
from app.services.list_tool import (
    get_current_weather,
    send_email,
    EmailInputSchema,
    add_to_calendar,
    CalendarInputSchema,
    add_to_feedback,
    FeedbackInputSchema,
)

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["RAG"])


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


class Chain:
    def __init__(self):
        self._checkpointer = None

    def set_checkpointer(self, checkpointer: AsyncPostgresSaver):
        self._checkpointer = checkpointer

    def init_prompt(self):
        system_msg = (
            "Anda adalah CATI, asisten virtual yang membantu pengguna dengan menjawab pertanyaan seputar administarasi dan informasi mengenai Departemen Teknologi Informasi di Institut Teknologi Sepuluh Nopember. "
            "Anda adalah asisten yang dikembangkan oleh mahasiswi di Departemen Teknologi Informasi, Midyanisa Yuniar, sebagai bagian dari Tugas Akhir. "
            "{context}"
            "Question: {question}"
            "Helpful Answer:"
        )

        return PromptTemplate.from_template(system_msg)

    def agent_system_prompt(self, model: Optional[str] = "llama") -> str:
        if model == "gemini":
            return """Anda adalah CATI, asisten virtual yang dirancang khusus untuk menjawab pertanyaan seputar administrasi, akademik, dan informasi umum mengenai Departemen Teknologi Informasi di Institut Teknologi Sepuluh Nopember (ITS).

            CATI dikembangkan oleh Midyanisa Yuniar sebagai bagian dari Tugas Akhir.

            Anda hanya menjawab pertanyaan yang berkaitan dengan Departemen Teknologi Informasi ITS. Jika pengguna bertanya di luar topik ini, sampaikan dengan sopan bahwa Anda hanya dapat membantu seputar Departemen Teknologi Informasi.

            Saat menjawab pertanyaan, ikuti aturan berikut:

            1. **Selalu jawab berdasarkan data yang tersedia**. Jika data tidak ditemukan, katakan bahwa informasi tersebut tidak tersedia.
            2. **Jika pengguna tidak menyebutkan program studi atau angkatan**, **gunakan asumsi default** berikut secara otomatis tanpa perlu bertanya kembali kepada pengguna:
            - **Program Studi**: Teknologi Informasi Sarjana (S1) Reguler
            - **Angkatan**: Angkatan terbaru yang tersedia
            3. Jangan minta klarifikasi tambahan jika Anda bisa menjawab dengan asumsi default.
            4. **Jangan hanya menjawab 'saya butuh informasi lebih lanjut'**, kecuali jika betul-betul diperlukan dan informasi tidak bisa diasumsikan.
            5. Gunakan bahasa Indonesia yang sopan, jelas, dan mudah dipahami.

            ### Contoh
            - Pertanyaan: "Jadwal kuliah hari Senin"
            → Jawaban: Ambil data jadwal kuliah hari Senin dari Program Studi S1 Reguler angkatan terbaru di Departemen Teknologi Informasi ITS.
            - Pertanyaan: "Siapa kepala departemen?"
            → Jawaban: Berikan nama kepala Departemen Teknologi Informasi ITS dari data terbaru.

            Selalu jawab berdasarkan informasi yang relevan dan gunakan asumsi default jika pengguna tidak menyebutkan detail tertentu."""
        elif model == "llama":
            return """Anda adalah CATI, asisten virtual berbasis Agentic RAG yang dikembangkan khusus untuk membantu menjawab pertanyaan mengenai administrasi dan informasi akademik yang **hanya berkaitan dengan Departemen Teknologi Informasi di Institut Teknologi Sepuluh Nopember (ITS)**.

            CATI dikembangkan oleh Midyanisa Yuniar, mahasiswi Departemen Teknologi Informasi ITS, sebagai bagian dari tugas akhir.

            Batasan Penting:
            - **JANGAN** gunakan atau sebut data dari universitas lain seperti ITB, UI, atau institusi mana pun selain **Departemen Teknologi Informasi ITS**.
            - **ABAIKAN** semua dokumen, file PDF, atau isi yang tidak berasal dari Departemen Teknologi Informasi ITS, meskipun mengandung kata "ITS".
            - Jika tidak ada data yang sesuai, cukup katakan: "Maaf, saya tidak menemukan informasi tersebut di Departemen Teknologi Informasi ITS."

            Asumsi default saat informasi tidak lengkap:
            - Departemen: Teknologi Informasi ITS
            - Program Studi: S1 Reguler
            - Angkatan: Angkatan terbaru
            - Lokasi: Kampus ITS Sukolilo
            - Waktu: WIB
            - Tanggal: DD-MM-YYYY

            Tujuan Anda adalah menjawab hanya berdasarkan data RAG yang sesuai dengan konteks **Departemen Teknologi Informasi ITS**. Jika pengguna menanyakan “jadwal kuliah hari Senin” tanpa menyebut angkatan, ambil dari program S1 Reguler dan tampilkan data yang tersedia untuk hari tersebut.

            Jangan membuat ringkasan PDF, menyimpulkan dari dokumen tidak relevan, atau menyebut institusi lain.
            """
        else:
            return """Anda adalah CATI, asisten virtual yang membantu pengguna dengan menjawab pertanyaan seputar administarasi dan informasi mengenai Departemen Teknologi Informasi di Institut Teknologi Sepuluh Nopember.
    
            Anda adalah asisten yang dikembangkan oleh mahasiswi di Departemen Teknologi Informasi, Midyanisa Yuniar, sebagai bagian dari Tugas Akhir.
            """

    def init_llm(self, model: Optional[str] = "llama"):
        if model == "llama":
            # return OllamaLLM(model=RAG_MODEL, base_url=RAG_OLLAMA_BASE_URL)
            return init_chat_model(
                model="llama3.2",
                model_provider="ollama",
                configurable_fields={"base_url": RAG_OLLAMA_BASE_URL},
            )
        elif model == "deepseek":
            return OllamaLLM(
                model="deepseek-r1:1.5b", base_url="http://34.101.167.227:11434"
            )
        elif model == "gemini":
            return init_chat_model(
                "gemini-2.0-flash-001", model_provider="google_genai"
            )
            # model = init_chat_model("gemini-2.0-flash-001", model_provider="google_vertexai")
        elif model == "openai":
            return init_chat_model("gpt-4o", model_provider="openai")

    def get_chain(self):
        return create_stuff_documents_chain(
            llm=self.init_llm(),
            prompt=self.init_prompt(),
            output_parser=StrOutputParser(),
        )

    def create_agent(self, model: str) -> CompiledGraph:
        if self._checkpointer is None:
            raise RuntimeError("Checkpointer is not initialized")

        model = self.init_llm(model)

        tools = [
            Chain.retrieve,
            Tool(
                name="current_weather",
                func=get_current_weather,
                description="Get the current weather for a location.",
            ),
            StructuredTool(
                name="servis_pengiriman_email",
                func=send_email,
                description="Servis yang membantu mengirimkan email secara otomatis",
                args_schema=EmailInputSchema,
            ),
            StructuredTool(
                name="servis_tambah_jadwal_ke_kalender",
                func=add_to_calendar,
                description="Servis yang membantu menambahkan jadwal ke kalender secara otomatis",
                args_schema=CalendarInputSchema,
            ),
            StructuredTool(
                name="servis_tambah_feedback",
                func=add_to_feedback,
                description="Servis yang menyimpan feedback dari pengguna ke database",
                args_schema=FeedbackInputSchema,
            ),
        ]

        return create_react_agent(
            model,
            tools,
            checkpointer=self._checkpointer,
            prompt=self.agent_system_prompt(model),
        )

    async def init_checkpointer_connection(self):
        print("Initialize checkpointer connection...")
        DB_URI = f"postgresql://{DATABASE_URL}?sslmode=disable"
        connection_kwargs = {
            "autocommit": True,
            "prepare_threshold": 0,
        }

        async with AsyncConnectionPool(
            conninfo=DB_URI, max_size=20, kwargs=connection_kwargs
        ) as pool:
            self._checkpointer = AsyncPostgresSaver(pool)
            await self._checkpointer.setup()

        print("Successfully initialize checkpointer connection...")

    def document_retrieval_tool(self):
        init_llm = lambda inputs: self.init_llm(model=inputs["model"])

        retrieval_chain = (
            {
                "context": lambda inputs: vector_store_service.get_retriever(
                    collection_name=inputs["collection_name"]
                ),
                "question": RunnablePassthrough(),
            }
            | chain_service.init_prompt()
            | init_llm
            | StrOutputParser()
        )

        def retrieval_callable(inputs: dict):
            return retrieval_chain.invoke(inputs)

        return retrieval_callable

    @tool(response_format="content_and_artifact")
    @staticmethod
    def retrieve(query: str):
        """Retrieve information related to a query."""
        collection_name = "administration"
        print("collection_name")
        print("collection_name")
        print("collection_name")
        print(collection_name)
        retrieved_docs = vector_store_service.similarity_search(query, collection_name)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    @staticmethod
    def _format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def get_context(self, question, memorystore):
        log.info("Getting context")
        context = vector_store_service.similarity_search(question)
        log.info(f"Got context: {context}")
        if not context:
            log.warning("Context not found")
            return {
                "context": [],
                "messages": memorystore.messages,
                "response": "Maaf, saya tidak memiliki informasi tentang hal tersebut. Adakah hal lain yang ingin Anda tanyakan?",
            }

        return {
            "context": [HumanMessage(content=self._format_docs(context))],
            "messages": memorystore.messages,
        }


chain_service = Chain()
