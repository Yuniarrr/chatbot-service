import logging
import os
import json
from typing import Annotated, List, Optional, TypedDict
import operator
import asyncio
import joblib

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
from langchain_core.messages import BaseMessage, FunctionMessage, SystemMessage
from sqlalchemy import Sequence
from langchain_core.runnables import RunnableLambda, Runnable
from langchain_core.tools import tool
from langchain_core.tools import StructuredTool
from langchain_community.document_transformers import LongContextReorder
from langchain_core.runnables import RunnablePassthrough, RunnableSerializable
from langgraph.graph.graph import CompiledGraph
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

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
    get_opportunity,
    OpportunityInputSchema,
    AddNewOpportunityInputSchema,
    add_new_opportunity,
    QueryInput,
    preview_email,
    ask_consent_tool,
    AskConsent,
    CollectionSelectorInput,
    select_collection,
)
from app.services.collection import collection_service
from app.retrieval.embed import embedding_service

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["RAG"])


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


class Chain:
    collections_status = []
    model = None

    def __init__(self):
        self._checkpointer = None
        self.max_messages = 3

    @classmethod
    async def init_collection_status(cls):
        print("Init collection status")
        collections_list = await collection_service.get_active_collections()
        collections = collections_list.get("data", [])
        for c in collections:
            cls.collections_status.append(
                {
                    "name": c["name"],
                    "is_active": c["is_active"],
                }
            )

    @classmethod
    def is_collection_active(cls, chosen_collection_name: str):
        for collection in cls.collections_status:
            if collection["name"] == chosen_collection_name:
                return collection["is_active"]
        return False

    @classmethod
    def update_collection_status(cls, collection_name: str, is_active: bool):
        for c in cls.collections_status:
            if c["name"] == collection_name:
                c["is_active"] = is_active
                return True
        return False

    @classmethod
    def load_model_collection(cls):
        print("Load model collection")
        model_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "notebooks",
                "predict_classification",
                "best_improved_question_classifier.pkl",
            )
        )
        model_data = joblib.load(model_path)
        pipeline = model_data["pipeline"]
        cls.model = pipeline

    def set_checkpointer(self, checkpointer: AsyncPostgresSaver):
        self._checkpointer = checkpointer
        
    def limit_messages(self, messages: List[BaseMessage], max_count: int = None) -> List[BaseMessage]:
        """
        Batasi pesan yang dikirim ke agent, hanya ambil pesan terakhir
        """
        if max_count is None:
            max_count = self.max_messages
            
        if len(messages) <= max_count:
            return messages
            
        # Pisahkan system messages dan messages lainnya
        system_messages = [msg for msg in messages if isinstance(msg, SystemMessage)]
        other_messages = [msg for msg in messages if not isinstance(msg, SystemMessage)]
        
        # Ambil pesan terakhir sesuai limit
        recent_messages = other_messages[-max_count:]
        
        # Gabungkan system messages dengan recent messages
        return system_messages + recent_messages        

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
            
            Setiap kali selesai membantu user atau pengguna dalam bentuk jawaban, selalu tanyakan mengenai adakah feedback atau saran yang dapat diberikan mengenai chatbot berbasis Agentic RAG yang sedang dikembangkan. Hal ini berguna untuk mengumpulkan data seberapa bermanfaat chatbot yang sedang dikembangkan. Gunakan tool servis_simpan_feedback untuk menyimpan saran atau feedback dari pengguna.

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
            
            Setiap kali selesai membantu user atau pengguna dalam bentuk jawaban, selalu tanyakan mengenai adakah feedback atau saran yang dapat diberikan mengenai chatbot berbasis Agentic RAG yang sedang dikembangkan. Hal ini berguna untuk mengumpulkan data seberapa bermanfaat chatbot yang sedang dikembangkan. Gunakan tool servis_simpan_feedback untuk menyimpan saran atau feedback dari pengguna.

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
            return """Anda adalah CATI, asisten virtual untuk Departemen Teknologi Informasi ITS. Anda adalah asisten yang dikembangkan oleh mahasiswi di Departemen Teknologi Informasi, Midyanisa Yuniar, sebagai bagian dari Tugas Akhir. Anda akan menjawab pertanyaan berdasarkan dokumen yang tersedia, dan dapat menggunakan tool berikut jika diperlukan:

            - `retrieve`: mengambil informasi dari RAG.
            - `preview_email_template`: menampilkan isi email sebelum dikirim untuk dikonfirmasi pengguna.
            - `servis_pengiriman_email`: mengirimkan email.
            - `servis_tambah_jadwal_ke_kalender`: menambahkan jadwal ke kalender.
            - `servis_simpan_feedback`: menyimpan saran atau kritik pengguna.
            - `servis_get_opportunity`: mengambil data acara atau kegiatan.
            - `servis_add_opportunity`: menyimpan data acara atau kegiatan.
            
            Jika pengguna tampak bingung atau tidak tahu harus mulai dari mana, bantu arahkan dengan memberikan beberapa pilihan atau contoh topik yang bisa ditanyakan, seperti:
            - Jadwal perkuliahan
            - Informasi dosen
            - Kegiatan ARA
            - Menambahkan acara ke kalender
            
            Jika dalam proses pencarian data melalui RAG tidak ditemukan informasi yang relevan, informasikan kepada pengguna bahwa mereka dapat menghubungi Tata Usaha Departemen Teknologi Informasi ITS untuk bantuan lebih lanjut.
            
            Selalu lakukan validasi sebelum menyimpan data ke sistem. Misalnya, jika pengguna ingin menambahkan acara atau kegiatan, pastikan data yang diberikan benar. Jika data yang diberikan tidak sesuai, maka jangan simpan data tersebut.

            Jika pengguna meminta pengiriman email:
            1. Tampilkan dulu isi email menggunakan tool `preview_email_template`.
            2. Tanyakan: "Apakah Anda ingin mengirim email ini?".
            3. Jika pengguna menyetujui (misalnya menjawab "Ya, kirim"), **barulah** gunakan tool `servis_pengiriman_email`.

            Contoh:
            Pengguna: Saya ingin menambahkan acara seminar.
            Asisten: Apakah Anda ingin menyimpan informasi ini ke sistem? Saya butuh izin Anda.
            Pengguna: Iya, silakan simpan.
            Asisten: Baik, tolong berikan detail acara seperti nama, tanggal, dan deskripsi.

            Setelah menyelesaikan setiap interaksi:
            - Selalu tanyakan apakah pengguna memiliki saran atau feedback untuk chatbot ini.
            - Feedback ini penting untuk pengembangan chatbot agentic berbasis RAG ke depannya.
            """

    def init_llm(self, model: Optional[str] = "llama"):
        if model == "llama":
            return init_chat_model(
                model="llama3.2",
                model_provider="ollama",
                configurable_fields={"base_url": RAG_OLLAMA_BASE_URL},
            )
        elif model == "deepseek":
            return init_chat_model(
                model="deepseek-r1:8b",
                model_provider="ollama",
                configurable_fields={"base_url": RAG_OLLAMA_BASE_URL},
            )
        elif model == "gemini":
            return init_chat_model(
                "gemini-2.0-flash-001", model_provider="google_genai"
            )
        elif model == "openai":
            return init_chat_model("gpt-4o", model_provider="openai")
        elif model == "qwen":
            return init_chat_model(
                "qwen3:8b",
                model_provider="ollama",
                configurable_fields={"base_url": RAG_OLLAMA_BASE_URL},
            )

    def get_chain(self):
        return create_stuff_documents_chain(
            llm=self.init_llm(),
            prompt=self.init_prompt(),
            output_parser=StrOutputParser(),
        )

    def create_agent(
        self, model: str, custom_prompt: Optional[str] = None
    ) -> CompiledGraph:
        if self._checkpointer is None:
            raise RuntimeError("Checkpointer is not initialized")

        model = self.init_llm(model)

        tools = [
            StructuredTool(
                name="retrieve",
                description="Retrieve documents based on query.",
                args_schema=QueryInput,
                coroutine=Chain.retrieve,
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
                name="servis_simpan_feedback",
                description="Servis yang menyimpan feedback, saran, dan kritik dari pengguna ke database",
                args_schema=FeedbackInputSchema,
                coroutine=add_to_feedback,
            ),
            StructuredTool(
                name="servis_get_opportunity",
                description="Servis yang akan memberikan data terkait program seperti beasiswa, magang, lomba, internship, dan seminar",
                args_schema=OpportunityInputSchema,
                coroutine=get_opportunity,
            ),
            StructuredTool(
                name="servis_add_opportunity",
                description="Servis yang akan menambahkan dan menyimpan data seperti beasiswa, magang, lomba, internship, dan seminar"
                "Pastikan jika mengunggah file atau gambar, isikan field image_url dengan URL dari file yang sudah diberikan "
                "pada SystemMessage (contoh: https://example.com/image.jpg).",
                args_schema=AddNewOpportunityInputSchema,
                coroutine=add_new_opportunity,
            ),
            StructuredTool(
                name="preview_email_template",
                func=preview_email,
                description="Menampilkan template email untuk dikonfirmasi oleh user sebelum dikirim.",
                args_schema=EmailInputSchema,
            ),
            # StructuredTool(
            #     name="ask_consent_tool",
            #     func=ask_consent_tool,
            #     description="Gunakan ini untuk meminta persetujuan pengguna sebelum menyimpan data.",
            #     args_schema=AskConsent,
            # ),
        ]

        if custom_prompt is not None:
            prompt = custom_prompt
        else:
            prompt = self.agent_system_prompt(model)

        return create_react_agent(
            model,
            tools,
            checkpointer=self._checkpointer,
            prompt=prompt,
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

    @staticmethod
    async def retrieve(query: str):
        """Retrieve information from all active collections."""
        try:
            print(f"Retrieving information for query: {query}")
            agent_response = Chain.model.predict([query])[0]
            print("agent_response")
            print(agent_response)
            chosen_collection_name = agent_response

            print(f"Chosen collection: {chosen_collection_name}")

            if not Chain.is_collection_active(chosen_collection_name):
                print(f"Collection {chosen_collection_name} is not active.")
                return "Tidak dapat menemukan dokumen yang relevan", []

            docs = []
            # Hybrid Retriever with LLM-based Contextual Reranking
            hybrid_retriever = await vector_store_service.get_hybrid_retriever(
                collection_name=chosen_collection_name,
            )
            docs = await hybrid_retriever.ainvoke(query)

            # self query
            # retriever = vector_store_service.get_self_query_retriever(
            #     collection_name=chosen_collection_name
            # )
            # docs = await retriever.ainvoke(query)

            # semantic similarity
            # docs = await vector_store_service.async_similarity_search(
            #     query, collection_name=chosen_collection_name
            # )

            # Hybrid Retrieval with CrossEncoder Reranking
            # docs = await vector_store_service.retrieve_with_rerank(
            #     query=query, collection_name=chosen_collection_name
            # )

            print(f"Jumlah dokumen yang dikembalikan: {len(docs)}")

            # 3. Jika masih 0, fallback ke full semantic similarity search tanpa filter
            if len(docs) == 0:
                # return "Tidak ditemukan dokumen relevan.", []
                print("Fallback: menggunakan full-text search (tanpa filter metadata)")
                docs = await vector_store_service.async_similarity_search(
                    query, collection_name=chosen_collection_name
                )
                print(f"Jumlah dokumen yang dikembalikan setelah fallback: {len(docs)}")

            # for doc in docs:
            #     print(f"Document file name: {doc.metadata.get('file_name', 'unknown')}")

            serialized = "\n\n".join(
                f"Source: {doc.metadata}\nContent: {doc.page_content}" for doc in docs
            )

            return serialized, docs
        except Exception as e:
            return f"Error during retrieval: {e}", []

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
