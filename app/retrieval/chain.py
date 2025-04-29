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

from app.retrieval.vector_store import vector_store_service
from app.core.logger import SRC_LOG_LEVELS
from app.env import RAG_MODEL, RAG_OLLAMA_BASE_URL
from app.services.list_tool import (
    get_current_weather,
    send_email,
    EmailInputSchema,
    add_to_calendar,
    CalendarInputSchema,
)

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["RAG"])


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


class Chain:
    def init_prompt(self):
        system_msg = (
            "Anda adalah CATI, asisten virtual yang membantu pengguna dengan menjawab pertanyaan seputar administarasi dan informasi mengenai Departemen Teknologi Informasi di Institut Teknologi Sepuluh Nopember. "
            "Anda adalah asisten yang dikembangkan oleh mahasiswi di Departemen Teknologi Informasi, Midyanisa Yuniar, sebagai bagian dari Tugas Akhir. "
            "{context}"
            "Question: {question}"
            "Helpful Answer:"
        )

        return PromptTemplate.from_template(system_msg)

    def agent_system_prompt(self) -> str:
        return """Anda adalah CATI, asisten virtual yang membantu pengguna dengan menjawab pertanyaan seputar administarasi dan informasi mengenai Departemen Teknologi Informasi di Institut Teknologi Sepuluh Nopember.
    
        Anda adalah asisten yang dikembangkan oleh mahasiswi di Departemen Teknologi Informasi, Midyanisa Yuniar, sebagai bagian dari Tugas Akhir.
        """

    def init_llm(self, model: Optional[str] = "ollama"):
        if model == "ollama":
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
            # with open(
            #     os.path.join(os.path.dirname(__file__), "./google.json")
            # ) as source:
            #     info = json.load(source)

            # credentials = service_account.Credentials.from_service_account_info(
            #     info, scopes=["https://www.googleapis.com/auth/cloud-platform"]
            # )
            # return ChatVertexAI(
            #     model_name="gemini-2.0-flash-001",
            #     credentials=credentials,
            # )
        elif model == "openai":
            return init_chat_model("gpt-4o", model_provider="openai")

    def get_chain(self):
        return create_stuff_documents_chain(
            llm=self.init_llm(),
            prompt=self.init_prompt(),
            output_parser=StrOutputParser(),
        )

    def create_agent(self, model: str) -> CompiledGraph:
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
        ]

        return create_react_agent(model, tools)

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
    def retrieve(query: str, collection_name: str = "administration"):
        """Retrieve information related to a query."""
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
