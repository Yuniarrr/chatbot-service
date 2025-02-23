import logging

from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate

from app.retrieval.vector import Vector

log = logging.getLogger(__name__)
log.setLevel("RAG")


class Chain:
    def __init__(self, vector) -> None:
        self._vector: Vector = vector

    def init_prompt(self):
        # Prompt engineering
        system_msg = (
            "Anda adalah KADIN AI, chatbot resmi dari Kamar Dagang Indonesia (KADIN). Tugas Anda adalah memberikan informasi tentang KADIN Indonesia kepada pengguna dalam bahasa Indonesia.\n\n"
            "**Instruksi Utama:**\n\n"
            "1. **Menjawab Pertanyaan Umum:**\n"
            "   - Berikan informasi yang relevan tentang KADIN Indonesia berdasarkan data yang tersedia.\n"
            "   - Jika Anda tidak memiliki informasi yang cukup untuk menjawab pertanyaan pengguna, berikan respons berikut: "
            "Maaf, saya tidak memiliki informasi tersebut. Adakah hal lain yang ingin Anda tanyakan?\n\n"
        )

        return PromptTemplate.from_template(system_msg)

    def init_llm(self):
        # Large language model
        return

    def get_chain(self):
        # The chain will automatically update since vectorstore update even with no reinitialization
        return create_stuff_documents_chain(
            llm=self.init_llm(),
            prompt=self.init_prompt(),
            output_parser=StrOutputParser(),
        )

    @staticmethod
    def _format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    async def get_context(self, question, memorystore):
        log.info("Getting context")
        context = await self._vector.similarity_search(question)
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
