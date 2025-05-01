from bottle import route, run, post, request, error, Bottle
from twilio.twiml.messaging_response import MessagingResponse
from langchain_core.runnables import RunnablePassthrough, RunnableSerializable
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from app.retrieval.vector_store import vector_store_service
from app.retrieval.chain import chain_service
from app.core.database import session_manager, pgvector_session_manager

app = Bottle()


async def initialize_services():
    await session_manager.initialize()
    vector_store_service.initialize_embedding_model()
    vector_store_service.initialize_pg_vector()
    await pgvector_session_manager.initialize()


async def shutdown_services():
    await session_manager.close()
    await pgvector_session_manager.close()


@app.post("/message")
def reply_agentic_rag():
    msg = request.forms.get("Body")

    print("msg")
    print(msg)
    print("MediaUrl0")
    print(request.forms.get("MediaUrl0"))

    agent_executor = chain_service.create_agent("openai")
    system_prompt = chain_service.agent_system_prompt()

    response = agent_executor.invoke(
        {
            "messages": [
                SystemMessage(content=system_prompt),
                HumanMessage(content=msg),
            ]
        }
    )

    messages = response["messages"]

    print(messages)

    ai_content = next(
        (
            message.content
            for message in messages
            if isinstance(message, AIMessage) and message.content.strip() != ""
        ),
        "No AI response found.",
    )

    print("ai_content")
    print(ai_content)

    twilio_response = MessagingResponse()
    twilio_response.message(ai_content)
    return str(twilio_response)


def reply_rag_chain():
    msg = request.forms.get("Body")

    rag_chain = (
        {
            "context": vector_store_service.get_retriever(
                collection_name="administration"
            ),
            "question": RunnablePassthrough(),
        }
        | chain_service.init_prompt()
        | chain_service.init_llm("openai")
        | StrOutputParser()
    )

    res_chain = rag_chain.invoke(msg)

    print("response")
    print(res_chain)

    twilio_response = MessagingResponse()
    twilio_response.message(res_chain)
    return str(twilio_response)


import asyncio

if __name__ == "__main__":

    async def main():
        try:
            await initialize_services()
            run(app, host="localhost", port=8080, debug=True)
        finally:
            await shutdown_services()

    asyncio.run(main())
