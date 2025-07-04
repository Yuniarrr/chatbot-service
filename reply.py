import asyncio
import sys

from contextlib import asynccontextmanager
from fastapi import (
    FastAPI,
    Request,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_core.messages import HumanMessage, AIMessage
from twilio.twiml.messaging_response import MessagingResponse
from fastapi.responses import Response

from app.env import PY_ENV, DATABASE_URL
from app.core.database import session_manager, pgvector_session_manager
from app.core.exceptions import DatabaseException, DuplicateValueException
from app.retrieval.vector_store import vector_store_service
from app.retrieval.chain import chain_service

print(
    rf"""
  ___     _  _______ __  
 / __\   / \|__   __|  |
| |     / _ \  | |  |  |
| |__  / /_\ \ | |  |  |
 \___//_/   \_\|_|  |__|

CHATBOT IT
https://github.com/Yuniarrr/chatbot-service
"""
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await session_manager.initialize()
    vector_store_service.initialize_embedding_model()
    vector_store_service.initialize_pg_vector()
    await pgvector_session_manager.initialize()
    DB_URI = f"postgresql://{DATABASE_URL}?sslmode=disable"
    connection_kwargs = {
        "autocommit": True,
        "prepare_threshold": 0,
    }

    async with AsyncConnectionPool(
        conninfo=DB_URI, max_size=20, kwargs=connection_kwargs
    ) as pool:
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()
        chain_service.set_checkpointer(checkpointer)

        try:
            yield {"pool": pool}
        finally:
            await session_manager.close()
            await pgvector_session_manager.close()


app = FastAPI(
    docs_url=f"/{PY_ENV}/docs" if PY_ENV == "dev" else None,
    openapi_url=f"/{PY_ENV}/openapi.json" if PY_ENV == "dev" else None,
    redoc_url=None,
    root_path=f"/api/v1/{PY_ENV}",
    title="Chatbot Service",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/message")
async def reply(request: Request):
    form_data = await request.form()
    message = form_data.get("Body")
    print(message)

    agent_executor = chain_service.create_agent("gemini")

    response = await agent_executor.ainvoke(
        {
            "messages": [
                HumanMessage(content=message),
            ]
        },
        {"configurable": {"thread_id": "1"}},
    )

    messages = response["messages"]

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

    resp = MessagingResponse()
    resp.message(ai_content)
    print("response")
    print(resp)
    return Response(content=str(resp), media_type="application/xml")


# except Exception as e:
#     raise InternalServerException(str(e))


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "status_code": 422,
            "message": "Validation error",
            "error": exc.errors(),
        },
    )


@app.exception_handler(DatabaseException)
async def database_exception_handler(request, exc: DatabaseException):
    return JSONResponse(status_code=500, content={"message": str(exc.detail)})


@app.exception_handler(DuplicateValueException)
async def duplicate_value_exception_handler(request, exc: DuplicateValueException):
    return JSONResponse(status_code=400, content={"message": str(exc.detail)})


if __name__ == "__main__":
    import uvicorn

    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    uvicorn.run("app.main:app", host="0.0.0.0", port=8080)
