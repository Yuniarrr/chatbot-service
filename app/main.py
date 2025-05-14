import asyncio
import os
import sys

from contextlib import asynccontextmanager
from fastapi import (
    FastAPI,
    Request,
)
from mimetypes import guess_type
from os.path import isfile

# from fastapi.middleware.cors import CORSMiddleware

from starlette.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

# from fastapi.staticfiles import StaticFiles
from starlette.staticfiles import StaticFiles

from app.env import PY_ENV, DATABASE_URL, UPLOAD_DIR
from app.routers import auth, user, conversation, chat, opportunity
from app.routers.file import router
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


@app.middleware("http")
async def add_cors_headers(request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    response.headers["Access-Control-Allow-Methods"] = (
        "GET, POST, PUT, PATCH, DELETE, OPTIONS"
    )
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(router, prefix="/file", tags=["file"])
app.include_router(user.router, prefix="/user", tags=["user"])
app.include_router(conversation.router, prefix="/conversation", tags=["conversation"])
app.include_router(chat.router, prefix="/chat", tags=["chat"])
app.include_router(opportunity.router, prefix="/opportunity", tags=["opportunity"])


# @app.get("/uploads/{filename}")
# async def get_file(filename: str, request: Request):
#     file_path = os.path.join(UPLOAD_DIR, filename)
#     print("file_path")
#     print(file_path)
#     # if not os.path.exists(file_path):
#     #     raise HTTPException(status_code=404, detail="File not found")
#     # # return FileResponse(path=file_path, media_type="application/pdf")
#     # response = FileResponse(file_path, media_type="application/pdf")
#     # # Manually add CORS headers
#     # origin = request.headers.get("origin")
#     # if origin:
#     #     response.headers["Access-Control-Allow-Origin"] = origin
#     #     response.headers["Access-Control-Allow-Credentials"] = "true"
#     # return response

#     if not isfile(file_path):
#         return Response(status_code=404)

#     with open(file_path) as f:
#         content = f.read()

#     content_type, _ = guess_type(file_path)
#     return Response(content, media_type=content_type)


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
