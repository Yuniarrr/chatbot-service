from contextlib import asynccontextmanager
from fastapi import (
    FastAPI,
    Request,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.env import PY_ENV
from app.routers import auth, user
from app.routers.file import router
from app.core.database import session_manager
from app.core.exceptions import DatabaseException, DuplicateValueException

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
    try:
        yield
    finally:
        await session_manager.close()


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

app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(router, prefix="/file", tags=["file"])
app.include_router(user.router, prefix="/user", tags=["user"])


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
