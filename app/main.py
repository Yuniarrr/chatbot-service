from fastapi import (
    FastAPI,
    Request,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.env import PY_ENV

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

app = FastAPI(
    docs_url=f"/{PY_ENV}/docs" if PY_ENV == "dev" else None,
    openapi_url=f"/{PY_ENV}/openapi.json" if PY_ENV == "dev" else None,
    redoc_url=None,
    openapi_prefix=f"/{PY_ENV}",
    title="Chatbot Service",
    # lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# app.include_router(ollama.router, prefix="/ollama", tags=["ollama"])


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
