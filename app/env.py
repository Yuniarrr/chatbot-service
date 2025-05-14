import os
import getpass

from dotenv import load_dotenv
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
APP_DIR = CURRENT_FILE.parent
PARENT_DIR = APP_DIR.parent


load_dotenv()

####################################
# APP SETTINGS
####################################

# ENV (dev,test,prod)
PY_ENV = os.environ.get("PY_ENV", "dev")
FRONTEND_URL = os.environ.get("FRONTEND_URL", "http://localhost:5173")

####################################
# AUTH SETTINGS
####################################

JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "hrRVHOX7-GqzMrR;].youw0~;L")
JWT_ACCESS_TOKEN_EXPIRE = int(os.environ.get("JWT_ACCESS_TOKEN_EXPIRE", "6"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.environ.get("REFRESH_TOKEN_EXPIRE_DAYS", "7"))
API_KEY = os.environ.get("API_KEY", "Bdzb59z3SQ")

####################################
# DATABASE SETTINGS
####################################

DATABASE_NAME = os.environ.get("DATABASE_NAME")
DATABASE_USER = os.environ.get("DATABASE_USER")
DATABASE_PASSWORD = os.environ.get("DATABASE_PASSWORD")
DATABASE_HOST = os.environ.get("DATABASE_HOST", "localhost")
DATABASE_PORT = os.environ.get("DATABASE_PORT", 5432)

DATABASE_URL = f"{DATABASE_USER}:{DATABASE_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}"

####################################
# PGVECTOR SETTINGS
####################################

PGVECTOR_INITIALIZE_MAX_VECTOR_LENGTH = int(
    os.environ.get("PGVECTOR_INITIALIZE_MAX_VECTOR_LENGTH", "1536")
)

PGVECTOR_DB_NAME = os.environ.get("PGVECTOR_DB_NAME")
PGVECTOR_DB_USER = os.environ.get("PGVECTOR_DB_USER")
PGVECTOR_DB_PASSWORD = os.environ.get("PGVECTOR_DB_PASSWORD")
PGVECTOR_DB_HOST = os.environ.get("PGVECTOR_DB_HOST", "localhost")
PGVECTOR_DB_PORT = os.environ.get("PGVECTOR_DB_PORT", 5432)

PGVECTOR_DB_URL = f"{PGVECTOR_DB_USER}:{PGVECTOR_DB_PASSWORD}@{PGVECTOR_DB_HOST}:{PGVECTOR_DB_PORT}/{PGVECTOR_DB_NAME}"

VECTOR_TABLE_NAME = os.environ.get("VECTOR_TABLE_NAME", "chatbot_vector")

####################################
# REDIS SETTINGS
####################################

# REDIS_HOST = os.environ.get("REDIS_HOST")
# REDIS_PORT = os.environ.get("REDIS_PORT")

####################################
# EMAIL SETTINGS
####################################

# EMAIL_USER = os.environ.get("EMAIL_USER")
# EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")


####################################
# STORAGE SETTINGS
####################################

DATA_DIR = PARENT_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
if not os.path.exists(UPLOAD_DIR):
    raise Exception(f"Directory not found: {UPLOAD_DIR}")

####################################
# RAG SETTINGS
####################################

RAG_EMBEDDING_ENGINE = os.environ.get(
    "RAG_EMBEDDING_ENGINE", "facebook/dpr-question_encoder-single-nq-base"
)
RAG_EMBEDDING_MODEL = os.environ.get("RAG_EMBEDDING_MODEL", "facebook/rag-token-nq")
RAG_EMBEDDING_BATCH_SIZE = int(
    os.environ.get("RAG_EMBEDDING_BATCH_SIZE")
    or os.environ.get("RAG_EMBEDDING_OPENAI_BATCH_SIZE", "1")
)
RAG_OLLAMA_BASE_URL = os.environ.get(
    "RAG_OLLAMA_BASE_URL", default="http://localhost:11434"
)
RAG_OLLAMA_API_KEY = os.environ.get("RAG_OLLAMA_API_KEY")
RAG_EMBEDDING_MODEL_TRUST_REMOTE_CODE = (
    os.environ.get("RAG_EMBEDDING_MODEL_TRUST_REMOTE_CODE", "True").lower() == "true"
)
SENTENCE_TRANSFORMERS_HOME = os.environ.get("SENTENCE_TRANSFORMERS_HOME")

USE_CUDA = os.environ.get("USE_CUDA_DOCKER", "false")

if USE_CUDA.lower() == "true":
    try:
        import torch

        assert torch.cuda.is_available(), "CUDA not available"
        DEVICE_TYPE = "cuda"
    except Exception as e:
        cuda_error = (
            "Error when testing CUDA but USE_CUDA_DOCKER is true. "
            f"Resetting USE_CUDA_DOCKER to false: {e}"
        )
        os.environ["USE_CUDA_DOCKER"] = "false"
        USE_CUDA = "false"
        DEVICE_TYPE = "cpu"
else:
    DEVICE_TYPE = "cpu"

try:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        DEVICE_TYPE = "mps"
except Exception:
    pass

####################################
# RAG SETTINGS
####################################

RAG_MODEL = os.environ.get("RAG_MODEL", "llama3.2")

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

####################################
# EMAIL SETTINGS
####################################

GOOGLE_EMAIL = os.environ.get("GOOGLE_EMAIL")
GOOGLE_PASSWORD = os.environ.get("GOOGLE_PASSWORD")

####################################
# GOOGLE SETTINGS
####################################

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET")
GOOGLE_CALENDAR_JSON = os.environ.get("GOOGLE_CALENDAR_JSON", "client_secret.json")

####################################
# REDIS SETTINGS
####################################

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = os.environ.get("REDIS_PORT", 6377)
QUEUE = os.environ.get("QUEUE", "chatbot_service")

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8080/api/v1")
ASSET_URL = BASE_URL + "/" + PY_ENV + "/uploads"
