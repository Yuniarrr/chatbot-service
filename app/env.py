import os
from dotenv import load_dotenv

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
JWT_ACCESS_TOKEN_EXPIRE = os.environ.get("JWT_ACCESS_TOKEN_EXPIRE", "6")
REFRESH_TOKEN_EXPIRE_DAYS = os.environ.get("REFRESH_TOKEN_EXPIRE_DAYS", "7")
API_KEY = os.environ.get("API_KEY", "Bdzb59z3SQ")

####################################
# DATABASE SETTINGS
####################################

DATABASE_NAME = os.environ.get("DATABASE_NAME")
DATABASE_USER = os.environ.get("DATABASE_USER")
DATABASE_PASSWORD = os.environ.get("DATABASE_PASSWORD")
DATABASE_HOST = os.environ.get("DATABASE_HOST")
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
PGVECTOR_DB_HOST = os.environ.get("PGVECTOR_DB_HOST")
PGVECTOR_DB_PORT = os.environ.get("PGVECTOR_DB_PORT", 5432)

PGVECTOR_DB_URL = f"{PGVECTOR_DB_USER}:{PGVECTOR_DB_PASSWORD}@{PGVECTOR_DB_HOST}:{PGVECTOR_DB_PORT}/{PGVECTOR_DB_NAME}"

####################################
# REDIS SETTINGS
####################################

REDIS_HOST = os.environ.get("REDIS_HOST")
REDIS_PORT = os.environ.get("REDIS_PORT")

####################################
# EMAIL SETTINGS
####################################

EMAIL_USER = os.environ.get("EMAIL_USER")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")
