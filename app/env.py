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

JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY")

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
# REDIS SETTINGS
####################################

REDIS_HOST = os.environ.get("REDIS_HOST")
REDIS_PORT = os.environ.get("REDIS_PORT")

####################################
# EMAIL SETTINGS
####################################

EMAIL_USER = os.environ.get("EMAIL_USER")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")
