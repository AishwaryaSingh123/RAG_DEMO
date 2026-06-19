# backend/src/core/config.py
import os
from dotenv import load_dotenv

# Load env file once at load time
load_dotenv()

class Settings:
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    MODEL_PATH: str = os.getenv("MODEL_PATH", "BAAI/bge-m3")
    DB_PATH: str = os.getenv("DB_PATH", "database")
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "research_rag")
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "super_secret_jwt_key_replace_in_prod")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 1 day

    # Cost tracking parameters (Gemini 3.5 Flash)
    COST_PER_1M_INPUT_TOKENS: float = 0.075
    COST_PER_1M_OUTPUT_TOKENS: float = 0.30
    COST_PER_1M_EMBEDDING_TOKENS: float = 0.02

settings = Settings()
