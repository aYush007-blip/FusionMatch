import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MODEL_NAME: str = "google/siglip-base-patch16-224"
    SIMILARITY_THRESHOLD: float = 0.86
    EMBEDDING_DIM: int = 768
    FAISS_INDEX_PATH: str = os.getenv("FAISS_INDEX_PATH", "/app/data/catalog.index")
    METADATA_PATH: str = os.getenv("METADATA_PATH", "/app/data/metadata.json")

settings = Settings()