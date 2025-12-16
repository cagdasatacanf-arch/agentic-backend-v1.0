from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str
    openai_chat_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-large"
    vector_db_url: str = "http://qdrant:6333"
    redis_url: str = "redis://redis:6379/0"
    internal_api_key: str | None = None

    class Config:
        env_file = ".env"


settings = Settings()
