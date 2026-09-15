from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""
    ## OPENAI
    OPENAI_API_KEY: str
    CHAT_MODEL: str
    TEMPERATURE: Optional[float] = None
    STREAMING: bool
    GRAPH_RECURSION_LIMIT: int
    RETRY_ATTEMPTS: int
    DATABASE_URL: str

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="allow"
    )

settings = Settings()