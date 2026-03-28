from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # LLM settings
    classifier_model: str = "gemini-2.5-flash"
    llm_temperature: float = 0.2
    llm_timeout_seconds: int = 30
    openai_api_key: str = ""
    google_api_key: str = ""  # Added this line so Pydantic allows your new key!

    # Classifier settings
    classifier_confidence_threshold: float = 0.8

    # Cache settings
    cache_ttl_seconds: int = 3600

    class Config:
        env_file = ".env"
        extra = "allow" # This tells Pydantic not to crash if it finds extra stuff in your .env

def get_settings() -> Settings:
    return Settings()