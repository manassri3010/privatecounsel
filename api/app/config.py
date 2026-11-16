from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo-root .env, found from this file so it loads whichever directory uvicorn starts in.
# Real environment variables still win, so Docker Compose and CI are unaffected.
ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    # extra="ignore": .env also holds keys for other services (POSTGRES_*, HF_TOKEN).
    model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore")

    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "qwen2.5:7b"
    vllm_api_key: str = "not-needed-for-ollama"
    database_url: str = ""


settings = Settings()
