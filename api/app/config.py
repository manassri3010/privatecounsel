from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "qwen2.5:7b"
    vllm_api_key: str = "not-needed-for-ollama"
    database_url: str = ""


settings = Settings()
