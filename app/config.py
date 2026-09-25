from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / '.env', extra='ignore')

    host: str = '127.0.0.1'
    port: int = 8000

    # Generic LLM Configuration (Works for OpenRouter)
    model_provider: str = 'mistral'
    model_name: str = 'labs-leanstral-1-5'
    llm_api_key: str = ''
    llm_base_url: str = 'https://api.mistral.ai/v1'

    # Agent limits
    max_steps: int = Field(default=6, ge=1, le=6)
    max_tool_retries: int = Field(default=2, ge=0, le=2)
    max_output_tokens: int = Field(default=512, ge=1)
    run_timeout_seconds: float = Field(default=40, gt=0, le=40)
    max_history_messages: int = Field(default=12, ge=2, le=20)

settings = Settings()
