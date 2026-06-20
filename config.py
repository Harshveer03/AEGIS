from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache
from pathlib import Path

class Settings(BaseSettings):

    # ── LLM ────────────────────────────────────────────────────
    llm_provider: str = Field(default="ollama")
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="llama3.2")
    ollama_embed_model: str = Field(default="nomic-embed-text")
    anthropic_api_key: str = Field(default="")

    # ── MongoDB ─────────────────────────────────────────────────
    mongodb_url: str = Field(default="mongodb://localhost:27017")
    mongodb_db: str = Field(default="aegis")

    # ── Redis ───────────────────────────────────────────────────
    redis_url: str = Field(default="redis://localhost:6379")

    # ── Paths ───────────────────────────────────────────────────
    registry_path: str = Field(default="registry_store")
    generated_skills_path: str = Field(default="generated/skills")
    generated_agents_path: str = Field(default="generated/agents")
    log_path: str = Field(default="logs/aegis.log")

    # ── Trust levels ────────────────────────────────────────────
    default_trust_write: str = Field(default="approve")
    default_trust_delete: str = Field(default="approve")
    default_trust_send: str = Field(default="approve")
    default_trust_read: str = Field(default="autonomous")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # ── Derived helpers ─────────────────────────────────────────
    @property
    def agents_registry_path(self) -> Path:
        return Path(self.registry_path) / "agents.json"

    @property
    def skills_registry_path(self) -> Path:
        return Path(self.registry_path) / "skills.json"

    @property
    def pipelines_registry_path(self) -> Path:
        return Path(self.registry_path) / "pipelines.json"

    @property
    def is_ollama(self) -> bool:
        return self.llm_provider == "ollama"

    @property
    def is_claude(self) -> bool:
        return self.llm_provider == "claude"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


# Single import point used everywhere
settings = get_settings()