"""Application configuration via environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configuration loaded from environment variables."""

    model_path: str = 'models/code_classifier.pt'
    host: str = '0.0.0.0'  # noqa: S104
    port: int = 8080
    log_level: str = 'info'

    model_config = {'env_prefix': 'CLASSIFIER_'}


settings = Settings()
