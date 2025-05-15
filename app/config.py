from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings) :
    data_dir: Path
    model_dir: Path
    api_port: int
    log_file: Path
    log_level: str

    class Config :
        env_file = ".env"

settings = Settings()