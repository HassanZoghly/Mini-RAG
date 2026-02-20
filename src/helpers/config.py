import re
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):

    APP_NAME: str

    FILE_ALLOWED_TYPES: str
    FILE_MAX_SIZE: int
    FILE_DEFAULT_CHUNK: int

    MONGODB_URL: str
    MONGODB_DATABASE: str

    class Config:
        env_file = ".env"

def get_settings():
    return Settings()
