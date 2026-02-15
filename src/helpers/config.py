import re
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):

    APP_NAME: str

    class Config:
        env_file = ".env"

def get_settings():
    return Settings()
