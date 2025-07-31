from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    google_api_key: str
    pinecone_api_key: str
    pinecone_index_name: str

    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()