from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # This tells Pydantic to find a variable named GOOGLE_API_KEY in the .env file.
    google_api_key: str

    model_config = SettingsConfigDict(env_file=".env")

# Create a single, reusable instance of the settings for the whole app.
settings = Settings()