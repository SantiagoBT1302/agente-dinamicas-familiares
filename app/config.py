from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # OpenAI
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"

    # Databricks
    databricks_host: str
    databricks_token: str
    databricks_http_path: str = "/sql/1.0/warehouses/bd303a51ca7560aa"
    databricks_catalog: str = "workspace"

    # App
    app_name: str = "Agente Dinámicas Familiares Eje Cafetero"
    app_version: str = "1.0.0"
    cors_origins: list[str] = ["*"]
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
