from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_path: str = Field(default="model/sentiment.onnx", alias="MODEL_PATH")
    tokenizer_name: str | None = Field(default=None, alias="TOKENIZER_NAME")
    mock_model: bool = Field(default=False, alias="MOCK_MODEL")
    max_batch_size: int = Field(default=32, alias="MAX_BATCH_SIZE")
    max_text_length: int = Field(default=2000, alias="MAX_TEXT_LENGTH")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
