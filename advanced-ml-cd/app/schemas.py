from typing import List

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, description="Batch of text inputs")

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, texts: List[str]) -> List[str]:
        from app.config import settings

        if len(texts) > settings.max_batch_size:
            raise ValueError(f"Batch size exceeds limit of {settings.max_batch_size}")

        for text in texts:
            if not isinstance(text, str):
                raise ValueError("Each item must be a string")
            if len(text) > settings.max_text_length:
                raise ValueError(f"Input text exceeds {settings.max_text_length} characters")
        return texts


class Prediction(BaseModel):
    label: str
    score: float


class PredictResponse(BaseModel):
    predictions: List[Prediction]
