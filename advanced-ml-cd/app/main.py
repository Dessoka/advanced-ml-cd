from fastapi import FastAPI, HTTPException

from app.model_service import SentimentService
from app.schemas import PredictRequest, PredictResponse

app = FastAPI(
    title="Advanced ML Continuous Delivery Sentiment API",
    version="1.0.0",
    description="FastAPI service for ONNX sentiment analysis.",
)

service = SentimentService()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, bool]:
    return {"ready": service.ready}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    if not service.ready:
        raise HTTPException(status_code=503, detail="Model is not ready")
    predictions = service.predict(payload.texts)
    return PredictResponse(predictions=predictions)
