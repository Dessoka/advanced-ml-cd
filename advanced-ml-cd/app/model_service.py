from __future__ import annotations

import os
from pathlib import Path
from typing import List

import numpy as np

from app.config import settings


class SentimentService:
    """ONNX sentiment inference service with deterministic mock mode for CI."""

    def __init__(self) -> None:
        self.mock = settings.mock_model
        self.session = None
        self.tokenizer = None
        self.input_names: list[str] = []

        if self.mock:
            return

        import onnxruntime as ort
        from transformers import AutoTokenizer

        model_path = Path(settings.model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. Add the lab ONNX file or set MOCK_MODEL=1."
            )

        providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_names = [inp.name for inp in self.session.get_inputs()]

        tokenizer_source = settings.tokenizer_name or str(model_path.parent)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)

    @property
    def ready(self) -> bool:
        return self.mock or self.session is not None

    def predict(self, texts: List[str]) -> list[dict[str, float | str]]:
        if self.mock:
            return [self._mock_predict(text) for text in texts]
        return self._onnx_predict(texts)

    def _mock_predict(self, text: str) -> dict[str, float | str]:
        negative_terms = {"bad", "awful", "hate", "terrible", "worst", "poor"}
        positive_terms = {"good", "great", "love", "excellent", "best", "amazing"}
        words = set(text.lower().split())

        if words & positive_terms and not words & negative_terms:
            return {"label": "POSITIVE", "score": 0.95}
        if words & negative_terms and not words & positive_terms:
            return {"label": "NEGATIVE", "score": 0.95}
        return {"label": "NEUTRAL", "score": 0.60}

    def _onnx_predict(self, texts: List[str]) -> list[dict[str, float | str]]:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="np",
        )

        feeds = {}
        for name in self.input_names:
            if name in encoded:
                feeds[name] = encoded[name]
            elif name == "token_type_ids":
                feeds[name] = np.zeros_like(encoded["input_ids"])

        outputs = self.session.run(None, feeds)
        logits = outputs[0]
        probs = self._softmax(logits)
        labels = ["NEGATIVE", "POSITIVE"]

        results = []
        for row in probs:
            idx = int(np.argmax(row))
            label = labels[idx] if idx < len(labels) else str(idx)
            results.append({"label": label, "score": float(row[idx])})
        return results

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.sum(exp, axis=1, keepdims=True)
