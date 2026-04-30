import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_health_and_ready():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        health = await client.get("/health")
        ready = await client.get("/ready")

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert ready.status_code == 200
    assert ready.json()["ready"] is True


@pytest.mark.asyncio
async def test_functional_predictions():
    transport = ASGITransport(app=app)
    payload = {"texts": ["I love this excellent product", "This was the worst experience"]}
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/predict", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert len(body["predictions"]) == 2
    assert body["predictions"][0]["label"] == "POSITIVE"
    assert body["predictions"][1]["label"] == "NEGATIVE"


@pytest.mark.asyncio
async def test_edge_case_empty_and_neutral_text():
    transport = ASGITransport(app=app)
    payload = {"texts": ["", "The package arrived yesterday."]}
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/predict", json=payload)

    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 2


@pytest.mark.asyncio
async def test_invalid_input_is_rejected():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/predict", json={"texts": []})
        malicious = await client.post("/predict", json={"texts": ["<script>alert('x')</script>"]})

    assert response.status_code == 422
    assert malicious.status_code == 200


@pytest.mark.asyncio
async def test_oversized_payload_is_rejected(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "max_text_length", 10)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/predict", json={"texts": ["x" * 100]})

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_concurrent_requests_stress():
    import asyncio

    transport = ASGITransport(app=app)

    async def send_request(client, i):
        return await client.post("/predict", json={"texts": [f"I love test {i}"]})

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        responses = await asyncio.gather(*(send_request(client, i) for i in range(50)))

    assert all(response.status_code == 200 for response in responses)
