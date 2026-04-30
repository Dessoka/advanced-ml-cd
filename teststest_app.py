import requests

def test_home():
    response = requests.get("http://localhost:8000")
    assert response.status_code == 200

def test_predict():
    response = requests.post(
        "http://localhost:8000/predict",
        json={"text": "I love this"}
    )
    assert response.status_code == 200
