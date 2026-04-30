# advanced-ml-cd

Continuous Delivery project for a containerized ONNX sentiment-analysis model served with FastAPI.

> Add the lab-provided ONNX file to `model/sentiment.onnx` before running the real model. The automated tests use `MOCK_MODEL=1`, so CI can run even when the large model file is not committed.

## Features

- FastAPI REST API for sentiment prediction
- ONNX Runtime inference wrapper
- Health and readiness endpoints
- Concurrent request handling with FastAPI and Uvicorn workers
- Docker containerization using a slim Python image
- GitHub Actions CI pipeline
- Functional, edge-case, robustness, integration, and stress/concurrency tests

## Repository setup

```bash
git init
git branch -M main
git remote add origin https://github.com/<your-user>/advanced-ml-cd.git
```

## Prerequisites

- Python 3.11+
- Docker
- GitHub account
- Lab-provided ONNX sentiment-analysis model

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
```

Place your model at `model/sentiment.onnx`.

If your ONNX model requires a Hugging Face tokenizer, also place tokenizer files in `model/`, or set `TOKENIZER_NAME=distilbert-base-uncased`.

## Run the API locally

```bash
MOCK_MODEL=1 uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Real ONNX mode:

```bash
MODEL_PATH=model/sentiment.onnx uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/docs`.

## Example request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"texts":["I love this product","This was awful"]}'
```

## Run tests

```bash
MOCK_MODEL=1 pytest -q
```

## Docker

```bash
docker build -t advanced-ml-cd:latest .
docker run -p 8000:8000 -e MOCK_MODEL=1 advanced-ml-cd:latest
```

Run with a mounted model:

```bash
docker run -p 8000:8000 \
  -e MODEL_PATH=/app/model/sentiment.onnx \
  -v "$PWD/model:/app/model" \
  advanced-ml-cd:latest
```

## CI/CD pipeline

The workflow at `.github/workflows/ci.yml` runs on pushes and pull requests to `main`. It installs dependencies, runs linting, executes automated tests, builds the Docker image, performs a container smoke test, and uploads the built image as an artifact. A production version can extend this by pushing the image to GitHub Container Registry and deploying through canary or blue-green release stages.

## Part 1: Conceptual Questions

### 1. Importance of CI/CD for operationalizing ML models

Continuous Integration and Continuous Delivery are important because ML systems change along three dimensions: code, data, and model artifacts. Traditional software CI usually validates source code behavior. ML CI/CD must also validate training data assumptions, preprocessing logic, model accuracy, inference latency, artifact compatibility, and production behavior after deployment.

CI helps by automatically running unit tests, data validation, model-loading checks, security checks, and regression tests whenever code changes. CD helps by packaging a tested model and service into a repeatable release artifact, such as a container image, and deploying it through controlled environments.

ML projects have additional challenges compared with traditional software. Model behavior is probabilistic, performance can degrade because of data drift, and the same code can behave differently when trained on different data. CI/CD reduces these risks by making builds reproducible, enforcing model quality gates, tracking artifacts, and supporting safe rollout strategies.

### 2. Packaging an ML model into a container

Packaging an ML model into a container involves placing the model artifact, inference code, dependency files, and serving framework into a Docker image. A typical process is to save the model artifact, build an API with FastAPI or Flask, load the model at startup, add prediction and health endpoints, write a Dockerfile, expose the service port, and define a production start command.

Key considerations include image size, dependency pinning, CPU/GPU compatibility, secure non-root execution, startup time, model file size, concurrency, and reproducibility. Containerization is beneficial because it makes the runtime environment consistent across developer machines, CI runners, staging, and production. It also simplifies scaling and rollback in platforms such as Kubernetes.

### 3. Blue-green versus canary deployment

Blue-green deployment uses two production-like environments. The current version runs in the blue environment while the new version is deployed to the green environment. After validation, traffic is switched from blue to green. Its advantages are fast rollback and clean separation between old and new versions. Drawbacks include higher infrastructure cost and the risk that a full traffic switch exposes all users if validation is incomplete.

Canary deployment gradually sends a small percentage of traffic to the new model while most traffic remains on the existing model. Its advantages are lower blast radius, real-world validation, and easier monitoring of model behavior on production traffic. Drawbacks include operational complexity, longer rollout time, and the need for careful metric comparison.

For a critical healthcare ML application, I would recommend canary deployment with strict safety gates, human oversight, and immediate rollback. Healthcare applications have high user-safety risk, so exposing the new model to a small controlled traffic segment is safer than switching all traffic at once. Blue-green can still be useful for infrastructure rollback, but canary release is better for validating model behavior under real-world conditions.

### 4. Critical checks and automated tests

A reliable ML CI/CD pipeline should include model loading tests, functional inference tests, data validation tests, performance tests, robustness/security tests, regression tests, container smoke tests, and dependency checks. These tests matter because an ML service can fail through application bugs, incompatible model artifacts, invalid inputs, performance regressions, or silent prediction quality degradation.

### 5. Monitoring and logging

Monitoring and logging are essential after deployment because CI/CD only proves that a release passed pre-production checks. Production conditions can still change. Monitoring tracks latency, error rate, throughput, CPU, memory, prediction distribution, confidence scores, data drift, and accuracy when ground truth becomes available.

Logging supports debugging and auditability. Structured logs can capture request IDs, timestamps, model version, validation errors, and prediction metadata without storing sensitive raw data unnecessarily. Together, monitoring and logging enable faster incident response, rollback decisions, continuous model improvement, and detection of drift or bias after release.
