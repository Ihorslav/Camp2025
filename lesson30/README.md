# Simple Computer Vision API (FastAPI) — Digits Classifier (8x8)

## Overview
This is a minimal computer vision solution that classifies **handwritten digits (0–9)** using a simple **linear model (Logistic Regression)** trained on the **scikit-learn `digits` dataset** (8×8 grayscale).  
It includes:
- **Model training script** (uses only local, offline dataset from scikit-learn).
- **FastAPI** service for inference.
- **Visualization demo** (optional) and example logs.
- **Dockerfile** for containerized deployment.

The chosen dataset satisfies the requirement **resolution ≤ 80×80** (it's 8×8).

## Deployment Info
You can run it locally with Uvicorn or via Docker.

### Run locally
```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

### Run with Docker
```bash
docker build -t cv-fastapi-digits .
docker run -it --rm -p 8000:8000 cv-fastapi-digits
```

The API will be available at: `http://localhost:8000`.  
Docs (Swagger UI): `http://localhost:8000/docs`

## Installation Instructions
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# (Optional) re-train model
python train.py
# Run API
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Modeling Info
- **Dataset**: `sklearn.datasets.load_digits()` (8×8 grayscale, values 0..16).
- **Preprocessing**: flatten to 64-D vector, divide by 16 → [0,1].
- **Model**: `LogisticRegression(max_iter=2000, lbfgs)` — linear classifier.
- **Metric**: accuracy on hold-out test. Current model accuracy: **0.9583**.

## Interface Description
Base URL: `/`

### `GET /health`
- **Description**: Health check.
- **Response**: `{ "status": "ok" }`

### `GET /version`
- **Description**: Returns model/meta info.
- **Response**: JSON with training accuracy and model info.

### `POST /predict-array`
- **Description**: Predict digit from **raw 8×8 vector** (length 64) or list of lists 8×8.
- **Request JSON**:
  ```json
  {
    "pixels": [0, 0, 1, ..., 0]  // length 64 or shape 8x8
  }
  ```
- **Response**:
  ```json
  {
    "pred": 7,
    "proba": [ ... 10 floats ... ]
  }
  ```

### `POST /predict-image`
- **Description**: Predict digit from an **image file** (PNG/JPG). Image will be converted to grayscale, resized to 8×8, values scaled to 0..1 using division by 255 and mapped to 0..16 scale to approximate the dataset.
- **Form-data** (multipart):
  - `file`: uploaded image file
- **Response**:
  ```json
  {
    "pred": 3,
    "proba": [ ... 10 floats ... ]
  }
  ```

## Example Processes

### cURL examples
```bash
# Health
curl -s http://localhost:8000/health

# Version
curl -s http://localhost:8000/version

# Predict from array (toy example: zeros)
curl -s -X POST http://localhost:8000/predict-array \
  -H "Content-Type: application/json" \
  -d '{"pixels": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}'

# Predict from image
curl -s -X POST http://localhost:8000/predict-image \
  -F "file=@sample_digit.png"
```

### Example logs (when running Uvicorn)
```
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     127.0.0.1:54012 - "GET /health HTTP/1.1" 200 OK
INFO:     127.0.0.1:54020 - "POST /predict-array HTTP/1.1" 200 OK
INFO:     127.0.0.1:54028 - "POST /predict-image HTTP/1.1" 200 OK
```
