import io
import math
import joblib
import numpy as np
from PIL import Image, ImageOps
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from typing import List

ARTIFACT_PATH = "artifacts/digits_logreg.joblib"

app = FastAPI(title="Digits Classifier API (8x8)")

_model = None
_meta = None

def load_artifacts():
    global _model, _meta
    if _model is None:
        bundle = joblib.load(ARTIFACT_PATH)
        _model = bundle["model"]
        _meta = bundle.get("meta", {})
    return _model, _meta

class ArrayPayload(BaseModel):
    pixels: List[float] = Field(..., description="Either length-64 flat list or 8x8 list of lists")

def preprocess_array(pix):
    arr = np.array(pix, dtype=np.float32)
    if arr.ndim == 2:
        if arr.shape != (8, 8):
            raise HTTPException(status_code=400, detail=f"2D array must be 8x8, got {arr.shape}")
        arr = arr.flatten()
    elif arr.ndim == 1:
        if arr.shape[0] != 64:
            raise HTTPException(status_code=400, detail=f"Flat array must be length 64, got {arr.shape[0]}")
    else:
        raise HTTPException(status_code=400, detail="Pixels must be 1D (64) or 2D (8x8)")

    maxv = float(arr.max()) if arr.size else 1.0
    if maxv > 1.0:
        if maxv > 32:
            arr = (arr / 255.0)  
            arr = arr 
        else:
            arr = arr / 16.0
    return arr.astype(np.float32)

def preprocess_image_to_flat8x8(file_bytes: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("L")  
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")
    img = ImageOps.fit(img, (8, 8), method=Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) 
    arr = arr / 255.0  
    flat = arr.flatten().astype(np.float32)
    return flat

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/version")
def version():
    _, meta = load_artifacts()
    return {"model": "LogisticRegression (digits 8x8)", "meta": meta}

@app.post("/predict-array")
def predict_array(payload: ArrayPayload):
    model, _ = load_artifacts()
    x = preprocess_array(payload.pixels).reshape(1, -1)  
    logits = model.predict_proba(x)[0]
    pred = int(np.argmax(logits))
    return {"pred": pred, "proba": logits.tolist()}

@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
        raise HTTPException(status_code=400, detail="Only image files are supported (png/jpg/jpeg/bmp)")
    data = await file.read()
    x = preprocess_image_to_flat8x8(data).reshape(1, -1)
    model, _ = load_artifacts()
    logits = model.predict_proba(x)[0]
    pred = int(np.argmax(logits))
    return {"pred": pred, "proba": logits.tolist()}
