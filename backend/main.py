"""FastAPI backend serving HybridDRNet predictions with Grad-CAM explanations.

Run from the repository root:
    uvicorn backend.main:app --host 127.0.0.1 --port 8000
"""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dr_model import CLASS_NAMES, IMAGE_SIZE  # noqa: E402
from inference import DISCLAIMER, DRPredictor, decode_image_bytes, encode_png_b64  # noqa: E402

CHECKPOINT_PATH = Path(os.environ.get("DR_MODEL_CHECKPOINT", ROOT / "models" / "best_model.pth"))
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


class PredictionResponse(BaseModel):
    grade: int
    label: str
    confidence: float
    probabilities: dict[str, float]
    ordinal_score: float
    preprocessed_png_base64: str
    gradcam_png_base64: str | None
    disclaimer: str


class ModelInfo(BaseModel):
    architecture: str
    classes: list[str]
    input_size: int
    checkpoint_epoch: int | None
    checkpoint_val_qwk: float | None
    device: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not CHECKPOINT_PATH.exists():
        raise RuntimeError(f"Model checkpoint not found at {CHECKPOINT_PATH}")
    app.state.predictor = DRPredictor(CHECKPOINT_PATH)
    yield


app = FastAPI(
    title="HybridDRNet Diabetic Retinopathy API",
    description="5-stage diabetic retinopathy grading from fundus photos, with Grad-CAM. " + DISCLAIMER,
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": hasattr(app.state, "predictor")}


@app.get("/model-info", response_model=ModelInfo)
def model_info():
    predictor = app.state.predictor
    val_qwk = predictor.metadata.get("val_qwk")
    return ModelInfo(
        architecture="HybridDRNet (EfficientNetV2-S + Transformer fusion, classification + ordinal heads)",
        classes=CLASS_NAMES,
        input_size=IMAGE_SIZE,
        checkpoint_epoch=predictor.metadata.get("epoch"),
        checkpoint_val_qwk=float(val_qwk) if val_qwk is not None else None,
        device=str(predictor.device),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(
    file: UploadFile = File(..., description="Fundus photograph (PNG or JPEG)"),
    gradcam: bool = Query(True, description="Include a Grad-CAM overlay for the predicted grade"),
):
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File larger than 20 MB.")
    try:
        bgr = decode_image_bytes(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    result = app.state.predictor.predict(bgr, with_gradcam=gradcam)
    return PredictionResponse(
        grade=result["grade"],
        label=result["label"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        ordinal_score=result["ordinal_score"],
        preprocessed_png_base64=encode_png_b64(result["preprocessed_rgb"]),
        gradcam_png_base64=encode_png_b64(result["gradcam_overlay_rgb"]) if gradcam else None,
        disclaimer=DISCLAIMER,
    )
