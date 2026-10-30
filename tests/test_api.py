import base64
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.main import CHECKPOINT_PATH, app
from dr_model import CLASS_NAMES

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = sorted((ROOT / "samples").glob("stage_*.png"))

pytestmark = pytest.mark.skipif(not CHECKPOINT_PATH.exists(), reason="model checkpoint not present")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_model_info(client):
    body = client.get("/model-info").json()
    assert body["classes"] == CLASS_NAMES
    assert body["checkpoint_epoch"] == 19
    assert body["input_size"] == 320


@pytest.mark.skipif(not SAMPLES, reason="run scripts/run_sample_predictions.py to create samples/")
@pytest.mark.parametrize("sample", SAMPLES, ids=lambda p: p.stem)
def test_predict_returns_valid_prediction_and_gradcam(client, sample):
    response = client.post("/predict", files={"file": (sample.name, sample.read_bytes(), "image/png")})
    assert response.status_code == 200
    body = response.json()

    assert 0 <= body["grade"] <= 4
    assert body["label"] == CLASS_NAMES[body["grade"]]
    assert set(body["probabilities"]) == set(CLASS_NAMES)
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-4)
    assert body["confidence"] == pytest.approx(max(body["probabilities"].values()))

    for key in ("gradcam_png_base64", "preprocessed_png_base64"):
        img = cv2.imdecode(np.frombuffer(base64.b64decode(body[key]), np.uint8), cv2.IMREAD_COLOR)
        assert img is not None and img.shape == (320, 320, 3)


def test_predict_without_gradcam(client):
    ok, buf = cv2.imencode(".jpg", np.full((300, 300, 3), 90, np.uint8))
    response = client.post("/predict", params={"gradcam": "false"}, files={"file": ("x.jpg", buf.tobytes(), "image/jpeg")})
    assert response.status_code == 200
    assert response.json()["gradcam_png_base64"] is None


def test_rejects_non_image(client):
    response = client.post("/predict", files={"file": ("notes.txt", b"definitely not an image", "text/plain")})
    assert response.status_code == 400


def test_rejects_empty_file(client):
    response = client.post("/predict", files={"file": ("empty.png", b"", "image/png")})
    assert response.status_code == 400
