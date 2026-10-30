"""Inference wrapper around HybridDRNet: preprocessing, TTA prediction and Grad-CAM.

Used by the FastAPI backend, the sample-prediction script and the tests, so all
three run the exact same pipeline the model was trained with.
"""

import base64
import threading

import cv2
import numpy as np
import torch

from dr_model import (
    CLASS_NAMES,
    IMAGE_SIZE,
    GradCAM,
    ben_graham_preprocess,
    load_model_for_inference,
    predict_tta,
)

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DISCLAIMER = (
    "Research prototype only. Not a medical device and not a substitute for "
    "examination by a qualified ophthalmologist."
)


def encode_png_b64(rgb):
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def decode_image_bytes(data):
    """Decode uploaded bytes to a BGR image; raises ValueError on anything unreadable."""
    if not data:
        raise ValueError("Empty file.")
    bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("File is not a readable image (expected PNG or JPEG).")
    return bgr


class DRPredictor:
    def __init__(self, checkpoint_path, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model, self.metadata = load_model_for_inference(checkpoint_path, self.device)
        self.gradcam = GradCAM(self.model)
        # GradCAM keeps per-forward state on the model, so inference is serialized.
        self._lock = threading.Lock()

    def preprocess(self, bgr):
        """Returns (preprocessed RGB uint8 image, normalized 1xCxHxW tensor)."""
        processed_rgb = cv2.cvtColor(ben_graham_preprocess(bgr, out_size=IMAGE_SIZE), cv2.COLOR_BGR2RGB)
        normalized = (processed_rgb.astype(np.float32) / 255.0 - MEAN) / STD
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0)
        return processed_rgb, tensor

    def predict_probabilities(self, bgr_images, batch_size=16):
        """Batched TTA prediction without Grad-CAM; returns (probs Nx5, ordinal N)."""
        all_probs, all_ordinal = [], []
        with self._lock:
            for start in range(0, len(bgr_images), batch_size):
                batch = torch.cat([self.preprocess(b)[1] for b in bgr_images[start:start + batch_size]])
                probs, ordinal = predict_tta(self.model, batch, self.device)
                all_probs.append(probs.cpu().numpy())
                all_ordinal.append(ordinal.cpu().numpy())
        return np.concatenate(all_probs), np.concatenate(all_ordinal)

    def predict(self, bgr, with_gradcam=True):
        processed_rgb, x = self.preprocess(bgr)
        with self._lock:
            probs, ordinal = predict_tta(self.model, x, self.device)
            probs = probs[0].cpu().numpy()
            grade = int(probs.argmax())

            overlay = heatmap_rgb = None
            if with_gradcam:
                cams, _ = self.gradcam(x.to(self.device), target_class=torch.tensor([grade], device=self.device))
                overlay, heatmap_rgb = _overlay_cam(processed_rgb, cams[0])

        return {
            "grade": grade,
            "label": CLASS_NAMES[grade],
            "confidence": float(probs[grade]),
            "probabilities": {CLASS_NAMES[i]: float(p) for i, p in enumerate(probs)},
            "ordinal_score": float(ordinal[0]),
            "preprocessed_rgb": processed_rgb,
            "gradcam_overlay_rgb": overlay,
            "gradcam_heatmap_rgb": heatmap_rgb,
        }


def _overlay_cam(processed_rgb, cam):
    cam_resized = cv2.resize(cam, (processed_rgb.shape[1], processed_rgb.shape[0]))
    heatmap_bgr = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(processed_rgb, 0.55, heatmap_rgb, 0.45, 0)
    return overlay, heatmap_rgb
