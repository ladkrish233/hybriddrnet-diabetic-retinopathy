"""
dr_model.py
Shared model definition, preprocessing, losses, and inference helpers for the
Diabetic Retinopathy (DR) grading project.

This module is used by BOTH:
  - DR_Training_Colab.ipynb   (training, on Colab GPU)
  - inference.py / backend/   (FastAPI backend + Streamlit UI, local)

Architecture rationale (how this fills gaps in the 18-paper comparison):
  - CNN backbone (EfficientNetV2-S)   -> local lesion features (microaneurysms,
    hemorrhages, exudates), same family as Paper 12 (RA-EfficientNet).
  - Transformer encoder over spatial
    tokens with a CLS token           -> global context (vessel layout, optic
    disc position), same idea as Paper 9 (TOVIT) / Paper 11 (MVCAVIT), but
    fused directly onto CNN features instead of a pure ViT, which is cheaper
    to train from scratch on a single GPU.
  - Dual head: softmax classification
    + scalar ordinal regression       -> matches Paper 15's multi-task 3-head
    design, which achieved the best Quadratic Weighted Kappa (0.9255) in the
    comparison table. DR grades are ordinal (0..4), so a regression head
    trained jointly with the classifier directly optimizes for QWK instead of
    only accuracy.
  - Focal loss + weighted sampling   -> addresses class imbalance without the
    cost of training a separate GAN (Paper 10's WDCGAN approach), while still
    targeting the same problem.
  - Grad-CAM utility                 -> addresses the explainability gap
    flagged in Paper 3 (narrative review calling for XAI), and doubles as the
    visualization shown in the app.
"""

import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from scipy.optimize import minimize

try:
    import timm
except ImportError as e:
    raise ImportError("Run `pip install timm` before importing dr_model.") from e

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_CLASSES = 5
CLASS_NAMES = [
    "0 - No DR",
    "1 - Mild",
    "2 - Moderate",
    "3 - Severe",
    "4 - Proliferative DR",
]
IMAGE_SIZE = 320

# Extracted from Diabetic_Retinopathy_Research_Comparison.docx, used to plot
# "your model" against the literature. Values are overall accuracy (%) where
# reported; None where the paper reported a different primary metric (e.g.
# QWK-only, AUROC-only, or a review with no single number).
PAPER_ACCURACY_COMPARISON = {
    "P1 Shah'24 (YOLOv8+SVM)": 84.0,
    "P2 Bhattacharjee'21 (RF)": 76.5,
    "P5 Zaylaa'25 (ResNet50/GoogleNet)": 93.56,
    "P6 Jabbar'24 (VGG16)": 97.60,
    "P8 Sushith'25 (TAHDL)": 96.90,
    "P9 Bhoopalan'25 (TOVIT)": 99.0,
    "P10 Sreekumar'26 (ViT-WDCGAN)": 98.62,
    "P11 Vijayalakshmi'25 (MVCAVIT)": 98.9,
    "P12 Yi'21 (RA-EfficientNet, 2-grade)": 98.36,
    "P15 Tymchenko'20 (3-Head, binary)": 99.3,
    "P17 Shaban'20 (VGG-19, 3-class)": 88.5,
    "P18 Yadav (DeepReti)": 95.34,
}

# QWK reported by the strongest ordinal-regression papers, for the QWK chart.
PAPER_QWK_COMPARISON = {
    "P15 Tymchenko'20 (3-Head)": 0.9255,
    "P16 Majumder'21 (MSEDenseNet, APTOS)": 0.90,
    "P16 Majumder'21 (MXception, EyePACS)": 0.88,
    "P17 Shaban'20 (VGG-19)": 0.915,
}


# ---------------------------------------------------------------------------
# Preprocessing: Ben Graham circle-crop + CLAHE
# (standard for APTOS/EyePACS fundus images; used explicitly or implicitly by
#  Papers 2, 5, 6, 8, 9, 10, 12, 13, 14)
# ---------------------------------------------------------------------------

def _crop_image_from_gray(img, tol=7):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray > tol
    if mask.sum() == 0:
        return img
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return img[y0:y1, x0:x1]


def ben_graham_preprocess(bgr_img, sigma_x=10, out_size=IMAGE_SIZE):
    """Circle-crop to the fundus, CLAHE on the green/luminance channel, and a
    Gaussian-blur background-subtraction that is the de-facto standard
    preprocessing for retinal-fundus DL pipelines (used by top APTOS/EyePACS
    Kaggle solutions and mirrored across most reviewed papers)."""
    img = _crop_image_from_gray(bgr_img)
    img = cv2.resize(img, (out_size, out_size))

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    blurred = cv2.GaussianBlur(img, (0, 0), sigma_x)
    img = cv2.addWeighted(img, 4, blurred, -4, 128)
    return img


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Focal loss for the classification head; down-weights the easy,
    over-represented 'No DR' class instead of requiring a GAN oversampler."""

    def __init__(self, gamma=2.0, class_weights=None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("class_weights", class_weights, persistent=False)

    def forward(self, logits, targets):
        logp = F.log_softmax(logits, dim=1)
        p = logp.exp()
        logp_t = logp.gather(1, targets.unsqueeze(1)).squeeze(1)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = -((1 - p_t) ** self.gamma) * logp_t
        if self.class_weights is not None:
            loss = loss * self.class_weights[targets]
        return loss.mean()


class CombinedLoss(nn.Module):
    """alpha * focal(classification) + (1 - alpha) * MSE(ordinal regression).
    The ordinal term is what lets training directly target QWK, not just
    accuracy (see Paper 15's rationale for a 3-head network)."""

    def __init__(self, class_weights=None, alpha=0.6, gamma=2.0):
        super().__init__()
        self.focal = FocalLoss(gamma=gamma, class_weights=class_weights)
        self.alpha = alpha

    def forward(self, logits, ord_pred, targets):
        cls_loss = self.focal(logits, targets)
        ord_loss = F.mse_loss(ord_pred, targets.float())
        return self.alpha * cls_loss + (1 - self.alpha) * ord_loss, cls_loss.item(), ord_loss.item()


# ---------------------------------------------------------------------------
# Model: EfficientNetV2-S backbone + Transformer fusion + dual head
# ---------------------------------------------------------------------------

class HybridDRNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, embed_dim=256, depth=4,
                 num_heads=8, drop=0.3, backbone_name="tf_efficientnetv2_s",
                 pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, features_only=True, out_indices=(4,)
        )
        feat_ch = self.backbone.feature_info.channels()[-1]
        self.proj = nn.Conv2d(feat_ch, embed_dim, kernel_size=1)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # base positional grid sized for a 320x320 input through EffNetV2-S
        # (stride 32 -> 10x10 = 100 tokens); interpolated at runtime for any
        # other input resolution.
        self.base_grid = 10
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.base_grid ** 2, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            dropout=drop, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

        self.head_drop = nn.Dropout(drop)
        self.cls_head = nn.Linear(embed_dim, num_classes)
        self.ord_head = nn.Sequential(nn.Linear(embed_dim, 64), nn.GELU(), nn.Dropout(drop), nn.Linear(64, 1))

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self._last_feat = None  # populated on forward, used by GradCAM

    def _interpolated_pos_embed(self, num_tokens):
        if num_tokens == self.base_grid ** 2:
            return self.pos_embed
        cls_pos = self.pos_embed[:, :1]
        grid_pos = self.pos_embed[:, 1:].transpose(1, 2)  # (1, D, N)
        grid_pos = F.interpolate(grid_pos, size=num_tokens, mode="linear", align_corners=False)
        grid_pos = grid_pos.transpose(1, 2)
        return torch.cat([cls_pos, grid_pos], dim=1)

    def forward(self, x):
        feat = self.backbone(x)[-1]        # (B, C, H, W)
        feat = self.proj(feat)             # (B, D, H, W)
        self._last_feat = feat             # for Grad-CAM
        if feat.requires_grad:
            feat.retain_grad()

        B, D, H, W = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)              # (B, HW, D)
        cls_tok = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tok, tokens], dim=1)
        tokens = tokens + self._interpolated_pos_embed(H * W)

        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)
        cls_feat = self.head_drop(tokens[:, 0])

        logits = self.cls_head(cls_feat)
        ordinal = self.ord_head(cls_feat).squeeze(-1)
        return logits, ordinal


# ---------------------------------------------------------------------------
# Regularization utilities added after observing overfitting past ~epoch 19
# (train QWK -> 0.99 while val QWK plateaued/declined from ~0.90): MixUp
# fights this by training on blended image/label pairs instead of exact
# examples, and EMA smooths out the epoch-to-epoch val-metric noise that came
# from an increasingly overfit raw model.
# ---------------------------------------------------------------------------

def mixup_batch(images, targets, num_classes, alpha=0.2):
    """Returns mixed images plus soft classification targets and a blended
    ordinal target, along with the lambda used (for logging)."""
    if alpha <= 0:
        return images, F.one_hot(targets, num_classes).float(), targets.float(), 1.0
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(images.size(0), device=images.device)
    mixed_images = lam * images + (1 - lam) * images[perm]
    onehot = F.one_hot(targets, num_classes).float()
    mixed_cls_target = lam * onehot + (1 - lam) * onehot[perm]
    mixed_ord_target = lam * targets.float() + (1 - lam) * targets.float()[perm]
    return mixed_images, mixed_cls_target, mixed_ord_target, lam


def soft_focal_loss(logits, soft_targets, gamma=2.0, class_weights=None):
    """Focal loss generalized to soft (mixed) targets: cross-entropy against
    a probability distribution instead of a hard class index."""
    logp = F.log_softmax(logits, dim=1)
    p = logp.exp()
    focal_weight = (1 - p) ** gamma
    loss = -(soft_targets * focal_weight * logp)
    if class_weights is not None:
        loss = loss * class_weights.unsqueeze(0)
    return loss.sum(dim=1).mean()


class ModelEMA:
    """Exponential moving average of model weights. The EMA'd model
    generalizes more smoothly than the raw, still-training model, which is
    why it's evaluated separately each epoch and checkpointed if it beats
    the raw model's validation QWK."""

    def __init__(self, model, decay=0.999):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        for ema_p, p in zip(self.ema.state_dict().values(), model.state_dict().values()):
            if ema_p.dtype.is_floating_point:
                ema_p.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
            else:
                ema_p.copy_(p)


# ---------------------------------------------------------------------------
# QWK-optimized decision thresholds for the ordinal-regression head.
# Standard trick from top APTOS/EyePACS Kaggle solutions (and the rationale
# behind Paper 15's regression head): argmax on the classifier ignores that
# grades are ordered, while rounding the continuous ordinal score at fixed
# 0.5 boundaries is not necessarily QWK-optimal. Fitting the boundaries
# directly against QWK on a validation set typically adds a few points of
# QWK for free, with no retraining.
# ---------------------------------------------------------------------------

class OptimizedRounder:
    def __init__(self, num_classes=NUM_CLASSES):
        self.num_classes = num_classes
        self.thresholds = None

    def _predict(self, scores, thresholds):
        preds = np.digitize(scores, thresholds)
        return np.clip(preds, 0, self.num_classes - 1)

    def fit(self, ordinal_scores, true_labels):
        from sklearn.metrics import cohen_kappa_score

        init_thresholds = np.arange(0.5, self.num_classes - 0.5, 1.0)

        def neg_qwk(thresholds):
            preds = self._predict(ordinal_scores, np.sort(thresholds))
            return -cohen_kappa_score(true_labels, preds, weights="quadratic")

        result = minimize(neg_qwk, init_thresholds, method="Nelder-Mead",
                           options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 2000})
        self.thresholds = np.sort(result.x)
        return self

    def predict(self, ordinal_scores):
        return self._predict(ordinal_scores, self.thresholds)


# ---------------------------------------------------------------------------
# Grad-CAM (manual hooks; targets the CNN feature map right before fusion, so
# it works regardless of the transformer sitting downstream)
# ---------------------------------------------------------------------------

class GradCAM:
    def __init__(self, model: HybridDRNet):
        self.model = model

    def __call__(self, x, target_class=None):
        self.model.zero_grad(set_to_none=True)
        logits, _ = self.model(x)
        if target_class is None:
            target_class = logits.argmax(dim=1)
        score = logits.gather(1, target_class.view(-1, 1)).sum()
        score.backward()

        feat = self.model._last_feat            # (B, D, H, W)
        grad = feat.grad                          # (B, D, H, W)
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * feat).sum(dim=1))  # (B, H, W)
        cam = cam - cam.amin(dim=(1, 2), keepdim=True)
        cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)
        return cam.detach().cpu().numpy(), logits.detach()


# ---------------------------------------------------------------------------
# Test-time augmentation inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_tta(model, x, device):
    """Average softmax predictions over identity + horizontal + vertical flip."""
    model.eval()
    views = [x, torch.flip(x, dims=[3]), torch.flip(x, dims=[2])]
    probs_sum = None
    ord_sum = 0.0
    for v in views:
        logits, ordinal = model(v.to(device))
        probs = F.softmax(logits, dim=1)
        probs_sum = probs if probs_sum is None else probs_sum + probs
        ord_sum = ord_sum + ordinal
    probs_avg = probs_sum / len(views)
    ord_avg = ord_sum / len(views)
    return probs_avg, ord_avg


def remap_legacy_state_dict(state_dict):
    """Checkpoints trained before the ordinal head gained a Dropout layer store
    its final Linear as `ord_head.2.*`; it now sits at `ord_head.3.*`. Dropout
    has no parameters and is identity at eval time, so renaming the keys gives
    numerically identical inference."""
    if "ord_head.2.weight" in state_dict and "ord_head.3.weight" not in state_dict:
        state_dict = dict(state_dict)
        state_dict["ord_head.3.weight"] = state_dict.pop("ord_head.2.weight")
        state_dict["ord_head.3.bias"] = state_dict.pop("ord_head.2.bias")
    return state_dict


def load_model_for_inference(checkpoint_path, device="cpu"):
    """Returns (model, metadata) where metadata holds the checkpoint's epoch/val_qwk."""
    model = HybridDRNet(pretrained=False)
    # weights_only=False: our own checkpoints carry plain metadata (epoch, val_qwk)
    # alongside the state dict; only load checkpoints you trust the source of.
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = state["model_state_dict"] if "model_state_dict" in state else state
    model.load_state_dict(remap_legacy_state_dict(state_dict))
    model.to(device)
    model.eval()
    metadata = {k: state[k] for k in ("epoch", "val_qwk", "source") if isinstance(state, dict) and k in state}
    return model, metadata
