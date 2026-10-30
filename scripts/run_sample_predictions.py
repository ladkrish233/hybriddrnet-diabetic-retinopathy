"""Evaluate the checkpoint on its own held-out validation split, then predict one
image per DR stage with Grad-CAM.

The training notebook split Kaggle's train.csv with
`train_test_split(test_size=0.15, stratify=diagnosis, random_state=42)`. That
split depends only on row order, labels and seed, so it is reproduced exactly
from a public APTOS mirror that keeps train.csv order (verified by matching
class counts and leading labels). As a control, the same number of rows from
the *training* side is evaluated too: if the split is reproduced correctly,
the model should score clearly higher there (it saw those images).

Caveat: the mirror stores images resized to 224x224, while training used the
original full-resolution photos, so these numbers are expected to sit somewhat
below the notebook's (acc 0.8255, QWK 0.9008).
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split  # noqa: E402

from dr_model import CLASS_NAMES, NUM_CLASSES  # noqa: E402
from inference import DRPredictor  # noqa: E402

MIRROR_URL = "https://huggingface.co/datasets/bumbledeep/aptos/resolve/main/data/train-00000-of-00001.parquet"
SEED = 42
VAL_FRACTION = 0.15


def load_mirror(cache_path):
    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading APTOS mirror (~250 MB) to {cache_path} ...")
        urllib.request.urlretrieve(MIRROR_URL, cache_path)
    df = pq.read_table(cache_path).to_pandas()
    df["diagnosis"] = df["label_code"].astype(int)
    return df


def decode(image_field):
    return cv2.imdecode(np.frombuffer(image_field["bytes"], np.uint8), cv2.IMREAD_COLOR)


def compute_metrics(y_true, probs):
    y_pred = probs.argmax(axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(NUM_CLASSES), zero_division=0
    )
    specificity = []
    for i in range(NUM_CLASSES):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        specificity.append(float(tn / (tn + fp)) if (tn + fp) else 0.0)
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "macro_auroc": float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro")),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_class": [
            {
                "name": CLASS_NAMES[i],
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "specificity": specificity[i],
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(NUM_CLASSES)
        ],
        "confusion_matrix": cm.tolist(),
    }


def save_confusion_matrix(cm, title, path):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.imshow(cm, cmap="Blues")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(NUM_CLASSES), CLASS_NAMES, rotation=35, ha="right")
    ax.set_yticks(range(NUM_CLASSES), CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_stage_grid(rows, path):
    fig, axes = plt.subplots(len(rows), 4, figsize=(15, 3.6 * len(rows)))
    for r, row in enumerate(rows):
        res = row["result"]
        correct = "correct" if res["grade"] == row["true_grade"] else "WRONG"
        axes[r, 0].imshow(cv2.cvtColor(row["bgr"], cv2.COLOR_BGR2RGB))
        axes[r, 0].set_title(f"Input — true: {CLASS_NAMES[row['true_grade']]}", fontsize=10)
        axes[r, 1].imshow(res["preprocessed_rgb"])
        axes[r, 1].set_title("Preprocessed (crop + CLAHE + Ben Graham)", fontsize=10)
        axes[r, 2].imshow(res["gradcam_overlay_rgb"])
        axes[r, 2].set_title(f"Grad-CAM — pred: {res['label']} ({correct})", fontsize=10)
        for c in range(3):
            axes[r, c].axis("off")

        probs = list(res["probabilities"].values())
        colors = ["#C44E52" if i == res["grade"] else "#4C72B0" for i in range(NUM_CLASSES)]
        axes[r, 3].barh(range(NUM_CLASSES), probs, color=colors)
        axes[r, 3].set_yticks(range(NUM_CLASSES), [n.split(" - ")[1] for n in CLASS_NAMES], fontsize=9)
        axes[r, 3].invert_yaxis()
        axes[r, 3].set_xlim(0, 1)
        axes[r, 3].set_title(f"Probabilities (ordinal score {res['ordinal_score']:.2f})", fontsize=10)
        for i, p in enumerate(probs):
            if p > 0.75:
                axes[r, 3].text(p - 0.02, i, f"{p:.2f}", va="center", ha="right", fontsize=8, color="white")
            else:
                axes[r, 3].text(p + 0.02, i, f"{p:.2f}", va="center", fontsize=8)
    fig.suptitle("HybridDRNet — one held-out validation image per DR stage", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=str(ROOT / "models" / "best_model.pth"))
    parser.add_argument("--cache", default=str(ROOT / ".cache" / "aptos.parquet"))
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    parser.add_argument("--samples-dir", default=str(ROOT / "samples"))
    parser.add_argument("--skip-eval", action="store_true", help="only run the per-stage predictions")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    samples_dir = Path(args.samples_dir)
    (results_dir / "gradcam").mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    predictor = DRPredictor(args.checkpoint)
    print(f"Loaded checkpoint {args.checkpoint} {predictor.metadata} on {predictor.device}")

    df = load_mirror(Path(args.cache))
    train_df, val_df = train_test_split(df, test_size=VAL_FRACTION, stratify=df["diagnosis"], random_state=SEED)
    print(f"Reproduced split: train={len(train_df)} val={len(val_df)} "
          f"val counts={val_df['diagnosis'].value_counts().sort_index().tolist()}")

    if not args.skip_eval:
        evaluation = {"note": "Images from a 224x224 public mirror; training used full resolution."}
        control = train_df.sample(n=len(val_df), random_state=SEED)
        for name, part in (("validation_heldout", val_df), ("training_control", control)):
            print(f"Evaluating {name} ({len(part)} images, TTA, CPU) ...")
            probs, _ = predictor.predict_probabilities([decode(img) for img in part["image"]])
            evaluation[name] = compute_metrics(part["diagnosis"].values, probs)
            m = evaluation[name]
            print(f"  acc={m['accuracy']:.4f} qwk={m['qwk']:.4f} auroc={m['macro_auroc']:.4f} macro_f1={m['macro_f1']:.4f}")
        save_confusion_matrix(
            evaluation["validation_heldout"]["confusion_matrix"],
            "Held-out validation (224px mirror) — acc {:.2%}, QWK {:.4f}".format(
                evaluation["validation_heldout"]["accuracy"], evaluation["validation_heldout"]["qwk"]),
            results_dir / "confusion_matrix_heldout.png",
        )
        (results_dir / "mirror_evaluation.json").write_text(json.dumps(evaluation, indent=2))

    # First validation image of each stage in split order — fixed in advance, not cherry-picked.
    rows, summary = [], []
    for grade in range(NUM_CLASSES):
        record = val_df[val_df["diagnosis"] == grade].iloc[0]
        bgr = decode(record["image"])
        sample_path = samples_dir / f"stage_{grade}_{CLASS_NAMES[grade].split(' - ')[1].replace(' ', '_').lower()}.png"
        cv2.imwrite(str(sample_path), bgr)

        result = predictor.predict(bgr)
        overlay_path = results_dir / "gradcam" / f"stage_{grade}_gradcam.png"
        cv2.imwrite(str(overlay_path), cv2.cvtColor(result["gradcam_overlay_rgb"], cv2.COLOR_RGB2BGR))

        rows.append({"bgr": bgr, "true_grade": grade, "result": result})
        summary.append({
            "sample": sample_path.relative_to(ROOT).as_posix(),
            "dataset_row": int(record.name),
            "true_grade": grade,
            "true_label": CLASS_NAMES[grade],
            "predicted_grade": result["grade"],
            "predicted_label": result["label"],
            "correct": result["grade"] == grade,
            "confidence": round(result["confidence"], 4),
            "ordinal_score": round(result["ordinal_score"], 4),
            "probabilities": {k: round(v, 4) for k, v in result["probabilities"].items()},
            "gradcam": overlay_path.relative_to(ROOT).as_posix(),
        })
        print(f"Stage {grade}: true={CLASS_NAMES[grade]:22s} pred={result['label']:22s} "
              f"conf={result['confidence']:.3f} ordinal={result['ordinal_score']:.2f}")

    save_stage_grid(rows, results_dir / "gradcam_all_stages.png")
    (results_dir / "sample_predictions.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved results to {results_dir}")


if __name__ == "__main__":
    main()
