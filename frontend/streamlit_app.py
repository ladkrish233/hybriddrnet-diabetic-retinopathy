"""Streamlit UI for the HybridDRNet FastAPI backend.

Run from the repository root (with the backend already running):
    streamlit run frontend/streamlit_app.py
"""

import base64
import os
from pathlib import Path

import altair as alt
import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = ROOT / "samples"
DEFAULT_BACKEND = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="HybridDRNet — DR Grading", page_icon="👁️", layout="wide")


@st.cache_data(ttl=30, show_spinner=False)
def fetch_model_info(backend_url):
    response = requests.get(f"{backend_url}/model-info", timeout=10)
    response.raise_for_status()
    return response.json()


def call_predict(backend_url, filename, data, mime):
    response = requests.post(
        f"{backend_url}/predict",
        files={"file": (filename, data, mime)},
        params={"gradcam": "true"},
        timeout=180,
    )
    if response.status_code != 200:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise RuntimeError(f"Backend returned {response.status_code}: {detail}")
    return response.json()


def b64_image(value):
    return base64.b64decode(value)


with st.sidebar:
    st.header("Backend")
    backend_url = st.text_input("FastAPI URL", DEFAULT_BACKEND).rstrip("/")
    try:
        info = fetch_model_info(backend_url)
        st.success("Connected")
        st.caption(info["architecture"])
        st.write(f"Checkpoint epoch: **{info['checkpoint_epoch']}**")
        if info["checkpoint_val_qwk"] is not None:
            st.write(f"Validation QWK: **{info['checkpoint_val_qwk']:.4f}**")
        st.write(f"Device: `{info['device']}`")
    except requests.RequestException as exc:
        st.error(f"Cannot reach backend at {backend_url}")
        st.caption(str(exc))
        info = None

st.title("Diabetic Retinopathy Grading")
st.write(
    "Upload a retinal fundus photograph. The model grades it into one of five DR stages "
    "and shows a Grad-CAM heatmap of the regions that drove the prediction."
)
st.warning(
    "Research prototype only — not a medical device and not a substitute for "
    "examination by a qualified ophthalmologist."
)

samples = sorted(SAMPLES_DIR.glob("*.png"))
source = st.radio("Image source", ["Upload my own", "Use a sample image"], horizontal=True,
                  index=0 if not samples else 1)

image_name = image_bytes = mime = None
if source == "Upload my own":
    uploaded = st.file_uploader("Fundus image", type=["png", "jpg", "jpeg"])
    if uploaded is not None:
        image_name, image_bytes, mime = uploaded.name, uploaded.getvalue(), uploaded.type
else:
    if samples:
        choice = st.selectbox("Sample (held-out validation images, one per stage)", [p.name for p in samples])
        path = SAMPLES_DIR / choice
        image_name, image_bytes, mime = path.name, path.read_bytes(), "image/png"
    else:
        st.info("No sample images found in samples/ — run scripts/run_sample_predictions.py first.")

analyze = st.button("Analyze", type="primary", disabled=image_bytes is None or info is None)

if analyze:
    with st.spinner("Running HybridDRNet with test-time augmentation and Grad-CAM..."):
        try:
            result = call_predict(backend_url, image_name, image_bytes, mime)
        except (requests.RequestException, RuntimeError) as exc:
            st.error(str(exc))
            st.stop()

    st.subheader(f"Predicted: {result['label']}")
    m1, m2, m3 = st.columns(3)
    m1.metric("Confidence", f"{result['confidence']:.1%}")
    m2.metric("Ordinal severity score", f"{result['ordinal_score']:.2f}", help="Continuous 0–4 output of the ordinal-regression head")
    m3.metric("Grade", result["grade"])

    c1, c2, c3 = st.columns(3)
    c1.image(image_bytes, caption="Input", width="stretch")
    c2.image(b64_image(result["preprocessed_png_base64"]), caption="Preprocessed (crop + CLAHE + Ben Graham)",
             width="stretch")
    if result.get("gradcam_png_base64"):
        c3.image(b64_image(result["gradcam_png_base64"]), caption=f"Grad-CAM for '{result['label']}'",
                 width="stretch")

    st.subheader("Class probabilities")
    probs = pd.DataFrame({"stage": list(result["probabilities"]), "probability": list(result["probabilities"].values())})
    probs["predicted"] = probs["stage"] == result["label"]
    chart = (
        alt.Chart(probs)
        .mark_bar()
        .encode(
            x=alt.X("probability:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
            y=alt.Y("stage:N", sort=None, title=None),
            color=alt.condition("datum.predicted", alt.value("#C44E52"), alt.value("#4C72B0")),
            tooltip=["stage", alt.Tooltip("probability:Q", format=".1%")],
        )
        .properties(height=220)
    )
    st.altair_chart(chart, width="stretch")
    st.caption(result["disclaimer"])
