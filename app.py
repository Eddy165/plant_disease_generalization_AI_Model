from pathlib import Path
import numpy as np
import streamlit as st
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from src.data import build_transforms
from src.models import build_model
from src.gradcam import GradCAM, get_target_layer
from src.utils import get_device

st.set_page_config(page_title="Plant Disease Research Demo", page_icon="🌿", layout="wide")
st.title("🌿 Plant Disease Classification")
st.caption("Research prototype — predictions are not a substitute for expert agricultural diagnosis.")

checkpoint_files = sorted(Path("artifacts").glob("*_best.pt"))
with st.sidebar:
    st.header("Model")
    if not checkpoint_files:
        st.warning("No trained checkpoint found.")
        st.code("python -m src.train --model mobilenet_v2 --epochs 8")
        st.stop()
    selected = st.selectbox("Checkpoint", checkpoint_files, format_func=lambda p: p.name)
    threshold = st.slider("Low-confidence warning threshold", 0.1, 0.95, 0.65, 0.05)
    st.caption("This threshold is a heuristic unless calibrated on held-out validation data.")

@st.cache_resource
def load_model(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["model_name"], ckpt["num_classes"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    device = get_device()
    model.to(device).eval()
    return model, ckpt, device

model, ckpt, device = load_model(str(selected))
uploaded = st.file_uploader("Upload a clear leaf image", type=["jpg", "jpeg", "png", "webp"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    left, right = st.columns(2)
    with left:
        st.image(image, caption="Uploaded image", use_container_width=True)
    transform = build_transforms(ckpt.get("image_size", 224), False)
    x = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        conf, idx = probs.max(0)
    label = ckpt["class_names"][int(idx)]
    with right:
        st.subheader("Prediction")
        st.metric("Predicted class", label)
        st.metric("Softmax confidence", f"{float(conf):.1%}")
        if float(conf) < threshold:
            st.warning("Low-confidence prediction. Do not rely on this result; seek expert review.")
        else:
            st.info("Confidence is above the selected threshold, but correctness is not guaranteed.")
        topv, topi = torch.topk(probs, min(5, len(probs)))
        st.write("Top predictions")
        st.dataframe([{"Class": ckpt["class_names"][int(i)], "Score": f"{float(v):.2%}"} for v, i in zip(topv, topi)],
                     use_container_width=True, hide_index=True)
    try:
        target = get_target_layer(model, ckpt["model_name"])
        cam_engine = GradCAM(model, target)
        cam, _ = cam_engine(x, int(idx))
        cam_engine.close()
        resized = image.resize((cam.shape[1], cam.shape[0]))
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(resized)
        ax.imshow(cam, cmap="jet", alpha=0.38)
        ax.axis("off"); ax.set_title("Grad-CAM explanation (qualitative)")
        st.pyplot(fig); plt.close(fig)
    except Exception as e:
        st.warning(f"Grad-CAM could not be generated for this model: {e}")
    st.caption("Grad-CAM highlights influential image regions but does not prove that the model used medically/agronomically valid evidence.")
