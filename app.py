from html import escape
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError

from src.data import build_transforms
from src.gradcam import GradCAM, get_target_layer
from src.models import build_model
from src.utils import get_device


# -----------------------------------------------------------------------------
# Page setup and visual system
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Fieldnote | Plant Health Research",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

CHECKPOINTS = sorted(Path("artifacts").glob("*_best.pt"))
DEFAULT_THRESHOLD = 0.65

# Streamlit reruns the app after the toggle changes, so the current mode can
# drive the CSS from the start of each rerun.
st.session_state.setdefault("dark_mode", False)
dark_mode = bool(st.session_state.get("dark_mode", False))

LIGHT_THEME_OVERRIDES = """
/* Explicitly keep upload and checkpoint controls light, readable surfaces. */
[data-testid="stFileUploader"],
[data-testid="stFileUploaderDropzone"],
section[data-testid="stFileUploaderDropzone"] {
  background: #ffffff !important;
  color: #26392d !important;
  border-color: #829985 !important;
}
[data-testid="stFileUploaderDropzoneInstructions"],
[data-testid="stFileUploaderDropzoneInstructions"] *,
[data-testid="stFileUploaderDropzone"] label,
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] p,
[data-testid="stFileUploaderDropzone"] svg {
  color: #26392d !important;
  fill: #26392d !important;
}
[data-testid="stFileUploaderDropzone"] button {
  color: #203e32 !important;
  background: #eaf0e8 !important;
  border: 1px solid #aab9a8 !important;
}
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
  background: #ffffff !important;
  border-color: #aab8aa !important;
  color: #202c24 !important;
}
[data-testid="stSelectbox"] div[data-baseweb="select"] *,
[data-baseweb="popover"], [data-baseweb="popover"] *,
[data-baseweb="menu"], [data-baseweb="menu"] *,
[role="listbox"], [role="listbox"] *, [role="option"] {
  color: #202c24 !important;
}
[data-baseweb="popover"], [data-baseweb="menu"], [role="listbox"] {
  background: #ffffff !important;
  border: 1px solid #d1dbd0 !important;
}
[role="option"] { background: #ffffff !important; }
[role="option"]:hover, [aria-selected="true"][role="option"] {
  background: #eaf0e8 !important;
}
"""

DARK_THEME_OVERRIDES = """
:root {
  --forest: #b9d8be;
  --forest-deep: #d8e9da;
  --paper: #151c17;
  --ink: #eef3ee;
  --muted: #c0ccc2;
  --line: #3c4a3f;
  --clay: #e6ad8d;
  --sidebar: #202a23;
  --soft-surface: #28352b;
  --white: #202b24;
}
.stApp, [data-testid="stAppViewContainer"] { background: #151c17 !important; color: #eef3ee !important; }
[data-testid="stHeader"] { background: rgba(21,28,23,.98) !important; }
[data-testid="stAppViewContainer"] .main .block-container { color: #eef3ee !important; }
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span,
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p,
[data-testid="stText"],
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label,
label, legend, small, p, li { color: #e6eee7 !important; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p,
.small-note, .hero-copy, .footer-line { color: #c0ccc2 !important; }
h1, h2, h3, h4, .brand-name, .brand-subtitle, .section-label,
.result-name, .result-label, .eyebrow { color: #c5dec9 !important; }
strong, b { color: #f1f6f1 !important; }
[data-testid="stSidebar"] { background: #202a23 !important; border-right-color: #3c4a3f !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] small { color: #e6eee7 !important; }
/* Upload target stays visibly distinct from the canvas in either mode. */
[data-testid="stFileUploader"],
[data-testid="stFileUploaderDropzone"],
section[data-testid="stFileUploaderDropzone"] {
  background: #29372d !important;
  color: #f0f5f0 !important;
  border: 1.5px dashed #94b59a !important;
  border-radius: 12px !important;
}
[data-testid="stFileUploaderDropzoneInstructions"],
[data-testid="stFileUploaderDropzoneInstructions"] *,
[data-testid="stFileUploaderDropzone"] label,
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] p,
[data-testid="stFileUploaderDropzone"] svg { color: #f0f5f0 !important; fill: #f0f5f0 !important; }
[data-testid="stFileUploaderDropzone"] button {
  background: #e0eadd !important; color: #1b3023 !important;
  border: 1px solid #abc0ac !important;
}
[data-testid="stFileUploaderDropzone"] button *,
[data-testid="stFileUploaderDropzone"] button p { color: #1b3023 !important; }
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
  background: #29372d !important; border-color: #637868 !important; color: #f0f5f0 !important;
}
[data-testid="stSelectbox"] div[data-baseweb="select"] *,
[data-baseweb="popover"], [data-baseweb="popover"] *,
[data-baseweb="menu"], [data-baseweb="menu"] *,
[role="listbox"], [role="listbox"] *, [role="option"] { color: #f0f5f0 !important; }
[data-baseweb="popover"], [data-baseweb="menu"], [role="listbox"] {
  background: #253128 !important; border: 1px solid #526554 !important;
}
[role="option"] { background: #253128 !important; }
[role="option"]:hover, [aria-selected="true"][role="option"] { background: #3a4d3e !important; }
input, textarea { background: #29372d !important; color: #f0f5f0 !important; }
[data-testid="stMetric"] { background: #202b24 !important; border-color: #3c4a3f !important; }
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] *, [data-testid="stMetricDelta"] { color: #c0ccc2 !important; }
[data-testid="stMetricValue"], [data-testid="stMetricValue"] * { color: #c5dec9 !important; }
[data-testid="stAlert"] { background: #28352b !important; border-color: #546856 !important; }
[data-testid="stAlert"] p, [data-testid="stAlert"] span, [data-testid="stAlert"] div { color: #eef3ee !important; }
[data-testid="stExpander"] { background: #202b24 !important; border-color: #3c4a3f !important; }
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary span,
[data-testid="stExpander"] svg { color: #d9e9dc !important; }
.panel { background: #202b24 !important; border-color: #3c4a3f !important; box-shadow: 0 3px 14px rgba(0,0,0,.12) !important; }
.hero-note { background: #29352b !important; color: #e6eee7 !important; border-left-color: #d29b79 !important; }
.hero-note b, .hero-note strong { color: #f1f6f1 !important; }
.small-note, .small-note b, .small-note strong { color: #c0ccc2 !important; }
.result-label { color: #c0ccc2 !important; }
.result-name { color: #d8e9da !important; }
.footer-line { border-top-color: #3c4a3f !important; color: #c0ccc2 !important; }
.footer-line b, .footer-line strong { color: #d8e9da !important; }
.eyebrow, .section-label { color: #e6ad8d !important; }
.brand-mark { background: #42634a !important; color: #ffffff !important; }
.brand-subtitle { color: #c0ccc2 !important; }
.stButton > button { background: #41694b !important; border-color: #6f9475 !important; color: #ffffff !important; }
.stButton > button *, .stButton > button p { color: #ffffff !important; }
[data-testid="stProgress"] > div { background: #354438 !important; }
[data-testid="stProgress"] > div > div > div > div { background: #9fc6a5 !important; }
hr { border-color: #3c4a3f !important; }
/* Override inline score colours used by result rows. */
div[style*="color:#173d2d"], div[style*="color: #173d2d"] { color: #c5dec9 !important; }
"""

THEME_OVERRIDES = DARK_THEME_OVERRIDES if dark_mode else LIGHT_THEME_OVERRIDES

BASE_CSS = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Serif+Display&display=swap');
    :root {
      --forest: #203e32;
      --forest-deep: #172e25;
      --paper: #f7f6f0;
      --ink: #202c24;
      --muted: #4b5a50;
      --line: #cbd4c8;
      --clay: #85472e;
      --sidebar: #e8ece3;
      --soft-surface: #f0f2eb;
      --white: #ffffff;
    }

    /* Base canvas and default typography */
    html, body, [class*="css"], .stApp {
      font-family: 'DM Sans', sans-serif;
      color: var(--ink);
    }
    .stApp { background: var(--paper); color: var(--ink); }
    [data-testid="stHeader"] { background: rgba(247,246,240,.98); }
    [data-testid="stAppViewContainer"] .main .block-container {
      max-width: 1380px;
      padding-top: 2rem;
      padding-bottom: 4rem;
    }

    /* Make Streamlit's generated text readable even when the browser theme differs */
    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] span,
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] p,
    [data-testid="stText"],
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] label,
    label, legend, small {
      color: var(--ink) !important;
    }
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] p {
      color: var(--muted) !important;
      font-size: .86rem !important;
      line-height: 1.55 !important;
    }
    [data-testid="stMarkdownContainer"] a { color: #1f6047 !important; }
    h1, h2, h3, h4 {
      color: var(--forest) !important;
      letter-spacing: -.025em !important;
    }
    h1 {
      font-family: 'DM Serif Display', Georgia, serif !important;
      font-size: clamp(2.7rem,5vw,4.4rem) !important;
      font-weight: 400 !important;
      line-height: 1.06 !important;
    }
    h2 { font-family: 'DM Serif Display', Georgia, serif !important; font-weight: 400 !important; }
    h3 { font-size: 1.12rem !important; font-weight: 700 !important; }
    strong, b { color: #1d3327 !important; font-weight: 700 !important; }

    /* Sidebar */
    [data-testid="stSidebar"] {
      background: var(--sidebar) !important;
      border-right: 1px solid #c5cfc2;
    }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.4rem; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] small {
      color: #26392d !important;
    }

    /* Inputs and upload area */
    [data-testid="stFileUploader"] {
      background: #fff !important;
      border: 1.5px dashed #718b78 !important;
      border-radius: 12px;
      padding: .8rem;
    }
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] p {
      color: #26392d !important;
    }
    [data-testid="stImage"] img { border-radius: 12px; }
    [data-testid="stSelectbox"] label,
    [data-testid="stSlider"] label { color: #26392d !important; font-weight: 600 !important; }
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
      background: #fff !important;
      border-color: #aab8aa !important;
      color: #202c24 !important;
    }
    [data-testid="stSelectbox"] div[data-baseweb="select"] *,
    [data-baseweb="popover"] *,
    [role="listbox"] *,
    [role="option"] {
      color: #202c24 !important;
    }
    input, textarea { color: #202c24 !important; }

    /* Metrics / alerts / expander surfaces */
    [data-testid="stMetric"] {
      background: #fff !important;
      border: 1px solid #cbd4c8;
      border-radius: 12px;
      padding: 1rem;
    }
    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] *,
    [data-testid="stMetricDelta"] {
      color: #4b5a50 !important;
    }
    [data-testid="stMetricValue"], [data-testid="stMetricValue"] * {
      color: #203e32 !important;
      font-size: 1.7rem;
      font-weight: 700 !important;
    }
    [data-testid="stAlert"] { border-radius: 10px; border: 1px solid #c4cec3; }
    [data-testid="stAlert"] p,
    [data-testid="stAlert"] span,
    [data-testid="stAlert"] div { color: #27372d !important; }
    [data-testid="stExpander"] {
      background: #fff !important;
      border: 1px solid #cbd4c8 !important;
      border-radius: 10px !important;
    }
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary span,
    [data-testid="stExpander"] svg { color: #203e32 !important; }
    [data-testid="stDataFrame"] { border: 1px solid #cbd4c8; }

    /* Brand and editorial elements */
    .brand { display: flex; align-items: center; gap: 10px; margin-bottom: 2.2rem; }
    .brand-mark {
      display: flex; width: 39px; height: 39px; align-items: center;
      justify-content: center; border-radius: 11px; background: var(--forest);
      color: white !important; font-size: 21px;
    }
    .brand-name { font-size: 1.08rem; font-weight: 700; letter-spacing: -.04em; color: var(--forest) !important; }
    .brand-subtitle { margin-top: 2px; font-size: .7rem; color: #46584b !important; letter-spacing: .09em; text-transform: uppercase; }
    .eyebrow, .section-label {
      text-transform: uppercase; font-size: .75rem; letter-spacing: .13em;
      font-weight: 700; margin-bottom: .8rem;
    }
    .eyebrow { color: var(--clay) !important; }
    .section-label { color: #46584b !important; margin-top: .3rem; }
    .hero-copy {
      color: #46554b !important; max-width: 700px; font-size: 1.06rem;
      line-height: 1.85; font-weight: 400;
    }
    .hero-note {
      margin-top: 1.3rem; padding: .95rem 1.05rem;
      border-left: 4px solid #9b5b3d; background: #eceee5;
      color: #27372d !important; font-size: .9rem; line-height: 1.75;
      border-radius: 0 8px 8px 0;
    }
    .hero-note b, .hero-note strong { color: #203e32 !important; }
    .panel {
      background: #fff !important; border: 1px solid #cbd4c8;
      border-radius: 14px; padding: 1.25rem 1.35rem;
      box-shadow: 0 3px 12px rgba(28, 51, 37, .035);
    }
    .result-label {
      color: #4b5a50 !important; font-size: .8rem;
      text-transform: uppercase; letter-spacing: .09em; font-weight: 700;
    }
    .result-name {
      color: #203e32 !important; font-family: 'DM Serif Display', Georgia, serif;
      font-size: clamp(1.7rem,3vw,2.5rem); line-height: 1.2;
      margin: .55rem 0 1rem; overflow-wrap: anywhere;
    }
    .small-note {
      color: #46554b !important; font-size: .86rem; line-height: 1.75;
    }
    .small-note b, .small-note strong { color: #203e32 !important; }
    .footer-line {
      border-top: 1px solid #cbd4c8; margin-top: 3rem; padding-top: 1rem;
      color: #4b5a50 !important; font-size: .8rem; line-height: 1.8;
    }
    .footer-line b, .footer-line strong { color: #203e32 !important; }

    /* Buttons and progress indicators */
    .stButton > button {
      border-radius: 8px; border: 1px solid var(--forest) !important;
      background: var(--forest) !important; color: #fff !important; font-weight: 700;
    }
    .stButton > button *, .stButton > button p { color: #fff !important; }
    .stButton > button:hover { border-color: var(--forest-deep) !important; background: var(--forest-deep) !important; }
    [data-testid="stProgress"] > div { background: #e0e7dc !important; }
    [data-testid="stProgress"] > div > div > div > div { background: var(--forest) !important; }
    hr { border-color: #cbd4c8 !important; }

    @media (max-width: 760px) {
      [data-testid="stAppViewContainer"] .main .block-container { padding: 1.1rem 1rem 2rem; }
      h1 { font-size: 2.65rem !important; }
      .hero-copy { font-size: .98rem; }
      .panel { padding: 1rem; }
    }
    </style>
    """
st.markdown(
    BASE_CSS.replace("</style>", THEME_OVERRIDES + "\n</style>"),
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Model and prediction helpers
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the selected model…")
def load_model(checkpoint_path: str):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    model = build_model(
        checkpoint["model_name"],
        checkpoint["num_classes"],
        pretrained=False,
    )
    model.load_state_dict(checkpoint["state_dict"])
    device = get_device()
    model.to(device).eval()
    return model, checkpoint, device


def readable_class_name(raw_name: str) -> str:
    """Improve display readability while retaining the original class meaning."""
    name = str(raw_name).replace("___", " — ").replace("__", " — ")
    return " ".join(name.replace("_", " ").split())


def predict(model, image, checkpoint, device):
    transform = build_transforms(checkpoint.get("image_size", 224), False)
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    confidence, predicted_index = probabilities.max(dim=0)
    top_values, top_indices = torch.topk(probabilities, k=min(5, len(probabilities)))
    return tensor, int(predicted_index), float(confidence), top_values, top_indices


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="brand">
          <div class="brand-mark">✳</div>
          <div>
            <div class="brand-name">Fieldnote</div>
            <div class="brand-subtitle">Plant health research</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.toggle(
        "Dark mode",
        key="dark_mode",
        help="Switch between the light botanical palette and the dark research palette.",
    )
    st.caption("Choose a comfortable display theme.")
    st.markdown('<div class="section-label">Analysis setup</div>', unsafe_allow_html=True)

    if not CHECKPOINTS:
        st.warning("No trained checkpoint was found in artifacts/.")
        st.code("python -m src.train --model mobilenet_v2 --epochs 8", language="powershell")
        st.stop()

    selected = st.selectbox(
        "Model checkpoint",
        CHECKPOINTS,
        format_func=lambda path: path.stem.replace("_best", "").replace("_", " ").title(),
    )
    threshold = st.slider(
        "Low-confidence warning",
        min_value=0.10,
        max_value=0.95,
        value=DEFAULT_THRESHOLD,
        step=0.05,
        help="A review heuristic, not a calibrated probability guarantee.",
    )
    st.caption("Changing this threshold does not change model accuracy.")
    st.divider()
    st.markdown('<div class="section-label">Workflow</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="small-note"><b>01</b> Upload a leaf photograph<br><br>'
        '<b>02</b> Review model predictions<br><br>'
        '<b>03</b> Inspect the Grad-CAM heatmap<br><br>'
        '<b>04</b> Verify with an agricultural expert</p>',
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown(
        '<p class="small-note">Research prototype · Not a field diagnosis system</p>',
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Main page
# -----------------------------------------------------------------------------
st.markdown('<div class="eyebrow">Plant pathology · Image analysis</div>', unsafe_allow_html=True)
st.title("Look closer at plant health.")
st.markdown(
    '<p class="hero-copy">Examine a leaf photograph with a trained image-classification '
    'model. Review its leading prediction, compare alternative classes, and inspect '
    'the image regions highlighted by Grad-CAM.</p>'
    '<div class="hero-note"><b>Research note</b> &nbsp; Results depend on training data '
    'and image conditions. A photograph alone may not contain enough evidence for a '
    'reliable diagnosis.</div>',
    unsafe_allow_html=True,
)
st.write("")
st.markdown('<div class="section-label">01 / Leaf specimen</div>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Choose a clear photograph of one leaf",
    type=["jpg", "jpeg", "png", "webp"],
    help="Use a sharp image, preferably with one leaf clearly visible.",
)

if uploaded is None:
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.markdown(
            '<div class="panel"><div class="eyebrow">Getting started</div>'
            '<h3>Start with a clear specimen</h3><p class="small-note">Photograph one '
            'leaf in natural light. Keep the affected area in focus, avoid heavy filters, '
            'and include enough of the leaf for the model to assess visible features.</p>'
            '<p class="small-note">Supported formats: JPG, JPEG, PNG and WEBP.</p></div>',
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            '<div class="panel"><div class="eyebrow">What you will see</div>'
            '<h3>A structured model assessment</h3><p class="small-note"><b>Predicted class</b> '
            '— the class with the highest model score.</p><p class="small-note">'
            '<b>Alternative classes</b> — other classes with high scores.</p><p class="small-note">'
            '<b>Grad-CAM</b> — a qualitative view of image regions associated with the prediction.</p></div>',
            unsafe_allow_html=True,
        )
else:
    try:
        image = Image.open(uploaded).convert("RGB")
    except (UnidentifiedImageError, OSError):
        st.error("This image could not be opened. Try another JPG, PNG, or WEBP file.")
        st.stop()

    model, checkpoint, device = load_model(str(selected))
    with st.spinner("Analysing the leaf image…"):
        tensor, predicted_index, confidence, top_values, top_indices = predict(
            model, image, checkpoint, device
        )

    class_names = checkpoint["class_names"]
    label = readable_class_name(class_names[predicted_index])
    input_size = checkpoint.get("image_size", 224)

    st.caption(
        f"Model: {checkpoint.get('model_name', selected.stem)}  ·  "
        f"Input: {input_size} × {input_size}  ·  Device: {device}"
    )
    st.write("")
    st.markdown('<div class="section-label">02 / Model assessment</div>', unsafe_allow_html=True)
    image_col, result_col = st.columns([1.02, 1], gap="large")

    with image_col:
        st.image(image, caption="Uploaded leaf specimen", use_container_width=True)
        st.markdown(
            f'<p class="small-note"><b>Image dimensions</b> &nbsp; '
            f'{image.width} × {image.height} px<br><b>Colour mode</b> &nbsp; RGB</p>',
            unsafe_allow_html=True,
        )

    with result_col:
        st.markdown(
            f'<div class="panel"><div class="result-label">Leading prediction</div>'
            f'<div class="result-name">{escape(label)}</div>'
            f'<div class="result-label">Softmax score</div>'
            f'<div style="font-size:2.1rem;font-weight:700;color:#173d2d;margin:.25rem 0 .8rem">'
            f'{confidence:.1%}</div><div class="small-note">This is the model score for its '
            f'leading class, not the probability that the diagnosis is correct.</div></div>',
            unsafe_allow_html=True,
        )
        st.write("")
        if confidence < threshold:
            st.warning(
                "Review recommended: the leading score is below your selected threshold. "
                "Do not rely on this prediction alone."
            )
        else:
            st.info(
                "The leading score is above your selected threshold. This does not establish "
                "that the prediction is correct."
            )
        st.metric("Classes available to model", len(class_names))

    st.write("")
    st.markdown('<div class="section-label">03 / Alternative classes</div>', unsafe_allow_html=True)
    st.markdown("### What else did the model consider?")
    st.caption("Scores help compare model outputs; they are not independently calibrated probabilities.")
    top_col, info_col = st.columns([1.25, 0.75], gap="large")
    with top_col:
        for rank, (score_tensor, index_tensor) in enumerate(zip(top_values, top_indices), start=1):
            score = float(score_tensor)
            class_label = readable_class_name(class_names[int(index_tensor)])
            row_left, row_right = st.columns([3, 1])
            with row_left:
                st.markdown(f"**{rank:02d}** &nbsp; {escape(class_label)}")
            with row_right:
                st.markdown(
                    f"<div style='text-align:right;color:#173d2d;font-weight:700;font-size:1rem'>{score:.1%}</div>",
                    unsafe_allow_html=True,
                )
            st.progress(min(max(score, 0.0), 1.0))
    with info_col:
        st.markdown(
            '<div class="panel"><div class="eyebrow">How to read this</div>'
            '<h3>Compare, don’t assume</h3><p class="small-note">Closely scored classes may '
            'indicate difficulty distinguishing visual patterns.</p><p class="small-note">'
            'The real condition may not be represented in the model’s available classes.</p></div>',
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown('<div class="section-label">04 / Visual explanation</div>', unsafe_allow_html=True)
    st.markdown("### Where did the model focus?")
    st.caption(
        "Grad-CAM is a qualitative visualization. It does not prove that the model relied "
        "on valid disease symptoms."
    )
    try:
        target_layer = get_target_layer(model, checkpoint["model_name"])
        cam_engine = GradCAM(model, target_layer)
        try:
            cam, _ = cam_engine(tensor, predicted_index)
        finally:
            cam_engine.close()

        figure, axis = plt.subplots(figsize=(8, 5.5))
        figure.patch.set_facecolor("#202b24" if dark_mode else "#ffffff")
        axis.imshow(image.resize((cam.shape[1], cam.shape[0])))
        axis.imshow(cam, cmap="YlOrRd", alpha=0.42, vmin=0, vmax=1)
        axis.axis("off")
        figure.tight_layout(pad=0)
        st.pyplot(figure, use_container_width=True)
        plt.close(figure)
        st.markdown(
            '<p class="small-note">Warmer areas contribute more strongly to this heatmap. '
            'They are not a disease boundary or severity measurement.</p>',
            unsafe_allow_html=True,
        )
    except Exception as exc:
        st.warning(
            "The prediction was generated, but a Grad-CAM explanation could not be created. "
            f"Details: {type(exc).__name__}."
        )

    with st.expander("Model and analysis details"):
        st.write(f"**Checkpoint:** `{selected.name}`")
        st.write(f"**Architecture:** `{checkpoint.get('model_name', 'Unknown')}`")
        st.write(f"**Image size:** `{input_size}`")
        st.write(f"**Number of classes:** `{len(class_names)}`")
        st.write(f"**Inference device:** `{device}`")
        st.write(f"**Warning threshold:** `{threshold:.0%}`")
        st.caption("Calibrate warning thresholds on held-out validation data before making research claims.")

st.markdown(
    '<div class="footer-line"><b>FIELDNOTE</b> &nbsp; / &nbsp; Plant Health Research<br>'
    'An educational image-classification interface. Predictions and visual explanations '
    'require independent verification and should not be treated as a confirmed diagnosis.</div>',
    unsafe_allow_html=True,
)
