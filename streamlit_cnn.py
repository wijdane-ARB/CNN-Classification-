"""
Image classification app (Streamlit) - works with ANY Keras image classifier.

Nothing is hard-coded about the dataset:
  * input size        -> read from the model (model.input_shape)
  * class names       -> read from "<model_name>_classes.json" saved at training time
                         (or typed in the sidebar as a fallback)
  * pixel scaling     -> auto-detected (does the model already contain Rescaling?)

Run:  streamlit run streamlit_cnn.py
"""
import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

st.set_page_config(page_title="Image classifier", page_icon="🍎", layout="centered")

APP_DIR = Path(__file__).parent


# --------------------------------------------------------------------------- #
# Loading helpers
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading model...")
def load_model(path: str):
    return tf.keras.models.load_model(path)


def iter_layers(layer):
    """Yield a layer and, recursively, every layer nested inside it."""
    yield layer
    for sub in getattr(layer, "layers", []):
        yield from iter_layers(sub)


def model_has_scaling(model) -> bool:
    """True if the model already normalises pixels itself (Rescaling/Normalization/preprocess)."""
    for layer in iter_layers(model):
        name = layer.name.lower()
        if isinstance(layer, (tf.keras.layers.Rescaling, tf.keras.layers.Normalization)):
            return True
        if "preprocess" in name or "rescaling" in name:
            return True
    return False


def read_meta(model_path: Path) -> dict:
    """Optional extras stored in <model>_classes.json: 'label' (name shown in the menu), 'description'."""
    f = model_path.with_name(model_path.stem + "_classes.json")
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def load_class_names(model_path: Path, n_outputs: int):
    """Look for <model>_classes.json, then classes.json / class_names.json."""
    candidates = [
        model_path.with_name(model_path.stem + "_classes.json"),
        model_path.with_name("classes.json"),
        model_path.with_name("class_names.json"),
    ]
    for c in candidates:
        if c.exists():
            data = json.loads(c.read_text(encoding="utf-8"))
            names = data["class_names"] if isinstance(data, dict) else data
            return list(names), c.name
    return [f"class_{i}" for i in range(n_outputs)], None


# --------------------------------------------------------------------------- #
# Pre-processing (must mirror training)
# --------------------------------------------------------------------------- #
def to_rgb(image: Image.Image) -> Image.Image:
    """Fix EXIF rotation and flatten transparency onto WHITE (a plain
    .convert('RGB') turns transparent pixels black or into a random palette colour)."""
    image = ImageOps.exif_transpose(image)
    if image.mode in ("P", "LA", "PA") or "transparency" in image.info:
        image = image.convert("RGBA")
    if image.mode == "RGBA":
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        background.alpha_composite(image)
        image = background
    return image.convert("RGB")


def preprocess(image: Image.Image, size, divide_by_255: bool) -> np.ndarray:
    h, w = size
    arr = np.asarray(to_rgb(image), dtype="float32")
    # same resize op as tf.keras.utils.image_dataset_from_directory (bilinear)
    arr = tf.image.resize(arr, (h, w), method="bilinear").numpy()
    if divide_by_255:
        arr = arr / 255.0
    return np.expand_dims(arr, axis=0)


# --------------------------------------------------------------------------- #
# Sidebar: model + classes
# --------------------------------------------------------------------------- #
st.sidebar.header("Model")

found = sorted(p.name for p in APP_DIR.glob("*.keras") if not p.name.startswith("_"))
labels = {n: read_meta(APP_DIR / n).get("label", Path(n).stem) for n in found}
found.sort(key=lambda n: labels[n].lower())
uploaded_model = st.sidebar.file_uploader("...or upload a .keras model", type=["keras"])

if uploaded_model is not None:
    model_path = APP_DIR / "_uploaded_model.keras"
    model_path.write_bytes(uploaded_model.getbuffer())
    label_stem = Path(uploaded_model.name).stem
    meta = {}
elif found:
    default_idx = next((i for i, n in enumerate(found) if read_meta(APP_DIR / n).get("default")), 0)
    choice = st.sidebar.selectbox("Choose a model", found, index=default_idx, format_func=lambda n: labels[n])
    model_path = APP_DIR / choice
    label_stem = model_path.stem
    meta = read_meta(model_path)
else:
    st.error("No .keras model found next to this script. Upload one in the sidebar.")
    st.stop()

model = load_model(str(model_path))
input_shape = model.input_shape  # (None, H, W, C)
img_h, img_w = int(input_shape[1]), int(input_shape[2])
n_outputs = int(model.output_shape[-1])

default_names, source = load_class_names(
    model_path if uploaded_model is None else APP_DIR / f"{label_stem}.keras", n_outputs
)

names_text = st.sidebar.text_input(
    "Class names (same order as training, comma-separated)",
    value=", ".join(default_names),
    help="Auto-filled from <model>_classes.json when that file exists. "
         "Keras sorts class folders alphabetically.",
)
class_names = [c.strip() for c in names_text.split(",") if c.strip()]
if len(class_names) != n_outputs:
    st.sidebar.error(f"The model has {n_outputs} outputs but you gave {len(class_names)} names.")
    st.stop()
st.sidebar.caption(
    f"Class names loaded from `{source}`" if source else "No *_classes.json found - using the names above."
)

scaling_mode = st.sidebar.radio(
    "Pixel scaling",
    ["Auto", "Model already rescales (feed 0-255)", "Divide by 255 here"],
    help="Your notebook model has a Rescaling(1/255) layer inside, so dividing "
         "by 255 again in the app destroys accuracy.",
)
if scaling_mode == "Auto":
    divide = not model_has_scaling(model)
elif scaling_mode.startswith("Model"):
    divide = False
else:
    divide = True

top_k = st.sidebar.slider("Top-k to display", 1, min(5, n_outputs), min(3, n_outputs))
threshold = st.sidebar.slider("Low-confidence warning below (%)", 0, 100, 60)

st.sidebar.markdown(
    f"**Input size:** {img_h}x{img_w}  \n**Classes:** {n_outputs}  \n"
    f"**Scaling in app:** {'/255' if divide else 'none (model handles it)'}"
)

# --------------------------------------------------------------------------- #
# Main page
# --------------------------------------------------------------------------- #
st.title("Image classification")
if meta.get("description"):
    st.caption(meta["description"])
if source is None and uploaded_model is None:
    st.warning(f"No `{model_path.stem}_classes.json` found next to the model, so generic names "
               "(class_0, class_1, ...) are used. Copy that file next to this script, or type the "
               "names in the sidebar, e.g. `apple, banana, orange`.")
st.write("Upload one or more images. Classes: " + ", ".join(f"**{c}**" for c in class_names))

files = st.file_uploader(
    "Upload images", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True
)

if not files:
    st.info("Please upload an image to classify.")
    st.stop()

for f in files:
    st.divider()
    image = Image.open(f)
    col_img, col_res = st.columns([1, 1.2])

    with col_img:
        st.image(to_rgb(image), caption=f.name, use_container_width=True)
        with st.expander(f"What the model sees ({img_h}x{img_w})"):
            small = preprocess(image, (img_h, img_w), divide_by_255=False)[0].astype("uint8")
            st.image(small, width=160)

    probs = model.predict(preprocess(image, (img_h, img_w), divide), verbose=0)[0]
    order = np.argsort(probs)[::-1]
    best = int(order[0])
    conf = float(probs[best]) * 100

    with col_res:
        st.subheader(class_names[best])
        st.metric("Confidence", f"{conf:.1f} %")
        if conf < threshold:
            st.warning("Low confidence - the image may be unlike the training data.")
        st.bar_chart({class_names[i]: float(probs[i]) * 100 for i in order[:top_k]}, horizontal=True)
