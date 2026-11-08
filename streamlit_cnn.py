"""
Image classification app (Streamlit) - works with ANY Keras image classifier, plus a project summary.

Fusion of the generic classifier app and the summary interface:
  * nothing is hard-coded about the dataset: input size read from the model, class names read from
    "<model>_classes.json" (or typed in the sidebar), pixel scaling auto-detected
  * pick a model (initial CNN, improved CNN, fruit model...) or upload your own .keras file
  * tabs: Overview | Predict | Test results | Curves | Model details | Compare models

Run:  streamlit run streamlit_cnn.py
"""
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

st.set_page_config(page_title="CNN image classification - summary", page_icon="🧠", layout="wide")

APP_DIR = Path(__file__).parent
RES_DIR = APP_DIR / "results"
SAMPLE_DIR = APP_DIR / "samples"

# colour follows the model (validated categorical slots: blue, orange, aqua, violet)
COLORS = {
    "cnn_intel_modele1": "#2a78d6",
    "cnn_intel_modele2": "#eb6834",
    "cnn_intel_modele3_mobilenetv2": "#1baf7a",
    "fruits_mobilenetv2": "#6250d6",
}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.facecolor": "none", "axes.facecolor": "none", "savefig.facecolor": "none",
})

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; max-width: 1250px;}
      div[data-testid="stMetric"] {background:#f1f0ed; padding:.8rem 1rem; border-radius:10px;}
      .pill {display:inline-block; padding:.15rem .6rem; border-radius:999px; font-size:.8rem;
             background:#f1f0ed; color:#52514e; margin-right:.3rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def show_image(img, **kw):
    """Version-tolerant st.image that fills the column."""
    try:
        st.image(img, width="stretch", **kw)
    except Exception:
        st.image(img, use_container_width=True, **kw)


@st.cache_resource(show_spinner="Loading model...")
def load_model(path: str):
    return tf.keras.models.load_model(path, compile=False)


def iter_layers(layer):
    yield layer
    for sub in getattr(layer, "layers", []):
        yield from iter_layers(sub)


def model_has_scaling(model) -> bool:
    for layer in iter_layers(model):
        if isinstance(layer, (tf.keras.layers.Rescaling, tf.keras.layers.Normalization)):
            return True
        if "preprocess" in layer.name.lower() or "rescaling" in layer.name.lower():
            return True
    return False


def read_meta(model_path: Path) -> dict:
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


def to_rgb(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image)
    if image.mode in ("P", "LA", "PA") or "transparency" in image.info:
        image = image.convert("RGBA")
    if image.mode == "RGBA":
        bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
        bg.alpha_composite(image)
        image = bg
    return image.convert("RGB")


def preprocess(image: Image.Image, size, divide_by_255: bool) -> np.ndarray:
    arr = np.asarray(to_rgb(image), dtype="float32")
    arr = tf.image.resize(arr, size, method="bilinear").numpy()
    if divide_by_255:
        arr = arr / 255.0
    return np.expand_dims(arr, axis=0)


def predict(model, image: Image.Image, divide=None) -> np.ndarray:
    """divide=None -> auto-detect (divide by 255 only if the model has no Rescaling layer)."""
    h, w = int(model.input_shape[1]), int(model.input_shape[2])
    if divide is None:
        divide = not model_has_scaling(model)
    return model.predict(preprocess(image, (h, w), divide), verbose=0)[0]


@st.cache_data
def load_eval(stem: str):
    f = RES_DIR / f"{stem}_eval.json"
    return json.loads(f.read_text()) if f.exists() else None


@st.cache_data
def load_probs(stem: str):
    f = RES_DIR / f"{stem}_probs.npz"
    if not f.exists():
        return None
    d = np.load(f)
    return d["probs"], d["labels"]


@st.cache_data
def load_summary():
    return json.loads((RES_DIR / "summary.json").read_text())


def load_history(stem: str, uploaded=None):
    """Keras history (dict of lists) from results/<stem>_history.json|csv or an uploaded file."""
    try:
        if uploaded is not None:
            if uploaded.name.endswith(".csv"):
                return pd.read_csv(uploaded).to_dict("list")
            return json.load(uploaded)
        j, c = RES_DIR / f"{stem}_history.json", RES_DIR / f"{stem}_history.csv"
        if j.exists():
            return json.loads(j.read_text())
        if c.exists():
            return pd.read_csv(c).to_dict("list")
    except Exception as e:
        st.error(f"Could not read the history file: {e}")
    return None


def roc_curve(pos: np.ndarray, score: np.ndarray):
    order = np.argsort(-score)
    pos = pos[order]
    tpr = np.concatenate([[0], np.cumsum(pos) / pos.sum()])
    fpr = np.concatenate([[0], np.cumsum(1 - pos) / (1 - pos).sum()])
    auc = float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2))
    return fpr, tpr, auc


def topk_chart(names, probs, color, k=3):
    order = np.argsort(probs)[::-1][:k]
    fig, ax = plt.subplots(figsize=(4.6, 0.55 * k + 0.5))
    vals = [probs[i] * 100 for i in order][::-1]
    labs = [names[i] for i in order][::-1]
    bars = ax.barh(labs, vals, color=color, height=0.55)
    for b, v in zip(bars, vals):
        if v > 80:
            ax.text(v - 2, b.get_y() + b.get_height() / 2, f"{v:.1f} %", va="center", ha="right", color="white")
        else:
            ax.text(v + 2, b.get_y() + b.get_height() / 2, f"{v:.1f} %", va="center", color=INK)
    ax.set_xlim(0, 110)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("probability (%)")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Sidebar: model selection
# --------------------------------------------------------------------------- #
found = sorted(p.name for p in APP_DIR.glob("*.keras") if not p.name.startswith("_"))
labels = {n: read_meta(APP_DIR / n).get("label", Path(n).stem) for n in found}
found.sort(key=lambda n: (not n.startswith("cnn_intel_modele"), n))

st.sidebar.title("🧠 CNN classification")
st.sidebar.caption("Choose a model, then explore the tabs.")
uploaded_model = st.sidebar.file_uploader("...or upload a .keras model", type=["keras"])

if uploaded_model is not None:
    import hashlib
    upload_bytes = uploaded_model.getvalue()
    upload_hash = hashlib.sha256(upload_bytes).hexdigest()[:16]
    upload_dir = APP_DIR / ".uploaded_models"
    upload_dir.mkdir(exist_ok=True)
    model_path = upload_dir / f"model_{upload_hash}.keras"
    if not model_path.exists():
        model_path.write_bytes(upload_bytes)
    label_stem = Path(uploaded_model.name).stem
    model_label = f"Uploaded: {uploaded_model.name}"
    meta = {}
elif found:
    choice = st.sidebar.radio("Model", found, format_func=lambda n: labels[n])
    model_path = APP_DIR / choice
    label_stem = model_path.stem
    model_label = labels[choice]
    meta = read_meta(model_path)
else:
    st.error("No .keras model found next to this script. Upload one in the sidebar.")
    st.stop()

stem = model_path.stem
model = load_model(str(model_path))
n_out = int(model.output_shape[-1])
img_h, img_w = int(model.input_shape[1]), int(model.input_shape[2])

default_names, names_source = load_class_names(
    model_path if uploaded_model is None else APP_DIR / f"{label_stem}.keras", n_out
)
names_text = st.sidebar.text_input(
    "Class names (same order as training, comma-separated)",
    value=", ".join(default_names),
    help="Auto-filled from <model>_classes.json when that file exists. Keras sorts class folders alphabetically.",
)
class_names = [c.strip() for c in names_text.split(",") if c.strip()]
if len(class_names) != n_out:
    st.sidebar.error(f"The model has {n_out} outputs but you gave {len(class_names)} names.")
    st.stop()
st.sidebar.caption(
    f"Class names loaded from `{names_source}`" if names_source else "No *_classes.json found - using the names above."
)

scaling_mode = st.sidebar.radio(
    "Pixel scaling",
    ["Auto", "Model already rescales (feed 0-255)", "Divide by 255 here"],
    help="A model with a Rescaling(1/255) layer inside must be fed raw 0-255 values: "
         "dividing by 255 again in the app destroys accuracy.",
)
if scaling_mode == "Auto":
    divide = not model_has_scaling(model)
elif scaling_mode.startswith("Model"):
    divide = False
else:
    divide = True

top_k = st.sidebar.slider("Top-k to display", 1, min(5, n_out), min(3, n_out))
threshold = st.sidebar.slider("Low-confidence warning below (%)", 0, 100, 60)

color = COLORS.get(stem, "#2a78d6")
summary = load_summary()
info = summary.get(stem)
is_intel = stem.startswith("cnn_intel") and uploaded_model is None
sample_kind = "intel" if is_intel else ("fruits" if stem.startswith("fruits") else None)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Input size:** {img_h}x{img_w}  \n**Classes:** {n_out}  \n**Parameters:** {model.count_params():,}  \n"
    f"**Scaling in app:** {'/255' if divide else 'none (model handles it)'}"
)
st.sidebar.caption(meta.get("description", ""))

st.title("Image classification with CNNs")
st.markdown(
    f"<span class='pill'>Selected model</span> **{model_label}**", unsafe_allow_html=True
)

tab_over, tab_pred, tab_test, tab_curves, tab_model, tab_cmp = st.tabs(
    ["Overview", "Predict", "Test results", "Curves", "Model details", "Compare models"]
)

# --------------------------------------------------------------------------- #
# Overview
# --------------------------------------------------------------------------- #
with tab_over:
    ev = load_eval(stem)
    st.subheader(model_label)
    st.write(meta.get("description", ""))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test accuracy", f"{ev['accuracy']*100:.2f} %" if ev else (f"{info['test_acc']*100:.2f} %*" if info else "n/a"))
    c2.metric("Macro F1 (test)", f"{ev['macro_f1']*100:.2f} %" if ev else (f"{info['macro_f1']*100:.2f} %*" if info else "n/a"))
    c3.metric("Parameters", f"{model.count_params():,}")
    c4.metric("Classes", n_out)
    if not ev:
        if info:
            st.info("*Test accuracy and macro F1 are reported from the original project; test predictions and evaluation artifacts are not included for this model. They have not been re-evaluated here.")
        else:
            st.info("No stored test-set evaluation for this model (the test images are not part of this folder).")
    st.markdown("#### How to use this interface")
    st.markdown(
        "1. **Pick a model** in the sidebar (or upload your own `.keras`) - start with the *initial CNN*, then switch to the *improved CNN*. "
        "Class names, pixel scaling, top-k and the low-confidence threshold are also set there.\n"
        "2. **Predict**: classify a test image or your own upload and read the confidence.\n"
        "3. **Test results**: accuracy, F1 and confusion matrix on the 3,000 test images.\n"
        "4. **Curves**: ROC curves per class, calibration curve and training history.\n"
        "5. **Model details**: architecture, parameters, optimizer and training budget.\n"
        "6. **Compare models**: all models side by side, and both CNNs on the same image."
    )
    rows = []
    for k, v in summary.items():
        rows.append({"Model": v["name"], "Available here": "yes" if (APP_DIR / f"{k}.keras").exists() else "report only",
                     "Test accuracy": f"{v['test_acc']*100:.2f} %", "Parameters": f"{v['params']:,}"})
    st.markdown("#### Models of the project")
    st.dataframe(pd.DataFrame(rows), hide_index=True)

# --------------------------------------------------------------------------- #
# Predict
# --------------------------------------------------------------------------- #
def sample_list(kind):
    if kind == "intel":
        mf = SAMPLE_DIR / "intel" / "manifest.json"
        tags = {"ok": "easy case", "m1ok_m2err": "initial CNN right, improved wrong",
                "hard": "hard case (both wrong)", "m2ok_m1err": "improved right, initial wrong"}
        items = json.loads(mf.read_text()) if mf.exists() else []
        return [(f"{o['true']} - {tags[o['tag']]}", SAMPLE_DIR / "intel" / o["file"], o["true"]) for o in items]
    if kind == "fruits":
        return [(p.stem.replace("_", " "), p, p.stem.split("_")[0]) for p in sorted((SAMPLE_DIR / "fruits").glob("*.jpg"))]
    return []


def pick_images(key: str, kind, multiple: bool = False):
    """Return a list of (PIL image, true label or None, display name)."""
    samples = sample_list(kind)
    src = "Upload my own"
    if samples:
        src = st.radio("Image source", ["Test-set example", "Upload my own"], horizontal=True, key=f"src_{key}")
    if src == "Test-set example":
        i = st.selectbox("Example", range(len(samples)), format_func=lambda i: samples[i][0], key=f"ex_{key}")
        return [(Image.open(samples[i][1]), samples[i][2], samples[i][0])]
    ups = st.file_uploader("Upload one or more images" if multiple else "Upload an image",
                           type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=multiple, key=f"up_{key}")
    if not ups:
        return []
    ups = ups if multiple else [ups]
    return [(Image.open(u), None, u.name) for u in ups]


def pick_image(key: str, kind):
    items = pick_images(key, kind)
    return items[0] if items else (None, None, None)


with tab_pred:
    st.subheader("Predict with the selected model")
    if names_source is None and uploaded_model is None:
        st.warning(f"No `{model_path.stem}_classes.json` found next to the model, so generic names "
                   "(class_0, class_1, ...) are used. Copy that file next to this script, or type the "
                   "names in the sidebar, e.g. `apple, banana, orange`.")
    st.write("Classes known by this model: " + ", ".join(f"**{c}**" for c in class_names))
    items = pick_images("pred", sample_kind, multiple=True)
    if not items:
        st.info("Please choose an example or upload an image to classify.")
    for img, true_label, nm in items:
        st.divider()
        left, right = st.columns([1, 1.2])
        probs = predict(model, img, divide)
        best = int(np.argmax(probs))
        with left:
            show_image(to_rgb(img), caption=nm)
            with st.expander(f"What the model sees ({img_h}x{img_w})"):
                small = preprocess(img, (img_h, img_w), divide_by_255=False)[0].astype("uint8")
                st.image(small, width=160)
        with right:
            st.markdown(f"### Prediction: **{class_names[best]}**")
            m1, m2 = st.columns(2)
            m1.metric("Confidence", f"{probs[best]*100:.1f} %")
            if true_label:
                ok = class_names[best] == true_label
                m2.metric("True class", true_label, "correct" if ok else "wrong",
                          delta_color="normal" if ok else "inverse")
            if probs[best] * 100 < threshold:
                st.warning("Low confidence - the image may be unlike the training data.")
            st.pyplot(topk_chart(class_names, probs, color, k=top_k), clear_figure=True)

# --------------------------------------------------------------------------- #
# Test results
# --------------------------------------------------------------------------- #
with tab_test:
    ev = load_eval(stem)
    st.subheader(f"Test-set results - {model_label}")
    if not ev:
        st.info("No stored test-set evaluation for this model. The summary tab may show previously reported metrics, which are not a fresh evaluation.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{ev['accuracy']*100:.2f} %")
        c2.metric("Macro F1", f"{ev['macro_f1']*100:.2f} %")
        c3.metric("Test loss", f"{ev['test_loss']:.3f}")
        c4.metric("Test images", f"{ev['n_test']:,}")
        cm = np.array(ev["confusion_matrix"])
        cls = ev["classes"]
        a, b = st.columns([1.15, 1])
        with a:
            st.markdown("**Confusion matrix** (rows: true class, columns: predicted)")
            fig, ax = plt.subplots(figsize=(5.4, 4.6))
            ax.grid(False)
            norm = cm / cm.sum(1, keepdims=True)
            im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
            ax.set_xticks(range(len(cls)), cls, rotation=35, ha="right")
            ax.set_yticks(range(len(cls)), cls)
            for i in range(len(cls)):
                for j in range(len(cls)):
                    ax.text(j, i, cm[i, j], ha="center", va="center",
                            color="white" if norm[i, j] > 0.55 else INK, fontsize=9)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_xlabel("predicted")
            ax.set_ylabel("true")
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)
        with b:
            st.markdown("**F1-score per class**")
            f1 = [ev["per_class"][c]["f1"] * 100 for c in cls]
            fig, ax = plt.subplots(figsize=(4.6, 3.2))
            bars = ax.barh(cls[::-1], f1[::-1], color=color, height=0.55)
            for bar, v in zip(bars, f1[::-1]):
                ax.text(v + 1, bar.get_y() + bar.get_height() / 2, f"{v:.1f}", va="center")
            ax.set_xlim(0, 112)
            ax.grid(axis="y", visible=False)
            ax.set_xlabel("F1 (%)")
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)
            off = [(cm[i, j], cls[i], cls[j]) for i in range(len(cls)) for j in range(len(cls)) if i != j]
            off.sort(reverse=True)
            st.markdown("**Most frequent confusions**")
            st.dataframe(pd.DataFrame([{"True": t, "Predicted": p, "Images": int(n)} for n, t, p in off[:4]]),
                         hide_index=True)
        pc = pd.DataFrame(ev["per_class"]).T
        pc.columns = ["Precision", "Recall", "F1", "Support"]
        pc[["Precision", "Recall", "F1"]] = (pc[["Precision", "Recall", "F1"]] * 100).round(1)
        pc["Support"] = pc["Support"].astype(int)
        st.dataframe(pc)

# --------------------------------------------------------------------------- #
# Curves
# --------------------------------------------------------------------------- #
with tab_curves:
    st.subheader(f"Curves - {model_label}")
    pr = load_probs(stem)
    if pr is None:
        st.info("No stored test predictions for this model, so ROC and calibration curves are unavailable.")
    else:
        probs_t, y_t = pr
        cls = class_names
        st.markdown("**ROC curve per class** (one-vs-rest, test set)")
        fig, axes = plt.subplots(2, 3, figsize=(10, 5.6), sharex=True, sharey=True)
        for k, ax in enumerate(axes.ravel()):
            fpr, tpr, auc = roc_curve((y_t == k).astype(float), probs_t[:, k])
            ax.plot([0, 1], [0, 1], color=GRID, lw=1.2, ls="--")
            ax.plot(fpr, tpr, color=color, lw=2)
            ax.set_title(f"{cls[k]}  (AUC {auc:.3f})", fontsize=10, color=INK)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.02)
            if k >= 3:
                ax.set_xlabel("false positive rate")
            if k % 3 == 0:
                ax.set_ylabel("true positive rate")
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)

        a, b = st.columns(2)
        with a:
            st.markdown("**Calibration curve** - is the confidence trustworthy?")
            conf = probs_t.max(1)
            correct = (probs_t.argmax(1) == y_t).astype(float)
            edges = np.linspace(0.3, 1.0, 8)
            mid, acc, cnt = [], [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = (conf >= lo) & (conf < hi + (1e-9 if hi == 1.0 else 0))
                if m.sum() >= 5:
                    mid.append((lo + hi) / 2 * 100)
                    acc.append(correct[m].mean() * 100)
                    cnt.append(int(m.sum()))
            fig, ax = plt.subplots(figsize=(4.8, 3.6))
            ax.plot([30, 100], [30, 100], color=GRID, lw=1.2, ls="--", label="perfect calibration")
            ax.plot(mid, acc, color=color, lw=2, marker="o", ms=6, mec="white", mew=1.5, label="model")
            ax.set_xlabel("confidence (%)")
            ax.set_ylabel("actual accuracy (%)")
            ax.legend(frameon=False, loc="upper left")
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)
        with b:
            st.markdown("**Confidence distribution**")
            fig, ax = plt.subplots(figsize=(4.8, 3.6))
            ax.hist(conf * 100, bins=np.linspace(15, 100, 18), color=color, rwidth=0.9)
            ax.set_xlabel("max predicted probability (%)")
            ax.set_ylabel("test images")
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)

    st.markdown("---")
    st.markdown("**Training history** (loss / accuracy per epoch)")
    up_hist = st.file_uploader("Load a Keras history (.json or .csv)", type=["json", "csv"], key="hist")
    hist = load_history(stem, up_hist)
    if hist:
        loss_keys = [k for k in ("loss", "val_loss") if k in hist]
        acc_keys = [k for k in hist if "acc" in k]
        a, b = st.columns(2)
        for col, keys, ttl in ((a, loss_keys, "Loss"), (b, sorted(acc_keys, key=lambda k: k.startswith("val")), "Accuracy")):
            with col:
                fig, ax = plt.subplots(figsize=(4.8, 3.4))
                for k in keys:
                    ep = np.arange(1, len(hist[k]) + 1)
                    val = k.startswith("val")
                    ax.plot(ep, hist[k], lw=2, ls="--" if val else "-", color=color if not val else INK2,
                            label="validation" if val else "train")
                ax.set_xlabel("epoch")
                ax.set_title(ttl, loc="left", fontsize=11)
                ax.legend(frameon=False)
                fig.tight_layout()
                st.pyplot(fig, clear_figure=True)
    else:
        st.info(
            "No training history stored for this model. Export it from the notebook "
            f"(`json.dump(history.history, open('results/{stem}_history.json', 'w'))`) "
            "or load the file above to plot the learning curves."
        )

# --------------------------------------------------------------------------- #
# Model details
# --------------------------------------------------------------------------- #
with tab_model:
    st.subheader(f"Model details - {model_label}")
    if info:
        st.write(info["description"])
    trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    try:
        lr = float(model.optimizer.learning_rate)
        opt = f"{model.optimizer.__class__.__name__} (lr = {lr:.2e})"
    except Exception:
        opt = "n/a"
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total parameters", f"{model.count_params():,}")
    c2.metric("Trainable", f"{trainable:,}")
    c3.metric("Input", f"{model.input_shape[1]}x{model.input_shape[2]}x3")
    c4.metric("Outputs", n_out)
    if info and opt == "n/a":
        opt = "not loaded (inference mode)"
    extra = [f"**Optimizer:** {opt}", "**Loss:** sparse categorical cross-entropy (from original training)",
             f"**Pixel scaling:** {'inside the model (Rescaling layer)' if model_has_scaling(model) else 'done in the app (/255)'}"]
    if info:
        extra += [f"**Epochs run:** {info['epochs']}", f"**Training time:** {info['time_min']} min",
                  f"**Validation accuracy:** {info['val_acc']*100:.2f} %"]
    st.markdown("  \n".join(extra))
    rows = []
    for layer in model.layers:
        try:
            shp = str(tuple(layer.output.shape))
        except Exception:
            shp = "-"
        rows.append({"Layer": layer.name, "Type": layer.__class__.__name__, "Output shape": shp,
                     "Parameters": f"{layer.count_params():,}"})
    st.markdown("**Layers**")
    st.dataframe(pd.DataFrame(rows), hide_index=True, height=min(35 * (len(rows) + 1) + 3, 1100))
    pl = [(r["Layer"], int(r["Parameters"].replace(",", ""))) for r in rows if r["Parameters"] != "0"]
    if len(pl) > 1:
        fig, ax = plt.subplots(figsize=(8, 0.38 * len(pl) + 0.8))
        names_, vals_ = zip(*pl[::-1])
        ax.barh(names_, vals_, color=color, height=0.55)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("parameters")
        fig.tight_layout()
        st.markdown("**Parameters per layer**")
        st.pyplot(fig, clear_figure=True)

# --------------------------------------------------------------------------- #
# Compare
# --------------------------------------------------------------------------- #
with tab_cmp:
    st.subheader("Compare models (Intel scene classification)")
    keys = list(summary)
    df = pd.DataFrame([{
        "Model": summary[k]["name"], "Parameters": f"{summary[k]['params']:,}", "Epochs": summary[k]["epochs"],
        "Time (min)": summary[k]["time_min"], "Val. accuracy": f"{summary[k]['val_acc']*100:.2f} %",
        "Test accuracy": f"{summary[k]['test_acc']*100:.2f} %", "Macro F1": f"{summary[k]['macro_f1']*100:.2f} %",
    } for k in keys])
    st.dataframe(df, hide_index=True)
    st.caption("All three Intel .keras models are included. Metrics for Model 3 come from the original report; "
               "Models 1 and 2 have stored test evaluation artifacts. Model 3 lacks the saved test predictions, "
               "so ROC/confusion matrix results for Model 3 are not shown.")

    a, b = st.columns(2)
    with a:
        st.markdown("**Test accuracy and macro F1**")
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        x = np.arange(len(keys))
        w = 0.36
        for off_, metric, lab in ((-w / 2, "test_acc", "accuracy"), (w / 2, "macro_f1", "macro F1")):
            vals = [summary[k][metric] * 100 for k in keys]
            bars = ax.bar(x + off_, vals, w * 0.92, color=[COLORS[k] for k in keys],
                          alpha=1.0 if metric == "test_acc" else 0.55)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.6, f"{v:.1f}", ha="center", fontsize=8.5)
        ax.set_xticks(x, ["Model 1\ninitial", "Model 2\nimproved", "Model 3\nMobileNetV2"])
        ax.set_ylim(70, 98)
        ax.set_ylabel("%  (dark = accuracy, light = macro F1)")
        ax.grid(axis="x", visible=False)
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
    with b:
        st.markdown("**F1 per class: initial vs improved**")
        e1, e2 = load_eval("cnn_intel_modele1"), load_eval("cnn_intel_modele2")
        if e1 and e2:
            cls = e1["classes"]
            fig, ax = plt.subplots(figsize=(5.2, 3.6))
            x = np.arange(len(cls))
            for off_, e, k, lab in ((-0.19, e1, "cnn_intel_modele1", "initial"), (0.19, e2, "cnn_intel_modele2", "improved")):
                ax.bar(x + off_, [e["per_class"][c]["f1"] * 100 for c in cls], 0.34, color=COLORS[k], label=lab)
            ax.set_xticks(x, cls, rotation=30, ha="right")
            ax.set_ylim(60, 100)
            ax.set_ylabel("F1 (%)")
            ax.grid(axis="x", visible=False)
            ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)

    st.markdown("---")
    st.markdown("**Same image, all three Intel models**")
    available = [n for n in ("cnn_intel_modele1.keras", "cnn_intel_modele2.keras",
                             "cnn_intel_modele3_mobilenetv2.keras") if n in found]
    if len(available) < 2:
        st.info("At least two Intel CNN models are needed for a prediction comparison.")
    else:
        img, true_label, nm = pick_image("cmp", "intel")
        if img is not None:
            show_image(to_rgb(img), caption=nm)
            columns = st.columns(len(available))
            for col, fn in zip(columns, available):
                m = load_model(str(APP_DIR / fn))
                names = read_meta(APP_DIR / fn)["class_names"]
                p = predict(m, img)
                bi = int(np.argmax(p))
                with col:
                    ttl = summary[Path(fn).stem]["name"]
                    mark = ""
                    if true_label:
                        mark = " - correct" if names[bi] == true_label else " - wrong"
                    st.markdown(f"**{ttl}**: {names[bi]} ({p[bi]*100:.1f} %){mark}")
                    st.pyplot(topk_chart(names, p, COLORS[Path(fn).stem], k=top_k), clear_figure=True)
