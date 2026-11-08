# CNN Image Classification — Streamlit Dashboard

Academic CNN classification project with **four saved Keras classifiers** and an interactive **six-tab dashboard**.

## Included models

| Model | Dataset | Classes | Input | Reported test accuracy |
|---|---|---|---|---|
| Initial CNN | Intel scenes | buildings, forest, glacier, mountain, sea, street | 150 × 150 | 88.10% |
| Improved CNN | Intel scenes | same six classes | 150 × 150 | 80.53% |
| MobileNetV2 | Intel scenes | same six classes | 150 × 150 | **91.90%** |
| MobileNetV2 | Fruits | apple, banana, orange | 128 × 128 | Not provided |

**Important:** Intel Model 3's 91.90% figure comes from the original training report. Its `.keras` model is included, but saved test-set probability arrays and per-image labels for that model are **not** included. Thus the app can run predictions with Model 3, but cannot show its test ROC/confusion matrix until those evaluation artifacts are generated. Fruit predictions are examples, not a reported test-set accuracy.

## Run locally

Install **Python 3.11**, then in this project folder:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run streamlit_cnn.py
```

On macOS/Linux, activate with `source .venv/bin/activate` instead.

Open `http://localhost:8501`.

## Streamlit Community Cloud

**Live app:** [cnnclassification.streamlit.app](https://cnnclassification.streamlit.app/)

1. Create a **new GitHub repository** and upload the **contents** of this ZIP, not the ZIP itself. Keep `streamlit_cnn.py`, `requirements.txt`, `runtime.txt`, `.streamlit/`, and all `.keras` files together at the repository root.
2. If GitHub blocks your model sizes or your repository uses Git LFS, review Streamlit Cloud's support before deploying. The complete ZIP includes model files; no external download step is required locally.
3. In Streamlit Community Cloud, choose **Create app** and select the repo, branch and entrypoint `streamlit_cnn.py`.
4. Deploy. TensorFlow can exceed free hosting memory/installation limits; if so, use a paid deployment platform or run locally. This bundle has not been runtime-tested on Streamlit Cloud.

## Dashboard tabs

- **Overview:** selected model, headline test metrics (reported-only model metrics are marked with `*`), configuration, project summary.
- **Predict:** sample photographs or uploaded images, predicted class, top-k confidence, source class when known, resized preview.
- **Test results:** confusion matrix and per-class F1 for Intel Models 1 and 2 using stored evaluation results.
- **Curves:** ROC, calibration and confidence histograms for Models 1 and 2; optional training-history import.
- **Model details:** model layers, parameter counts, and original project training metadata.
- **Compare models:** comparison table and charts, and **the same sample image classified by all three Intel models**.

## Directory structure

```text
streamlit_cnn.py
requirements.txt
runtime.txt
.streamlit/config.toml
cnn_intel_modele1.keras
cnn_intel_modele1_classes.json
cnn_intel_modele2.keras
cnn_intel_modele2_classes.json
cnn_intel_modele3_mobilenetv2.keras
cnn_intel_modele3_mobilenetv2_classes.json
fruits_mobilenetv2.keras
fruits_mobilenetv2_classes.json
results/                 # original Model 1 & 2 test evaluation, probability arrays, report metadata
samples/intel/           # labelled Intel image examples + manifest
samples/fruits/          # fruit prediction examples
Rapport_Interface_CNN_original.pdf
```

## Reproducibility / report notes

The PDF is the **original report** and predates inclusion of the third `.keras` model in the app. Its statements that Intel Model 3 is absent are **outdated for this bundle**. Original data, evaluation results, and reported training metrics were not fabricated or retrained. Loss/accuracy training-history curves still require an exported `results/<model_stem>_history.json` or CSV; Keras `.keras` archives do not include per-epoch history by default.

The image preprocessing preserves EXIF orientation and handles transparent images; each supplied model already contains a Keras Rescaling layer, so keep sidebar scaling on **Auto** to avoid dividing pixel values twice.

### Original attribution

Prepared by Wijdane AARROUB. Supervised by Pr. Asmae OUHMIDA. Master SDIA, ENSET Mohammedia (2026/2027), as described in the included original project report.
