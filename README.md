# Cardio AI: Explainable Heart Disease Prediction 🫀🤖

> **Using a CNN-LSTM Attention Network with Grad-CAM Interpretability**
>
> *Software Engineering Project (UCS503P) — Thapar Institute of Engineering and Technology*

Cardio AI is a complete, end-to-end clinical decision-support system that predicts
heart-disease risk from 13 routine clinical parameters and — crucially — **explains
every prediction** using three complementary Explainable-AI techniques (Grad-CAM,
Attention weights and SHAP). It ships with a trained model, an interactive web
dashboard, an automated PDF clinical report generator, and an optional MySQL
patient-history backend.

---

## 📖 Overview

Cardiovascular disease remains the leading cause of mortality globally. While deep
learning models can predict heart disease from routine clinical measurements, their
"black-box" nature makes physicians reluctant to trust automated diagnoses.

**Cardio AI** bridges this gap by prioritising clinical transparency: it translates
complex network confidence scores into actionable, human-readable clinical narratives
that map model reasoning directly back to a patient's physiology.

---

## 🎯 Problem Statement

| Challenge | Description |
| :--- | :--- |
| **Black-Box Predictions** | Deep learning models offer no clinical reasoning or transparent decision pathways. |
| **Feature Opacity** | It is difficult to understand which of the 13 heterogeneous clinical features drives a patient's risk score. |
| **Lack of Visual Evidence** | Clinicians lack interactive visual artifacts (heatmaps, attributions) mapping confidence back to physiology. |
| **Absence of Longitudinal Tracking** | Existing tools rarely support patient-history-aware predictions across repeated assessments. |

---

## ✨ Key Features

| Feature | Description |
| :--- | :--- |
| **Deep Learning Model** | CNN-LSTM-Attention network over **Gramian Angular Summation Field (GASF)** image encodings of the 13 clinical features. |
| **Multi-Method Explainability** | Grad-CAM 2D heatmaps + Attention weights + SHAP values, aggregated back to the original clinical features. |
| **Interactive Dashboard** | Premium SPA with live range validation, interactive Plotly charts (attribution bars, GAF & Grad-CAM heatmaps) and animated backgrounds. |
| **Automated PDF Reporting** | Publication-quality clinical reports with AI narratives and ACC/AHA-style guideline-referenced interpretation. |
| **Clinical Guidelines Engine** | Rule-based `clinical_guidelines.json` maps values to interpretable, evidence-aligned recommendations. |
| **Patient Management** | Optional Flask + MySQL backend for longitudinal patient records and diagnosis history search. |
| **Reproducible Pipeline** | A single CLI (`main.py`) runs EDA, training, ablation study, cross-validation and XAI report generation. |

---

## ⚙️ Core Workflow

| Step | Action | Description |
| :---: | :--- | :--- |
| **1** | **Data Entry** | Clinician enters 13 patient parameters via the web form. |
| **2** | **Preprocessing** | Inputs are validated, scaled with a custom `ClinicalRangeScaler`, one-hot encoded, and converted to a **GASF** image. |
| **3** | **Prediction** | The CNN-LSTM-Attention model outputs a calibrated disease probability. |
| **4** | **XAI Analysis** | The engine computes Grad-CAM, Attention and SHAP attributions and aggregates them to the 13 original features. |
| **5** | **Reporting** | The dashboard displays the XAI visualisations side-by-side and generates a downloadable PDF dossier. |
| **6** | **Persistence** | Patient records and AI diagnoses are stored in MySQL for longitudinal tracking. |

---

## 🏗️ Architecture

```
                    ┌──────────────────────────────────────────────┐
   Clinician ─────► │  Web Dashboard (templates/ + static/)         │
   (browser)        │  Plotly · Three.js · Vanta.js · Bootstrap 5   │
                    └───────────────┬──────────────────────────────┘
                                    │  JSON (REST)
                    ┌───────────────▼──────────────────────────────┐
                    │  Flask API  (code/app.py)                     │
                    │   POST /api/predict   GET /api/search         │
                    └───────┬───────────────────────┬──────────────┘
                            │                       │
        ┌───────────────────▼──────────┐   ┌────────▼───────────────┐
        │  ML / XAI Pipeline (src/)     │   │  MySQL  (optional)      │
        │  preprocessing → GASF         │   │  patients               │
        │  CNN-LSTM-Attention           │   │  clinical_records       │
        │  Grad-CAM · Attention · SHAP  │   │  ai_diagnoses           │
        │  PDF report generator         │   └────────────────────────┘
        └───────────────────────────────┘
```

---

## 🛠️ Technology Stack

| Category | Technologies Used |
| :--- | :--- |
| **Frontend** | HTML5, CSS3, Bootstrap 5, Vanilla JavaScript |
| **Visualisation & Aesthetics** | Plotly.js, Three.js, Vanta.js, Web Audio API |
| **Machine Learning** | TensorFlow/Keras (Keras API), scikit-learn, NumPy GASF encoding, SHAP |
| **Explainability** | Grad-CAM, Attention weights, SHAP |
| **Backend & Integration** | Flask (Python REST API), PyMySQL |
| **Database** | MySQL 8.0 (Docker), phpMyAdmin |
| **Reporting** | ReportLab / headless Chrome (HTML → PDF) |

> **No GPU or native TensorFlow required.** The project ships a lightweight
> NumPy / scikit-learn Keras emulator at `code/lib/tensorflow.py`, placed on
> `sys.path` ahead of site-packages, so the full pipeline runs on any CPU. Set
> `tensorflow==2.13.0` in `requirements.txt` to swap in real Keras transparently.

---

## 🚀 Quick Start

**Prerequisites:** Python 3.10+ (developed on 3.11), and optionally Docker Desktop.

```bash
git clone https://github.com/dsingla3be24/UCS503P-202627-Explainable-Heart-Disease-Prediction-.git
cd UCS503P-202627-Explainable-Heart-Disease-Prediction-/code

python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python app.py                        # dashboard at http://localhost:5001
```

On first run the model trains on the UCI Cleveland dataset (CPU, fast) and saves
its weights to `results/`. Subsequent runs load the saved weights instantly.

### Run the offline pipeline (EDA · training · ablation · XAI)

```bash
cd code
python main.py                 # full pipeline → results/
python main.py --cv            # add 5-fold cross-validation
python main.py --patient_idx 3 # explain a single test patient + PDF
python main.py --tune          # Keras-Tuner hyper-parameter search
```

### Optional: MySQL persistence

```bash
cd code
docker compose up -d           # MySQL on :3307, phpMyAdmin on :8080
```

If the database is unreachable the API keeps working — persistence is skipped and
history search returns an empty list.

---

## 🔌 API Reference

| Method | Endpoint | Description |
| :---: | :--- | :--- |
| `GET` | `/` | Serve the dashboard SPA. |
| `POST` | `/api/predict` | Predict risk; returns probability, XAI attributions, GAF/Grad-CAM matrices, guideline interpretation and a PDF report URL. |
| `GET` | `/api/search?q=<query>` | Search longitudinal patient history by name or hospital ID. |
| `GET` | `/static/temp/<file>` | Serve generated HTML/PDF reports. |

**Example**

```bash
curl -X POST http://localhost:5001/api/predict \
  -H "Content-Type: application/json" \
  -d '{"patient_name":"Test","age":63,"sex":1,"cp":4,"trestbps":145,"chol":233,
       "fbs":1,"restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":3,
       "ca":0,"thal":7}'
```

---

## 📊 Dataset

**UCI Cleveland Heart Disease** — 303 records × 14 attributes (13 clinical features + target).

| Feature | Clinical meaning |
| :--- | :--- |
| `age`, `sex` | Demographics |
| `cp` | Chest-pain type |
| `trestbps`, `chol` | Resting blood pressure, serum cholesterol |
| `fbs`, `restecg` | Fasting blood sugar, resting ECG |
| `thalach`, `exang`, `oldpeak`, `slope` | Max heart rate, exercise angina, ST depression, ST slope |
| `ca`, `thal` | Major vessels, thalassemia |

Missing values are median-imputed; numericals are scaled with a clinically bounded
`ClinicalRangeScaler`; categoricals are one-hot encoded; the training split is
SMOTE-balanced when `imbalanced-learn` is available.

---

## 📈 Results

Hold-out test set (61 patients: 33 normal, 28 disease):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| CNN | 0.902 | 0.844 | 0.964 | 0.900 | 0.918 |
| CNN-LSTM | 0.902 | 0.844 | 0.964 | 0.900 | 0.918 |
| **CNN-LSTM-Attention** | **0.902** | **0.844** | **0.964** | **0.900** | **0.918** |

All figures are regenerated by `python main.py` and committed under
[`code/results/`](code/results/):

- `eda_*.png` — class distribution, correlation matrix, feature distributions
- `*_confusion_matrix.png`, `*_roc_curve.png` — per-model evaluation
- `ablation_study_holdout.csv`, `ablation_study_comparison.png` — model comparison
- `global_attention_heatmap.png`, `global_grad_cam_heatmap.png` — global XAI
- `patient_*/` — per-patient explainability plots, GAF images, clinical text and PDF reports

> The bundled model is a CPU-friendly emulator; the ablation table reflects the
> current reproducible emulator run. Real Keras/TensorFlow weights can be produced
> by enabling `tensorflow==2.13.0`.

---

## 🔬 Explainability (XAI)

| Method | What it provides |
| :--- | :--- |
| **Grad-CAM** | 2D class-discriminative saliency map over the GASF image, pooled to per-feature importance. |
| **Attention** | Learned attention weights over the encoded clinical features. |
| **SHAP** | Shapley-value attributions (KernelExplainer) for per-feature contributions. |
| **Agreement metrics** | Spearman rank correlation and top-k Jaccard similarity between Attention and Grad-CAM. |

---

## 📁 Repository Layout

| Path | Contents |
| :--- | :--- |
| `code/` | **Complete application** — Flask API, ML/XAI pipeline, dashboard, results |
| `code/app.py` | Flask REST API + dashboard server |
| `code/main.py` | Offline CLI pipeline (EDA, training, ablation, CV, XAI) |
| `code/src/` | `preprocessing`, `models`, `train`, `explainability`, `pdf_generator`, `fallback_model`, `numpy_model` |
| `code/templates/`, `code/static/` | Integrated dashboard (Jinja + Bootstrap/Plotly/Three/Vanta) |
| `code/lib/` | Lightweight Keras/TF emulator + vendored JS |
| `code/results/` | Generated plots, metrics, weights and patient reports |
| `code/backend/` | Prototype-stage CNN baseline notebook + dataset |
| `code/frontend/` | Prototype-stage static UI (mock, no backend) |
| `diagrams/` | UML & design diagrams (use case, activity, sequence, collaboration, class, component, state, deployment, DFD) + use-case templates |
| `journals/` | Weekly individual work journals |
| `project-proposal/` | Proposal (LaTeX + PDF) |
| `project-report-prototype-stage/` | Mid-semester report (LaTeX + PDF) |
| `project-report-final/` | Final report |
| `docs/`, `mkdocs.yml` | Documentation site (MkDocs Material) |

Detailed application instructions live in [`code/README.md`](code/README.md).

---

## 👥 Authors

| Name | Roll Number |
| :--- | :--- |
| **Dhruv Singla** | 1024170434 |
| **Srishti** | 1024170451 |

**Guide:** Nisha Thakur — Computer Science and Engineering, TIET Patiala

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
