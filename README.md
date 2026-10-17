# Cardio AI: Explainable Heart Disease Prediction 🫀🤖

> **Using CNN-LSTM Attention Network with Grad-CAM Interpretability**
>
> *Software Engineering Project (UCS503P) - Thapar Institute of Engineering and Technology*

## 📖 Overview

Cardiovascular disease remains the leading cause of mortality globally. While deep learning models achieve high accuracy in predicting heart disease from clinical measurements, their "black-box" nature makes physicians reluctant to trust automated diagnoses. 

**Cardio AI** aims to bridge this gap by prioritizing clinical transparency. This project translates complex deep neural network confidence scores into actionable, human-readable clinical narratives, empowering physicians to trust AI-assisted diagnoses.

## 🎯 Problem Statement

| Challenge | Description |
| :--- | :--- |
| **Black-Box Predictions** | Deep learning models offer no clinical reasoning or transparent decision pathways. |
| **Feature Opacity** | It is difficult to understand which specific physiological parameters (out of 13 heterogeneous clinical features) drive a patient's risk score. |
| **Lack of Visual Evidence** | Clinicians lack interactive visual artifacts (like heatmaps) that map model confidence directly back to patient physiology. |
| **Absence of Longitudinal Tracking** | Existing tools rarely support patient-history-aware predictions across repeated assessments. |

## 💡 Proposed Solution

| Component | Description |
| :--- | :--- |
| **Deep Learning Model** | CNN-LSTM-Attention network trained on the UCI Cleveland dataset using Gramian Angular Summation Field (GASF) image encoding. |
| **Explainability Engine** | Combines **Grad-CAM heatmaps**, **Attention weights**, and **SHAP values** to provide complementary lenses on model reasoning. |
| **Interactive Dashboard** | Premium Single Page Application (SPA) with dynamic range validations, interactive Plotly charts, and 3D hardware-accelerated backgrounds. |
| **Automated Reporting** | Generates PDF reports featuring AI-driven narrative summaries and guideline-referenced clinical interpretations. |
| **Patient Management** | Flask-based backend with a MySQL database to securely track longitudinal patient records. |

## ⚙️ Core Workflow

| Step | Action | Description |
| :---: | :--- | :--- |
| **1** | **Data Entry** | Clinician enters 13 patient clinical parameters via the web form. |
| **2** | **Preprocessing** | System normalizes inputs using a custom `ClinicalRangeScaler` and converts them to a GAF image representation. |
| **3** | **Prediction** | Runs the prediction through the CNN-LSTM-Attention model. |
| **4** | **XAI Analysis** | Explainability engine generates a Grad-CAM heatmap, extracts attention weights, and computes SHAP attributions. |
| **5** | **Reporting** | Interactive dashboard displays XAI visualizations side-by-side; a downloadable PDF clinical report is generated. |
| **6** | **Persistence** | Patient records and AI diagnoses are saved to a MySQL database for longitudinal tracking. |

## 🛠️ Technology Stack

| Category | Technologies Used |
| :--- | :--- |
| **Frontend Engineering** | HTML5, CSS3, Bootstrap 5, Vanilla JavaScript |
| **Data Visualization & Aesthetics** | Plotly.js, Three.js, Vanta.js, Web Audio API |
| **Machine Learning** | TensorFlow/Keras (Keras API), scikit-learn, NumPy GASF encoding, SHAP |
| **Backend & Integration** | Flask (Python REST API), PyMySQL (Database connectivity) |

## 🚀 Quick Start

The complete, runnable application lives in [`code/`](code/) (see
[`code/README.md`](code/README.md) for full details).

```bash
cd code
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py            # dashboard at http://localhost:5001
```

Run the offline ML pipeline (EDA, training, cross-validation, XAI, reports):

```bash
cd code
python main.py           # writes plots + metrics to code/results/
```

## 📁 Repository Layout

| Path | Contents |
| :--- | :--- |
| `code/` | **Full application** — Flask API (`app.py`), ML/XAI pipeline (`src/`), dashboard (`templates/`, `static/`), `results/` |
| `code/backend/` | Prototype-stage CNN baseline notebook + dataset |
| `code/frontend/` | Prototype-stage static UI (mock, no backend) |
| `diagrams/` | UML diagrams (use case, activity, sequence, collaboration, class, component, deployment, DFD) |
| `journals/` | Individual weekly work journals |
| `project-proposal/` | Proposal (LaTeX + PDF) |
| `project-report-prototype-stage/` | Mid-semester report |
| `project-report-final/` | Final report |
| `docs/`, `mkdocs.yml` | Project documentation site |

## 👥 Authors

| Name | Roll Number |
| :--- | :--- |
| **Dhruv Singla** | 1024170434 |
| **Srishti** | 1024170451 |

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
