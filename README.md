# Cardio AI: Explainable Heart Disease Prediction 🫀🤖✨

> **Using CNN-LSTM Attention Network with Grad-CAM Interpretability 🧠🔬**
>
> *Software Engineering Project (UCS503P) - Thapar Institute of Engineering and Technology 🎓🏛️*

## 📖 Overview 🌍

Cardiovascular disease remains the leading cause of mortality globally 🩺📉. While deep learning models achieve high accuracy in predicting heart disease from clinical measurements, their "black-box" nature ⬛📦 makes physicians reluctant to trust automated diagnoses. 

**Cardio AI** 🚀 aims to bridge this gap by prioritizing clinical transparency 🔍. This project translates complex deep neural network confidence scores into actionable, human-readable clinical narratives 📄💡, empowering physicians to trust AI-assisted diagnoses 👨‍⚕️🤝.

## 🎯 Problem Statement ⚠️

| Challenge 🛑 | Description 📝 |
| :--- | :--- |
| **Black-Box Predictions 🧩** | Deep learning models offer no clinical reasoning or transparent decision pathways. |
| **Feature Opacity 🌫️** | It is difficult to understand which specific physiological parameters (out of 13 heterogeneous clinical features) drive a patient's risk score 📊. |
| **Lack of Visual Evidence 👁️‍🗨️** | Clinicians lack interactive visual artifacts (like heatmaps 🌡️) that map model confidence directly back to patient physiology. |
| **Absence of Longitudinal Tracking ⏳** | Existing tools rarely support patient-history-aware predictions across repeated assessments 📈. |

## 💡 Proposed Solution ✅

| Component 🧱 | Description 📜 |
| :--- | :--- |
| **Deep Learning Model 🧠** | CNN-LSTM-Attention network trained on the UCI Cleveland dataset using Gramian Angular Summation Field (GASF) image encoding 🖼️. |
| **Explainability Engine 🔍** | Combines **Grad-CAM heatmaps 🔥**, **Attention weights 🎯**, and **SHAP values ⚖️** to provide complementary lenses on model reasoning. |
| **Interactive Dashboard 💻** | Premium Single Page Application (SPA) with dynamic range validations, interactive Plotly charts 📊, and 3D hardware-accelerated backgrounds 🌌. |
| **Automated Reporting 📄** | Generates PDF reports featuring AI-driven narrative summaries and guideline-referenced clinical interpretations 📋. |
| **Patient Management 🗄️** | Flask-based backend with a MySQL database to securely track longitudinal patient records 🔐. |

## ⚙️ Core Workflow 🔄

| Step 🔢 | Action 🎬 | Description 📝 |
| :---: | :--- | :--- |
| **1** | **Data Entry ⌨️** | Clinician enters 13 patient clinical parameters via the web form 📋. |
| **2** | **Preprocessing 🧮** | System normalizes inputs using a custom `ClinicalRangeScaler` and converts them to a GAF image representation 🖼️. |
| **3** | **Prediction 🔮** | Runs the prediction through the CNN-LSTM-Attention model 🧠. |
| **4** | **XAI Analysis 🔬** | Explainability engine generates a Grad-CAM heatmap 🔥, extracts attention weights 🎯, and computes SHAP attributions ⚖️. |
| **5** | **Reporting 📑** | Interactive dashboard displays XAI visualizations side-by-side; a downloadable PDF clinical report is generated 📥. |
| **6** | **Persistence 💾** | Patient records and AI diagnoses are saved to a MySQL database for longitudinal tracking 📈. |

## 🛠️ Technology Stack 🧰

| Category 🗂️ | Technologies Used 💻 |
| :--- | :--- |
| **Frontend Engineering 🎨** | HTML5 🌐, CSS3 💅, Bootstrap 5 👢, Vanilla JavaScript 💛 |
| **Data Visualization & Aesthetics 📈** | Plotly.js 📊, Three.js 🧊, Vanta.js 🌊, Web Audio API 🔊 |
| **Machine Learning 🤖** | TensorFlow/Keras 🧠, scikit-learn 📉, pyts (GASF) 🖼️, SHAP ⚖️ |
| **Backend & Integration 🔌** | Flask (Python REST API) 🐍, PyMySQL (Database connectivity) 🗄️ |

## 👥 Authors 👨‍💻👩‍💻

| Name 👤 | Roll Number 🆔 |
| :--- | :--- |
| **Dhruv Singla** 👨‍💻 | 1024170434 |
| **Srishti** 👩‍💻 | 1024170451 |

## 📄 License ⚖️

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
