# Cardio AI: Explainable Heart Disease Prediction 🫀🤖

> **Using CNN-LSTM Attention Network with Grad-CAM Interpretability**
>
> *Software Engineering Project (UCS503P) - Thapar Institute of Engineering and Technology*

## 📖 Overview

Cardiovascular disease remains the leading cause of mortality globally. While deep learning models achieve high accuracy in predicting heart disease from clinical measurements, their "black-box" nature makes physicians reluctant to trust automated diagnoses. 

**Cardio AI** aims to bridge this gap by prioritizing clinical transparency. This project translates complex deep neural network confidence scores into actionable, human-readable clinical narratives, empowering physicians to trust AI-assisted diagnoses.

## 🎯 Problem Statement

*   **Black-Box Predictions:** Deep learning models offer no clinical reasoning or transparent decision pathways.
*   **Feature Opacity:** It is difficult to understand which specific physiological parameters (out of 13 heterogeneous clinical features) drive a patient's risk score.
*   **Lack of Visual Evidence:** Clinicians lack interactive visual artifacts (like heatmaps) that map model confidence directly back to patient physiology.
*   **Absence of Longitudinal Tracking:** Existing tools rarely support patient-history-aware predictions across repeated assessments.

## 💡 Proposed Solution

A web-based explainable cardiac risk prediction system featuring:

1.  **CNN-LSTM-Attention Deep Learning Model:** Trained on the UCI Cleveland Heart Disease dataset. It uses Gramian Angular Summation Field (GASF) image encoding to convert 1D feature vectors to 2D images for spatial feature extraction.
2.  **Multi-Method Explainability Engine:** Combines **Grad-CAM heatmaps**, **Attention weights**, and **SHAP values** to provide three complementary lenses on the model's reasoning.
3.  **Interactive Clinical Dashboard:** A premium Single Page Application (SPA) with dynamic clinical range validations, interactive Plotly charts, and 3D hardware-accelerated backgrounds.
4.  **Automated Clinical Reporting:** Generates automated PDF reports featuring AI-driven narrative summaries, feature-by-feature clinical interpretations, and guideline-referenced evaluations.
5.  **Patient Management Portal:** Flask-based backend with a MySQL database to securely track longitudinal patient records.

## ⚙️ Core Workflow

1.  **Data Entry:** Clinician enters 13 patient clinical parameters (age, blood pressure, cholesterol, ECG markers, etc.) via the web form.
2.  **Preprocessing:** System normalizes inputs using a custom `ClinicalRangeScaler` (physiological-bound normalization) and converts them to a GAF image representation.
3.  **Prediction:** Runs the prediction through the CNN-LSTM-Attention model.
4.  **XAI Analysis:** The explainability engine generates a Grad-CAM heatmap, extracts attention weights, and computes SHAP attributions.
5.  **Dashboard & Reporting:** An interactive dashboard displays the XAI visualizations side-by-side, and a downloadable PDF clinical report is generated.
6.  **Persistence:** Patient records and AI diagnoses are saved to a MySQL database for longitudinal tracking.

## 🛠️ Technology Stack

*   **Frontend Engineering:** HTML5, CSS3, Bootstrap 5, Vanilla JavaScript.
*   **Data Visualization & Aesthetics:** Plotly.js (interactive charts), Three.js & Vanta.js (dynamic backgrounds), Web Audio API (feedback cues).
*   **Machine Learning:** TensorFlow/Keras (CNN-LSTM-Attention), scikit-learn (preprocessing), pyts (GASF transformations), SHAP.
*   **Backend & Integration:** Flask (Python REST API), PyMySQL (Database connectivity).

## 👥 Authors

*   **Dhruv Singla** (1024170434)
*   **Srishti** (1024170451)
