# 🚀 Week 1: Project Proposal & Architecture Definition

**📅 Date:** August 8 – August 14, 2026  
**🎯 Core Task:** Architected the technical project proposal for the Cardio AI prediction engine, outlining the deep learning pipeline, tech stack, and evaluation metrics.

---

### 🛠️ Key Contributions & Decisions

* **Defined the Problem Space:** Drafted the core problem statement, shifting the focus from standard heart disease classification to **clinical transparency**—making deep learning explainable rather than a black box.
* **Architectural Blueprint:** Sketched out the initial model architecture utilizing a hybrid **CNN-LSTM-Attention network**. Decided to implement a multi-method XAI approach leveraging **Grad-CAM**, **Attention Weights**, and **SHAP**.
* **Tech Stack Finalization:** Mapped out the end-to-end system flow:
  * **Backend:** Flask for the REST API.
  * **Database:** MySQL for patient records and history.
  * **Frontend Visualization:** Plotly.js for rendering interactive, real-time XAI dashboards.
* **Evaluation Benchmarks:** Established our baseline metrics, aiming for a minimum **AUC-ROC of 0.85**, supported by robust 5-fold cross-validation to ensure generalization.
* **Risk Mitigation:** Documented operational constraints and identified potential risks (e.g., overfitting on the limited Cleveland dataset). Outlined strategies to deliver immediate value and fail fast through rapid iteration.
