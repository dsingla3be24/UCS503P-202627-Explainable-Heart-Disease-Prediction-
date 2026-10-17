# 🚀 Week 5: Physical Deployment Architecture & Hierarchical Data Flow Modeling

**📅 Date:** October 4 – October 11, 2026  
**🎯 Core Task:** Designed and finalized the system's physical node deployment topology (UML Deployment Diagram) and hierarchical Data Flow Diagrams (Level 0 Context Diagram and Level 1 Functional Decomposition) for the CardioAI clinical decision support ecosystem.

---

### 🛠️ Key Contributions & Decisions

#### 1. UML Deployment Diagram Design
*   **Multi-Tier Physical & Virtual Topology:** Modeled the distributed runtime environment across four key execution tiers:
    1.  **Client Device (`«Device»`):** Configured clinician workstations and mobile tablet endpoints hosting a sandboxed Web Browser execution environment to deliver the CardioAI Web Dashboard UI, interactive PyVis XAI visualizer, and PDF dossier viewer.
    2.  **Web Services Gateway (`«Execution Environment»`):** Positioned an intermediate REST API reverse proxy/dispatch layer handling incoming HTTPS requests (Port 5000) with JSON payload validation.
    3.  **Application Server (`«Device»`):** Dedicated backend node running Python 3.10 / Flask Runtime, encapsulating seven core computational modules:
        *   *Flask Web API Controller* (input sanitization & routing),
        *   *Clinical Data Preprocessing* (median imputation & feature scaling),
        *   *Gramian Angular Field (GAF) Generator* (1D-to-2D image transformation),
        *   *CNN-LSTM-Attention DL Engine* (spatial-temporal inference),
        *   *Explainable AI (XAI) Engine* (Grad-CAM heatmaps & SHAP values),
        *   *Clinical Decision Support* (ACC/AHA guideline validation), and
        *   *Diagnostic Report Generator* (ReportLab clinical dossier compiler).
    4.  **Data Storage Server (`«Device»`):** Containerized MySQL 8.0 DBMS executing over an internal Docker network with persistent Docker named volumes (`mysql_data`), initialized via `init.sql`.
    5.  **Database Administration Container (`«Execution Environment»`):** Deployed `cardio_ai_phpmyadmin` (Port 8080) for administrative health monitoring and SQL query management.
*   **Protocols & Inter-Node Communication:** Specified secure network protocols including HTTPS (TLS Port 5000) for client-server REST calls, MySQL Native Protocol (TCP Port 3307:3306) with lollipop socket interfaces for database persistence, and internal Docker bridge networking.
*   **Artifact Mapping:** Documented physical system deliverables, linking executable artifacts (`app.py`, `init.sql`, `index.html`) and model binaries (`model_best_weights.h5`, `clinical_guidelines.json`) to their runtime hosts.

#### 2. Level 0 Data Flow Diagram (Context Diagram)
*   **System Boundary Definition:** Modeled Process `0` (*CardioAI Heart Disease Prediction System*) as the central black-box transformation unit.
*   **External Entities (Terminators):** Mapped four primary stakeholders and external resources:
    *   *Clinician / Cardiologist:* Ingests 13 clinical biomarkers and patient search queries; receives risk predictions, XAI visual explanations, ACC/AHA recommendations, and generated PDF reports.
    *   *Database Administrator (DBA):* Issues SQL maintenance queries and schema commands; receives execution status and system audit logs.
    *   *UCI Cleveland Dataset Repository:* Supplies historical cardiac benchmark data (303 records, 14 attributes) for model training and cross-validation.
    *   *ACC/AHA Clinical Guidelines:* Ingests evidence-based clinical risk rules and evaluation thresholds.
*   **Balanced Data Pipelines:** Standardized all incoming and outgoing data flows using Yourdon & DeMarco notation with strictly balanced inputs and outputs.

#### 3. Level 1 Data Flow Diagram (Functional Decomposition)
*   **Process Decomposition:** Factored the central system into nine interconnected, single-responsibility sub-processes:
    *   `1.0 Receive & Validate Patient Data`: Handles range validation and schema sanitization of the 13 clinical parameters.
    *   `2.0 Preprocess Clinical Features`: Executes missing value imputation, `RobustScaler` normalization, and 2D Gramian Angular Field (GAF) matrix encoding.
    *   `3.0 Predict Heart Disease Risk`: Runs multimodal inference across the spatial CNN branch, temporal BiLSTM branch, and Multi-Head Self-Attention layers.
    *   `4.0 Generate XAI Explanations`: Computes convolutional Grad-CAM saliency heatmaps, SHAP feature attributions, and attention weight vectors.
    *   `5.0 Apply Clinical Decision Support`: Maps predicted risk against ACC/AHA cardiology guidelines to formulate individualized lifestyle and pharmacotherapy recommendations.
    *   `6.0 Generate Diagnostic Report`: Synthesizes patient vitals, DL predictions, visual XAI plots, and clinical recommendations into a publication-grade PDF report.
    *   `7.0 Store Patient Records`: Manages transactional persistence into relational database tables.
    *   `8.0 Search Patient History`: Queries multi-table patient timelines for longitudinal risk trend analysis.
    *   `9.0 Manage Database`: Coordinates administrative schema maintenance, table indexing, and database backups.
*   **Data Store Integration:** Formatted four internal open-ended repositories (`D1: Patient Database`, `D2: Clinical Records`, `D3: AI Diagnoses`, `D4: Trained Model Weights`) ensuring data at rest is cleanly separated from transient flows.

---

### 📎 Attachments
* [Deployment Diagram (PDF)](https://github.com/dsingla3be24/UCS503P-202627-Explainable-Heart-Disease-Prediction-/tree/master/diagrams/deployment_diagram.pdf)
* [Deployment Diagram (.drawio)](https://github.com/dsingla3be24/UCS503P-202627-Explainable-Heart-Disease-Prediction-/tree/master/diagrams/deployment_diagram.drawio)
* [Level 0 DFD - Context Diagram (PDF)](https://github.com/dsingla3be24/UCS503P-202627-Explainable-Heart-Disease-Prediction-/tree/master/diagrams/dfd_level_0.pdf)
* [Level 0 DFD - Context Diagram (.drawio)](https://github.com/dsingla3be24/UCS503P-202627-Explainable-Heart-Disease-Prediction-/tree/master/diagrams/dfd_level_0.drawio)
* [Level 1 DFD - Functional Decomposition (PDF)](https://github.com/dsingla3be24/UCS503P-202627-Explainable-Heart-Disease-Prediction-/tree/master/diagrams/level_1_dfd_Diagram.pdf)
* [Level 1 DFD - Functional Decomposition (.drawio)](https://github.com/dsingla3be24/UCS503P-202627-Explainable-Heart-Disease-Prediction-/tree/master/diagrams/dfd_level1.drawio)
