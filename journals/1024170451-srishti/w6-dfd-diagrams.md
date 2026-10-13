# 🚀 Week 6: Hierarchical Data Flow Modeling (Level-0 & Level-1 DFD)

**📅 Date:** September 18 – September 24, 2026  
**🎯 Core Task:** Designed and documented the hierarchical Data Flow Diagrams (Level 0 Context Diagram and Level 1 Functional Decomposition) to formally specify information boundaries, process transformations, external entities, and data stores for the CardioAI clinical decision support system.

---

### 🛠️ Key Contributions & Decisions

#### 1. Level 0 Data Flow Diagram (Context Diagram)
* **System Boundary & Primary Transformations:**
  * Modeled Process `0` (*CardioAI Heart Disease Prediction System*) as the high-level operational engine interfacing with external entities.
* **External Entities (Terminators):**
  * **Clinician / Cardiologist:** Supplies 13 patient physiological vitals and longitudinal search queries; receives diagnostic probability scores, XAI visual explanations (Grad-CAM, SHAP), and generated clinical PDF reports.
  * **Database Administrator (DBA):** Issues maintenance commands and queries; receives system audit logs and execution metrics.
  * **UCI Cleveland Dataset Repository:** Source for historical cardiac training and benchmarking records.
  * **ACC/AHA Clinical Guidelines:** Provides domain-specific evaluation thresholds and risk categorization rules.
* **Balanced Boundary Flows:**
  * Enforced strictly balanced information exchanges conforming to standard Yourdon & DeMarco DFD notations.

#### 2. Level 1 Data Flow Diagram (Functional Decomposition)
* **Sub-Process Decomposition:** Factored the system into modular, single-responsibility operational processes:
  * `1.0 Receive & Validate Patient Data`: Schema verification, type checking, and boundary validation of input markers.
  * `2.0 Preprocess Clinical Features`: Handles median imputation, `RobustScaler` normalization, and 2D Gramian Angular Field (GAF) transformation.
  * `3.0 Predict Heart Disease Risk`: Dispatches preprocessed tensors to the multimodal DL inference pipeline.
  * `4.0 Generate XAI Explanations`: Computes feature attributions via Grad-CAM heatmaps, SHAP values, and attention weights.
  * `5.0 Apply Clinical Decision Support`: Maps predicted risk indices against ACC/AHA guidelines to derive treatment suggestions.
  * `6.0 Generate Diagnostic Report`: Compiles clinical dossiers and visualization plots into downloadable reports.
  * `7.0 Store Patient Records`: Handles transactional persistence of clinical data and prediction results.
  * `8.0 Search Patient History`: Retrieves longitudinal records for trend analysis.
  * `9.0 Manage Database`: Coordinates administrative table maintenance, indexing, and backup routines.
* **Data Stores (Data at Rest):**
  * Formally delineated data stores separating persistent tables (`D1: Patient Database`, `D2: Clinical Records`, `D3: AI Diagnoses`, `D4: Trained Model Weights`) from active process pipelines.

---

### 📎 Attachments
* [Level 0 DFD - Context Diagram (PDF)](https://github.com/dsingla3be24/UCS503P-202627-Explainable-Heart-Disease-Prediction-/blob/master/diagrams/dfd_level_0.pdf)
* [Level 0 DFD - Context Diagram (.drawio)](https://github.com/dsingla3be24/UCS503P-202627-Explainable-Heart-Disease-Prediction-/blob/master/diagrams/dfd_level_0.drawio)
* [Level 1 DFD - Functional Decomposition (PDF)](https://github.com/dsingla3be24/UCS503P-202627-Explainable-Heart-Disease-Prediction-/blob/master/diagrams/level_1_dfd_Diagram.pdf)
* [Level 1 DFD - Functional Decomposition (.drawio)](https://github.com/dsingla3be24/UCS503P-202627-Explainable-Heart-Disease-Prediction-/blob/master/diagrams/dfd_level1.drawio)
