# 🚀 Week 2: UML Use Case Diagram & Actor Modeling

**📅 Date:** August 21 – August 27, 2026  
**🎯 Core Task:** Designed the comprehensive UML Use Case Diagram mapping out all actor interactions and system boundaries for Cardio AI.

---

### 🛠️ Key Contributions & Decisions

* **Actor Identification:** Defined the primary and supporting actors for the system:
  * **Clinician (Primary):** Interacts directly with the web interface to enter clinical data, run predictions, view XAI charts, and manage patient history. Also interacts with the CLI for model training and EDA.
  * **Patient (Stakeholder):** The indirect beneficiary who receives the generated diagnostic reports and whose longitudinal records are tracked over time.
  * **ML Engine & MySQL Database (Supporting):** Handle backend inference (CNN-LSTM-Attention), XAI computation (Grad-CAM, SHAP, Attention), and persistent record storage.
* **Core Use Case Mapping:** Mapped out all primary use cases, including running exploratory data analysis (EDA), predicting heart disease risk, viewing explainability charts, downloading PDF diagnostic reports, and managing patient records.
* **Complex Relationships:** Structured advanced `«include»` and `«extend»` UML relationships. For example, predicting heart disease *includes* preprocessing data and running the model, while viewing XAI explanations *extends* the core prediction flow.
* **Diagram Generation:** Produced the final use case diagram to be included in the official project documentation.

### 📎 Attachments
* [Use Case Diagram (PDF)](../../diagrams/use_case_diagram.pdf)
* [Use Case Diagram (.drawio)](../../diagrams/use_case_diagram.drawio)
