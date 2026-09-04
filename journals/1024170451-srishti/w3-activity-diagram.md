# 🚀 Week 3: System Flow Analysis & Activity Modeling

**📅 Date:** August 28 – September 3, 2026  
**🎯 Core Task:** Architected the complete system logic and developed a comprehensive UML Activity Diagram to map out the execution flow of the CardioAI diagnostic engine across multiple system boundaries.

---

### 🛠️ Key Contributions & Decisions

* **System Flow Analysis:** 
  * Analyzed the entire lifecycle of the application, from user data ingestion on the web interface to backend model prediction and PDF report generation.
  * Ensured logical consistency by mapping out all prerequisites, constraints, and data validation steps.

* **Swimlane Architecture:** 
  * Structured the UML Activity Diagram (`activity_diagram_6_complete_system_swimlanes.drawio`) using a 4-swimlane methodology to clearly define responsibilities across:
    1. **User / Clinician (Frontend)**
    2. **Flask Web Server (Middleware)**
    3. **ML Prediction Engine (Backend)**
    4. **MySQL Database (Storage)**

* **Path Optimization & Branching Logic:** 
  * Mapped out complex conditional routing, including Decision Nodes for valid/invalid JSON payloads.
  * Designed Parallel Processing flows (Fork/Join nodes) to demonstrate how Explainable AI models (SHAP, Grad-CAM, and Global Attention) compute feature attributions simultaneously after the primary CNN-LSTM inference.

* **Documentation & Rendering:** 
  * Compiled the final master diagrams (`activity diagram swimlanes.pdf` and `activity_diagram.pdf`) to serve as the official blueprint for the backend engineering phase.

### 📎 Attachments
* [Activity Diagram - Swimlanes (PDF)](../../diagrams/activity diagram swimlanes.pdf)
* [Activity Diagram - Standard (PDF)](../../diagrams/activity diagram_.pdf)
