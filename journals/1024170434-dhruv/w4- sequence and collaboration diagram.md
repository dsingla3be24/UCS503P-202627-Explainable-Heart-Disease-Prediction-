# 🚀 Week 4: Interaction Modeling & Dynamic System Behavior

**📅 Date:** September 4 – September 10, 2026  
**🎯 Core Task:** Architected and documented the dynamic behavioral models of the CardioAI system by designing the UML Sequence and Collaboration diagrams.

---

### 🛠️ Key Contributions & Decisions

#### 1. Sequence Diagram Design
*   **Chronological Execution Flow:** Mapped the strict chronological order of operations across the 5 primary lifelines (`Clinician`, `:Frontend WebApp`, `:Flask Server`, `:XAI Module`, and `:Database`).
*   **Synchronous Message Passing:** Documented the 13-step interaction pipeline triggered by the clinician entering patient data.
*   **Self-Delegation:** Explicitly modeled recursive processing nodes, including the Flask Server triggering data preprocessing and PDF report generation, and the XAI Module computing localized interpretability metrics (Grad-CAM, SHAP, Attention).

#### 2. Collaboration (Communication) Diagram Design
*   **Structural Object Relationships:** Translated the chronological sequence into a topological interaction map, emphasizing the structural links between system nodes rather than time.
*   **Actor & Object Notation:** Adhered to strict UML conventions, utilizing actor stickmen for external entities (Clinician, Database) and underlined rectangles for internal system objects.
*   **Message Routing:** Visualized the flow of stimulus and return data along the fixed communication links, highlighting the central orchestration role of the Flask Server in dispatching tasks to the Database and XAI components.

#### 3. Alignment & Verification
*   **Consistency Check:** Verified that both dynamic diagrams perfectly align with the static system boundaries defined in Week 2's Use Case model and the procedural flow of the Activity Diagram.

---

### 📎 Attachments
* [Sequence Diagram (PDF)](https://github.com/dsingla3be24/UCS503P-202627-Explainable-Heart-Disease-Prediction-/tree/master/diagrams/sequence%20diagram.pdf)
* [Sequence Diagram (.drawio)](https://github.com/dsingla3be24/UCS503P-202627-Explainable-Heart-Disease-Prediction-/tree/master/diagrams/sequence_diagram.drawio)
* [Collaboration Diagram (PDF)](https://github.com/dsingla3be24/UCS503P-202627-Explainable-Heart-Disease-Prediction-/tree/master/diagrams/collaboration%20diagram.pdf)
* [Collaboration Diagram (.drawio)](https://github.com/dsingla3be24/UCS503P-202627-Explainable-Heart-Disease-Prediction-/tree/master/diagrams/collaboration_diagram.drawio)
