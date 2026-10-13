# 🚀 Week 6: Deployment Modeling & System Physical Architecture

**📅 Date:** September 18 – September 24, 2026  
**🎯 Core Task:** Designed and documented the UML Deployment Diagram specifying the runtime environment, hardware nodes, and execution topologies for the CardioAI clinical prediction system.

---

### 🛠️ Key Contributions & Decisions

* **Node & Hardware Topologies:**
  * Modeled the multi-tier distributed execution architecture across the primary physical and virtual nodes:
    1. **Client Tier (`«Device»`):** Clinician workstations and mobile endpoints running the sandboxed browser client to interact with the CardioAI web dashboard and diagnostic viewers.
    2. **Web / Application Server (`«Device»`):** Flask runtime execution environment hosting the core prediction pipelines, input validation modules, and XAI engines.
    3. **Persistence Layer (`«Device»`):** Relational MySQL database server container managing patient records, diagnostic history, and system audit logs.
* **Component Allocation & Artifact Mapping:**
  * Mapped operational artifacts (`app.py`, trained model weights, static web assets, database schemas) to their respective execution environments.
* **Communication Protocols & Interface Boundaries:**
  * Defined communication channels and security interfaces between nodes, including HTTPS (REST API) between client and server tiers, and MySQL TCP sockets for database persistence.
* **Diagram Assets & Documentation:**
  * Generated and exported the source `.drawio` and vector `.pdf` diagram assets into the project repository.

---

### 📎 Attachments
* [Deployment Diagram (PDF)](https://github.com/dsingla3be24/UCS503P-202627-Explainable-Heart-Disease-Prediction-/blob/master/diagrams/deployment_diagram.pdf)
* [Deployment Diagram (.drawio)](https://github.com/dsingla3be24/UCS503P-202627-Explainable-Heart-Disease-Prediction-/blob/master/diagrams/deployment_diagram.drawio)
