# 🚀 Week 5: Static Structural Architecture & Modular Component Engineering

**📅 Date:** October 4 – October 11, 2026  
**🎯 Core Task:** Architected the complete static structural blueprint and modular component architecture of the CardioAI diagnostic platform through the design of the UML Class Diagram and UML Component Diagram.

---

### 🛠️ Key Contributions & Decisions

#### 1. UML Class Diagram Engineering
*   **Domain Entity Modeling & Generalization:**
    *   Designed the abstract base entity `User` (`userId`, `username`, `passwordHash`, `email`, with polymorphic `login()` and `logout()` operations).
    *   Derived concrete specialized roles `Patient` (capturing demographic vitals: `age`, `gender`, `contactInfo`) and `Doctor` (encapsulating clinical credentials: `licenseNum`, `specialization`, with `addNotes()` and `fetchHistory()`).
*   **Composition & Lifecycle Ownership:**
    *   Formulated strict black-diamond composite relationships where a `Patient` owns instances of `HealthRecord` (`recordId`, `date`, `bloodPressure`, `cholesterol`, `validateMetrics()`) and `PredictionResult` (`resultId`, `riskProb`, `predictedClass`, `doctorNotes`). Deletion or archiving of a patient cascades to their corresponding diagnostic records.
*   **Aggregation & Clinical Association:**
    *   Modeled white-diamond aggregation linking `Hospital` to its affiliated `Doctor` staff (`getDoctors()`), acknowledging independent entity lifecycles.
    *   Specified bidirectional association `treats` between `Doctor` and `Patient`.
*   **Service & Pattern Integration:**
    *   Introduced `MLPredictionService` (`modelVersion`, `predictRisk()`) for decoupling inference execution from domain entities.
    *   Architected `DatabaseController` adhering to the `<<Singleton>>` design pattern (`connectionStr`, `saveRecord()`, `savePrediction()`) to maintain a unified database connection pool.
*   **OCL Invariants & Domain Constraints:**
    *   Formally attached UML constraint notes specifying critical invariant boundaries:
        *   `{ age > 0 and age < 130 }` on `Patient`
        *   `{ 0.0 <= riskProb <= 1.0 }` on `PredictionResult`
        *   `{ cholesterol > 0.0 }` on `HealthRecord`

#### 2. UML Component Diagram Design
*   **Subsystem Boundary & Modular Decomposition:**
    *   Enclosed internal modules within the `CardioAISystem` subsystem (`<<Component>>`), establishing high cohesion and loose coupling.
    *   Structured three core processing components:
        1.  `FlaskAPI` (`<<Component>>`): Responsible for REST route dispatching, JSON schema validation, and session orchestration.
        2.  `MLCore` (`<<Component>>`): Dedicated deep learning computational engine executing GAF encoding, CNN-LSTM-Attention inference, and XAI attribution generation.
        3.  `Database` (`<<Component>>`): Encapsulating persistent relational data transactions and query execution.
*   **Port & Interface Specification (Ball-and-Socket Notation):**
    *   **Delegation Connector:** Exposed external boundary port delegating the `PredictionAPI (Patient input)` interface (`<<delegate>>`) directly into `FlaskAPI`.
    *   **Inference Interface:** Defined the `InferenceService` interface provided by `MLCore` (lollipop) and required by `FlaskAPI` (socket).
    *   **Persistence Interface:** Defined the `DatabaseAccess` interface provided by `Database` and consumed by `FlaskAPI`.
*   **Physical Artifact Dependency:**
    *   Explicitly mapped the `<<use>>` dependency from `MLCore` to the physical neural network artifact `model_best_weights.h5`, visualizing where trained binary parameters are ingested during system startup.

#### 3. Cross-Model Structural Consistency
*   **Static-to-Dynamic Traceability:** Verified that every method in the Class Diagram (`validateMetrics()`, `predictRisk()`, `saveRecord()`) corresponds directly to the interaction sequence paths mapped in Week 4's Sequence & Collaboration diagrams.
*   **Component-to-Deployment Alignment:** Confirmed that the `FlaskAPI`, `MLCore`, and `Database` components correspond directly to the execution artifacts and physical nodes defined in the Deployment model.

---

### 📎 Attachments
* [Class Diagram (PDF)](https://github.com/dsingla3be24/UCS503P-202627-Explainable-Heart-Disease-Prediction-/tree/master/diagrams/class%20diagram.pdf)
* [Class Diagram (.drawio)](https://github.com/dsingla3be24/UCS503P-202627-Explainable-Heart-Disease-Prediction-/tree/master/diagrams/class%20diagram.drawio)
* [Component Diagram (PDF)](https://github.com/dsingla3be24/UCS503P-202627-Explainable-Heart-Disease-Prediction-/tree/master/diagrams/component%20diagram.pdf)
* [Component Diagram (.drawio)](https://github.com/dsingla3be24/UCS503P-202627-Explainable-Heart-Disease-Prediction-/tree/master/diagrams/component%20diagram.drawio)
