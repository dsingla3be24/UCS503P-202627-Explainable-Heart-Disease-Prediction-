# 🚀 Week 3: Frontend Architecture & UI Prototyping

**📅 Date:** August 28 – September 3, 2026  
**🎯 Core Task:** Designed and implemented the complete Frontend Single Page Application (SPA) prototype for the CardioAI diagnostic dashboard, ensuring a premium medical cybernetic aesthetic.

---

### 🛠️ Key Contributions & Decisions

#### 1. Technology Stack (Frontend Only)
*   **Core Structure:** HTML5 (utilizing Jinja2 templating syntax structure for future backend integration).
*   **Styling & Layout:** Custom CSS3 paired with **Bootstrap 5** for responsive grid layouts.
*   **Typography:** Google Fonts (`Outfit` for headers, `Plus Jakarta Sans` for body text) to enhance readability.
*   **Dynamic Interactions:** Vanilla JavaScript (ES6+ modular approach) implemented in `dashboard-main.js`.
*   **Data Visualization:** **Plotly.js** containers prepared for rendering interactive XAI charts.
*   **3D Graphics:** **Three.js** combined with **Vanta.js** (NET topology effect) for the dynamic, hardware-accelerated background.

#### 2. UI/UX Design Principles
Moved away from traditional, sterile medical forms to a "Cybernetic / Medical Dashboard" aesthetic:
*   **Glassmorphism:** UI components (cards, modals, inputs) utilize semi-transparent backgrounds with `backdrop-filter: blur(20px)` and subtle CSS box-shadows. This creates depth and hierarchy without cluttering the screen.
*   **3D CSS Transforms:** Cards feature parallax hover effects (`transform-style: preserve-3d`), providing tactile feedback when users interact with the diagnostic panels.
*   **Color Psychology:** A dark theme (`#030712`) reduces eye strain for clinical users operating in low-light environments. Status indicators use intuitive semantic colors (Neon Blue for nominal, Neon Rose for critical risk, Emerald Green for success).

#### 3. Application Architecture & Layout (SPA States)
The interface is structured into three primary states, dynamically toggled via JavaScript Document Object Model (DOM) manipulation to prevent page reloads:

*   **State 1: Parameter Input (Data Acquisition Module)**
    *   A 3-column responsive grid capturing 13 critical physiological features.
    *   **Column A:** Demographics & Clinical Symptoms (Age, Gender, Chest Pain type). Features dynamic toggles for "New" vs "Existing" patients.
    *   **Column B:** Cardiac Diagnostics (Resting BP, Cholesterol, ECG). Utilizes synchronized dual-input controls (Range Sliders linked to Number Inputs).
    *   **Column C:** ECG Waveform & Imaging (ST Depression, ST Slope).
*   **State 2: System Processing (Scanning Overlay)**
    *   Upon submission, the UI triggers a non-blocking diagnostic scan overlay.
    *   Features CSS-based radar animations, an SVG sinus rhythm (ECG) drawing animation, and a synchronized progress bar to inform the user of simulated computation stages.
*   **State 3: Results Dashboard (Analytics Module Prototype)**
    *   **Risk Analysis Card:** Displays the binary classification probability with a dynamic progress bar and visual risk indicators (HIGH/LOW RISK).
    *   **Model Performance Metrics:** Real-time rendering of the model's localized confidence (Accuracy, Precision, Recall, F1, ROC-AUC).
    *   **Interactive Charting Areas:** Configured Plotly.js containers to render Explainable AI (XAI) charts (Bar charts and Heatmaps) once real data is integrated.

#### 4. Client-Side Validation & Interaction
*   **Form Validation:** HTML5 attributes (`min`, `max`, `step`, `required`) ensure data integrity before submission.
*   **Mock Data Integration:** Configured realistic mock JSON responses so the frontend prototype operates fully standalone and demonstrates realistic UI behavior before backend integration.

---

### 📎 Attachments
* [Frontend Source Code (GitHub)](../../code/frontend/)
* [index.html](../../code/frontend/index.html)
* [dashboard-main.js](../../code/frontend/static/js/dashboard-main.js)
