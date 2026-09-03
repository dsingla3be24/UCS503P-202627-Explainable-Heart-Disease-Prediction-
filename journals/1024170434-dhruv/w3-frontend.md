# 🚀 Week 3: Frontend Architecture & UI Prototyping

**📅 Date:** August 28 – September 3, 2026  
**🎯 Core Task:** Designed and implemented the complete Frontend Single Page Application (SPA) prototype for the CardioAI diagnostic dashboard, ensuring a premium medical cybernetic aesthetic.

---

### 🛠️ Key Contributions & Decisions

* **UI/UX Prototype Design:** Developed a premium dark-mode clinical dashboard using "Glassmorphism" principles (translucent backgrounds with `backdrop-filter: blur()`).
* **SPA Architecture (HTML5 & Bootstrap 5):** 
  * Structured the UI into three distinct states without page reloads: **Data Acquisition** (Parameter Input), **System Processing** (Scanning Overlay), and **Analytics Dashboard**.
  * Built a responsive 3-column grid layout to capture 13 critical clinical features seamlessly.
* **Dynamic Animations & Feedback:** 
  * Integrated **Vanta.js** and **Three.js** to render an interactive 3D particle network background.
  * Created custom CSS animations including an SVG sinus rhythm (ECG) drawing, scanning radar sweep, and pulse indicators mapped to diagnostic risk.
* **Frontend Logic (JavaScript):** Wrote vanilla JavaScript (`dashboard-main.js`) to handle DOM manipulation, form validation, and complex state transitions.
* **Mock Data Integration:** Configured realistic mock JSON responses and Plotly.js chart containers so the frontend prototype operates fully standalone before backend integration.

### 📎 Attachments
* [Frontend Source Code (GitHub)](../../code/frontend/)
* [index.html](../../code/frontend/index.html)
* [dashboard-main.js](../../code/frontend/static/js/dashboard-main.js)
