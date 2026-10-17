# Cardio AI — Application Code

Explainable Heart Disease Prediction using a CNN-LSTM-Attention network with
Grad-CAM / Attention / SHAP interpretability.

This folder contains the **complete runnable application** (Flask backend +
ML/XAI pipeline + integrated web dashboard).

```
code/
├── app.py                    # Flask REST API + serves the dashboard
├── main.py                   # CLI: EDA, training, ablation, XAI, reports
├── src/
│   ├── preprocessing.py      # cleaning, ClinicalRangeScaler, GAF (GASF) encoding
│   ├── models.py             # DB helpers + model builders
│   ├── train.py              # train/eval, plots, cross-validation
│   ├── explainability.py     # Grad-CAM 2D, Attention, SHAP, narrative
│   ├── pdf_generator.py      # clinical PDF report generator
│   ├── fallback_model.py     # ClinicalModelWrapper (predict + explain)
│   └── numpy_model.py        # pure-NumPy reference CNN-LSTM-Attention
├── lib/tensorflow.py         # lightweight Keras/TF emulator (no native DLLs)
├── templates/index.html      # integrated dashboard (Jinja)
├── static/{css,js}           # dashboard assets (Bootstrap, Plotly, Three, Vanta)
├── results/                  # EDA plots, confusion matrices, ROC curves, weights
├── clinical_guidelines.json  # ACC/AHA style rules used in narratives
├── heart_disease_cleveland.csv
├── requirements.txt
├── docker-compose.yml        # MySQL 8.0 + phpMyAdmin
├── init.sql                  # schema for cardio_ai_db
├── backend/                  # Week-2 CNN baseline notebook (prototype stage)
└── frontend/                 # Week-3 static UI prototype (mock, no backend)
```

> `backend/` and `frontend/` preserve the earlier prototype-stage artifacts that
> are referenced by the project report. The **live** frontend is served by
> `app.py` from `templates/` + `static/`.

## 1. Run the web app

```bash
cd code
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
# open http://localhost:5001
```

On first run the model is trained on the UCI Cleveland dataset (fast, CPU only)
and stored in `results/`. Predictions trigger the XAI engine and, if Chrome/Edge
is installed, a PDF clinical report (`static/temp/`).

## 2. Run the full offline pipeline (EDA + training + XAI)

```bash
cd code
python main.py                      # full pipeline -> results/
python main.py --cv                 # add 5-fold cross validation
python main.py --patient_idx 3      # per-patient XAI + PDF
```

## 3. Optional: MySQL persistence

```bash
cd code
docker compose up -d                # MySQL on :3307, phpMyAdmin on :8080
```

`app.py` writes patients/records/diagnoses to `cardio_ai_db`. If the database is
unreachable the API still returns predictions (DB errors are logged and skipped).

## API

| Method | Route | Purpose |
| --- | --- | --- |
| `GET`  | `/` | Dashboard UI |
| `POST` | `/api/predict` | Predict risk + return XAI data + PDF URL |
| `GET`  | `/api/search?q=` | Search longitudinal patient history |
| `GET`  | `/static/temp/<file>` | Serve generated PDF/HTML reports |
