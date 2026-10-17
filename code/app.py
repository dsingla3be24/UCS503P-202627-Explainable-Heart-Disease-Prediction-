import os
import sys

# Inject local lib directory so the mock TensorFlow emulator loads
# instead of the broken native DLL (Hyper-V/VBS DllMain failure)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'lib')))

import uuid
import json
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_from_directory
from src.fallback_model import ClinicalModelWrapper
from src.pdf_generator import generate_pdf_report
from src.models import build_cnn_lstm_attention, compile_model, insert_patient_record, search_patient_history, get_or_create_patient, get_patient_last_demographics

app = Flask(__name__)

# Configure local static and templates folders
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(BASE_DIR, "static", "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

# Instantiate the machine learning wrapper model
print("Initializing Heart Disease Diagnosis ML Backend...")
model_wrapper = ClinicalModelWrapper(dataset_path=os.path.join(BASE_DIR, "heart_disease_cleveland.csv"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Receives custom patient parameters, runs predictions, XAI attributions,
    interactive PyVis graph compilation, and PDF report creation.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        patient_name = data.get('patient_name', '')
        provided_patient_id = data.get('patient_id', '')
        patient_type = data.get('patient_type', 'new')
        
        # Unique session ID for output files
        session_id = str(uuid.uuid4())[:8]
        
        # 1. Deduplicate/Create Patient in DB first
        try:
            patient_db_id, hospital_patient_id, final_patient_name = get_or_create_patient(patient_name, session_id, provided_patient_id)
        except Exception as db_e:
            print(f"[DB ERROR] Skipping DB: {db_e}")
            patient_db_id = 1
            hospital_patient_id = provided_patient_id.strip() if provided_patient_id and provided_patient_id.strip() else f"PT-{session_id.upper()}"
            final_patient_name = patient_name if patient_name and patient_name.strip() else f"Patient {session_id.upper()}"

        # Handle demographic override for existing patients
        age_input = float(data.get('age', 55.0))
        sex_input = float(data.get('sex', 0.0))
        
        if patient_type == 'existing' and provided_patient_id:
            from datetime import datetime
            try:
                old_age, old_sex, old_recorded_at = get_patient_last_demographics(patient_db_id)
                if old_age is not None and old_recorded_at is not None:
                    years_passed = (datetime.now() - old_recorded_at).days / 365.25
                    age_input = float(old_age) + float(years_passed)
                    sex_input = float(old_sex)
            except Exception as db_e:
                print(f"[DB ERROR] Skipping demographics fetch: {db_e}")
                
        # Standardize features mapping to numeric variables
        raw_features = {
            'age': age_input,
            'sex': sex_input,
            'cp': float(data.get('cp', 4.0)),
            'trestbps': float(data.get('trestbps', 130.0)),
            'chol': float(data.get('chol', 240.0)),
            'fbs': float(data.get('fbs', 0.0)),
            'restecg': float(data.get('restecg', 0.0)),
            'thalach': float(data.get('thalach', 150.0)),
            'exang': float(data.get('exang', 0.0)),
            'oldpeak': float(data.get('oldpeak', 1.0)),
            'slope': float(data.get('slope', 2.0)),
            'ca': float(data.get('ca', 0.0)),
            'thal': float(data.get('thal', 3.0))
        }
        
        # Auto-detect matching patient from dataset based on raw input parameters
        csv_path = os.path.join(BASE_DIR, "heart_disease_cleveland.csv")
        df = pd.read_csv(csv_path)
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].replace({'?': None, '': None})
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.fillna(df.median())

        csv_sex = 1.0 if raw_features['sex'] == 0.0 else 0.0
        
        min_distance = float('inf')
        closest_target = None
        
        for idx, row in df.iterrows():
            dist = 0.0
            dist += abs(row['age'] - raw_features['age']) / (77 - 29)
            dist += abs(row['trestbps'] - raw_features['trestbps']) / (200 - 94)
            dist += abs(row['chol'] - raw_features['chol']) / (564 - 126)
            dist += abs(row['thalach'] - raw_features['thalach']) / (202 - 71)
            dist += abs(row['oldpeak'] - raw_features['oldpeak']) / 6.2
            
            dist += 1.0 if row['sex'] != csv_sex else 0.0
            dist += 1.0 if row['cp'] != raw_features['cp'] else 0.0
            dist += 1.0 if row['fbs'] != raw_features['fbs'] else 0.0
            dist += 1.0 if row['restecg'] != raw_features['restecg'] else 0.0
            dist += 1.0 if row['exang'] != raw_features['exang'] else 0.0
            dist += 1.0 if row['slope'] != raw_features['slope'] else 0.0
            dist += 1.0 if row['ca'] != raw_features['ca'] else 0.0
            dist += 1.0 if row['thal'] != raw_features['thal'] else 0.0
            
            if dist < min_distance:
                min_distance = dist
                closest_target = int(row['target'])
                
        # If distance is extremely small, we consider it a match
        if min_distance < 0.05:
            true_label = 1 if closest_target >= 1 else 0
        else:
            true_label = None
            
        # Run prediction
        prob = model_wrapper.predict_proba(raw_features)
        
        # Run explainability (XAI)
        explain_results = model_wrapper.explain(raw_features, prob)
        
        
        # Generate PDF report
        pdf_filename, clinical_details = generate_pdf_report(
            raw_patient_row=pd.Series(raw_features),
            importance_weights=explain_results['attention'],
            original_features=model_wrapper.original_features,
            predicted_prob=prob,
            patient_idx=final_patient_name,
            hospital_patient_id=hospital_patient_id,
            output_dir=TEMP_DIR,
            true_label=true_label,
            shap_weights=explain_results['shap']
        )
        
        # Map variables for Plotly charts in frontend
        original_features = list(model_wrapper.original_features)
        display_features = original_features
        
        plotly_xai_data = {
            'features': display_features,
            'processed_features': [f.split('_')[0] if '_' in f else f for f in model_wrapper.processed_feature_names],
            'attention': [float(val) for val in explain_results['attention']] if explain_results['attention'] is not None else [0.0] * len(original_features),
            'grad_cam': [float(val) for val in explain_results['grad_cam']] if explain_results['grad_cam'] is not None else [0.0] * len(original_features),
            'shap': [float(val) for val in explain_results['shap']] if explain_results['shap'] is not None else [0.0] * len(original_features)
        }
        
        # Calculate performance metrics dynamically according to the patient's demographic input subgroup
        subgroup_metrics = model_wrapper.calculate_subgroup_metrics(raw_features)
        
        response_data = {
            'probability': prob,
            'prediction_label': "HEART DISEASE RISKS DETECTED" if prob >= 0.5 else "LOW RISK / NORMAL",
            'xai': plotly_xai_data,
            'gaf': explain_results['gaf'],
            'grad_cam_2d': explain_results['grad_cam_2d'],
            'pdf_url': f"/static/temp/{pdf_filename}" if pdf_filename else None,
            'clinical_details': clinical_details,
            'true_target': true_label,
            'metrics': subgroup_metrics,
            'patient_name': patient_name,
            'hospital_patient_id': hospital_patient_id
        }
        
        # Save to database using raw MySQL
        try:
            # Extract top 3 driving features by attention weight for the database
            feat_weights = list(zip(original_features, explain_results['attention']))
            feat_weights.sort(key=lambda x: x[1], reverse=True)
            top_features_str = ", ".join([f[0] for f in feat_weights[:3]])
            
            insert_patient_record(
                patient_id=patient_db_id,
                raw_features=raw_features,
                prob=prob,
                top_features_str=top_features_str,
                pdf_filename=f"/static/temp/{pdf_filename}" if pdf_filename else None
            )
            print(f"[DB SUCCESS] Patient {patient_name} ({hospital_patient_id}) and diagnosis saved.")
        except Exception as db_e:
            print(f"[DB ERROR] Failed to save to database: {db_e}")
        
        return jsonify(response_data)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error in prediction endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/search', methods=['GET'])
def search_patients():
    query = request.args.get('q', '').strip()
        
    try:
        results = search_patient_history(query)
        return jsonify(results)
    except Exception as e:
        import traceback
        traceback.print_exc()
        # Degrade gracefully so the dashboard keeps working when the optional
        # MySQL backend is not running. Start it with: docker compose up -d
        print(f"[DB WARNING] Patient search unavailable (is MySQL running?): {e}")
        return jsonify([])

@app.route("/static/temp/<path:filename>")
def serve_temp_files(filename):
    return send_from_directory(TEMP_DIR, filename)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print(f"Starting CardioAI Flask Web Application on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)
