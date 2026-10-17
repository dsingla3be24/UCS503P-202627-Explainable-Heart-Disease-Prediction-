import os
import json
import subprocess
import numpy as np
from datetime import datetime

def load_guidelines(filepath="clinical_guidelines.json"):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading clinical guidelines: {e}")
        return {}

def generate_narrative_summary(raw_features, prob, all_evaluated):
    gender = "woman" if raw_features['sex'] == 1.0 else "man"
    age = int(raw_features['age'])
    
    if prob >= 0.5:
        risk_level = "higher risk"
        recommendation = "We strongly recommend scheduling a follow-up appointment with a cardiologist. They may want to do a few more checks, like a heart ultrasound or a closer look at your cholesterol, just to make sure you stay healthy and get the right care."
    else:
        risk_level = "low risk"
        recommendation = "Keep up the good work! We recommend sticking to a healthy lifestyle, eating well, staying active, and coming in for your regular check-ups to maintain your heart health."
        
    points = []
    
    # 1. Consensus Assessment
    points.append(f"<strong>Overall Summary:</strong> Based on the information provided, the AI system indicates that as a {age}-year-old {gender}, you currently have a <strong>{risk_level}</strong> profile for developing heart-related issues (estimated at <strong>{prob:.0%}</strong> probability).")
    
    # 2. Vitals & Lipids
    bp_val = raw_features['trestbps']
    chol_val = raw_features['chol']
    vitals_parts = []
    if bp_val >= 140:
        vitals_parts.append(f"your resting blood pressure is quite high ({bp_val:.0f} mmHg)")
    elif bp_val >= 130:
        vitals_parts.append(f"your resting blood pressure is slightly elevated ({bp_val:.0f} mmHg)")
    else:
        vitals_parts.append(f"your resting blood pressure is in a very healthy range ({bp_val:.0f} mmHg)")
        
    if chol_val >= 240:
        vitals_parts.append(f"your cholesterol levels are higher than we'd like ({chol_val:.0f} mg/dl), which means there might be some plaque building up in your arteries")
    elif chol_val >= 200:
        vitals_parts.append(f"your cholesterol is borderline high ({chol_val:.0f} mg/dl)")
    else:
        vitals_parts.append(f"your cholesterol is in a great, healthy range ({chol_val:.0f} mg/dl)")
        
    points.append("<strong>Everyday Health Signs:</strong> Looking at your basics, " + ", and ".join(vitals_parts) + ".")
    
    # 3. Ischemic ECG Markers
    oldpeak = raw_features['oldpeak']
    slope = raw_features['slope']
    exang = raw_features['exang']
    
    ecg_parts = []
    if oldpeak >= 1.0 or slope in [2.0, 3.0] or exang == 1.0:
        if exang == 1.0:
            ecg_parts.append("you experienced some chest discomfort during exercise")
        if oldpeak >= 1.0 or slope in [2.0, 3.0]:
            ecg_parts.append("your heart's electrical signals changed a bit when it had to work harder")
            
        points.append("<strong>During Exercise:</strong> When looking at how your heart responds to physical stress, we noticed that " + " and ".join(ecg_parts) + ". This can sometimes mean your heart muscle isn't getting quite enough oxygen when it's pumping hard.")
    else:
        points.append("<strong>During Exercise:</strong> Great news here! Your heart's electrical signals look completely normal when put under physical stress, and you didn't experience any chest discomfort.")
        
    # 4. Anatomical/Structural Findings
    ca = raw_features['ca']
    thal = raw_features['thal']
    struct_parts = []
    
    if ca > 0:
        struct_parts.append(f"scans picked up some calcium buildup in {ca:.0f} of your main heart vessels")
    else:
        struct_parts.append("scans show that your main blood vessels look clear with no obvious blockages")
        
    if thal == 6.0:
        struct_parts.append("a specialized test showed a small area of the heart might have old damage")
    elif thal == 7.0:
        struct_parts.append("a specialized test indicated a small area of the heart might not be getting enough blood flow temporarily")
    else:
        struct_parts.append("blood is flowing beautifully to all areas of your heart muscle")
        
    points.append("<strong>Heart Structure & Blood Flow:</strong> Taking a deeper look at the heart itself, " + ", and ".join(struct_parts) + ".")
    
    # 5. Recommendation
    points.append(f"<strong>Doctor's Advice:</strong> {recommendation}")
    
    return points

def generate_pdf_report(raw_patient_row, importance_weights, original_features, 
                        predicted_prob, patient_idx, hospital_patient_id, output_dir="static/temp", 
                        true_label=None, shap_weights=None):
    """
    Generates a beautiful HTML report with the clinical details and explanation,
    and uses Microsoft Edge in headless mode to convert it to a PDF file.
    Save HTML and PDF into output_dir.
    """
    guidelines = load_guidelines()
    diag = "HEART DISEASE PREDICTED (HIGH RISK)" if predicted_prob >= 0.5 else "NORMAL / LOW RISK"
    diag_class = "danger" if predicted_prob >= 0.5 else "success"
    
    xai_title = "Explainability Consensus (Attention vs. Grad-CAM vs. SHAP)" if shap_weights is not None else "Explainability Consensus (Attention vs. Grad-CAM)"
    
    # Pre-evaluate ALL features so we know which ones are abnormal
    all_evaluated = {}
    for feat_name in original_features:
        val = raw_patient_row[feat_name]
        guideline = guidelines.get(feat_name)
        ideal = "N/A"
        reasoning = f"Value is {val}."
        is_abnormal = False
        
        if guideline:
            age = raw_patient_row['age']
            age_predicted_max = 220.0 - age
            age_predicted_max_85 = age_predicted_max * 0.85
            percent_max_hr = (val / age_predicted_max) * 100.0 if age_predicted_max != 0 else 0.0
            
            context = {
                'value': val,
                'age_predicted_max': age_predicted_max,
                'age_predicted_max_85': age_predicted_max_85,
                'percent_max_hr': percent_max_hr,
                **raw_patient_row.to_dict()
            }
            # Coerce NumPy variables to standard Python floats/ints
            context = {k: (v.item() if hasattr(v, 'item') else v) for k, v in context.items()}
            
            ideal_tmpl = guideline.get('ideal_range_label', 'N/A')
            try:
                ideal = ideal_tmpl.format(**context)
            except Exception:
                ideal = ideal_tmpl
                
            for rule in guideline.get('rules', []):
                op = rule.get('operator')
                comp_val_raw = rule.get('compare_value')
                interpretation_tmpl = rule.get('interpretation', '')
                
                if isinstance(comp_val_raw, str) and comp_val_raw in context:
                    comp_val = context[comp_val_raw]
                else:
                    comp_val = comp_val_raw
                
                op = op.strip() if isinstance(op, str) else op
                matched = False
                if op == 'default':
                    matched = True
                elif op == '>=':
                    matched = (val >= comp_val)
                elif op == '>':
                    matched = (val > comp_val)
                elif op == '<=':
                    matched = (val <= comp_val)
                elif op == '<':
                    matched = (val < comp_val)
                elif op == '==':
                    matched = (val == comp_val)
                elif op == 'in':
                    matched = (val in comp_val) if isinstance(comp_val, list) else (val == comp_val)
                    
                if matched:
                    is_abnormal = rule.get('abnormal', False)
                    try:
                        reasoning = interpretation_tmpl.format(**context)
                    except Exception:
                        reasoning = interpretation_tmpl
                    break
                    
        all_evaluated[feat_name] = {
            'value': val,
            'ideal': ideal,
            'reasoning': reasoning,
            'abnormal': is_abnormal
        }
        
    # Primary Model Drivers (top 4 based on importance weight)
    if shap_weights is not None:
        active_weights = (importance_weights + shap_weights) / 2.0
    else:
        active_weights = importance_weights
    sorted_idx = np.argsort(active_weights)[::-1]
    primary_html = ""
    top_4_features = []
    
    for rank, idx in enumerate(sorted_idx[:4], 1):
        feat_name = original_features[idx]
        top_4_features.append(feat_name)
        eval_data = all_evaluated[feat_name]
        feat_display = feat_name
        primary_html += f"""
        <div class="card driver-card">
            <h4>{rank}. {feat_display}</h4>
            <p><strong>Patient Value:</strong> {eval_data['value']} &nbsp;|&nbsp; <strong>Ideal Range:</strong> {eval_data['ideal']}</p>
            <p class="interpretation"><strong>Interpretation:</strong> {eval_data['reasoning']}</p>
        </div>
        """
        
    # Secondary Clinical Abnormalities (out-of-bounds features not in the top 4)
    secondary_html = ""
    sec_count = 1
    for feat_name in original_features:
        if feat_name not in top_4_features and all_evaluated[feat_name]['abnormal']:
            eval_data = all_evaluated[feat_name]
            feat_display = feat_name
            secondary_html += f"""
            <div class="card abnormal-card">
                <h4>{sec_count}. {feat_display}</h4>
                <p><strong>Patient Value:</strong> {eval_data['value']} &nbsp;|&nbsp; <strong>Ideal Range:</strong> {eval_data['ideal']}</p>
                <p class="interpretation"><strong>Interpretation:</strong> {eval_data['reasoning']}</p>
            </div>
            """
            sec_count += 1
            
    if not secondary_html:
        secondary_html = "<p>No other abnormalities detected.</p>"

    narrative_list = generate_narrative_summary(raw_patient_row, predicted_prob, all_evaluated)
    narrative_pdf_html = "<ul style='margin: 0; padding-left: 20px;'>" + "".join([f"<li style='margin-bottom: 8px;'>{pt}</li>" for pt in narrative_list]) + "</ul>"
    clinical_details = {
        'primary_html': primary_html,
        'secondary_html': secondary_html,
        'narrative': narrative_list
    }

    # Map raw variables to readable text labels for the doctor
    sex_val = "Female" if raw_patient_row['sex'] == 1.0 else "Male"
    cp_val = {1.0: "Typical Angina", 2.0: "Atypical Angina", 3.0: "Non-anginal Pain", 4.0: "Asymptomatic"}.get(raw_patient_row['cp'], str(raw_patient_row['cp']))
    fbs_val = "Elevated (>120 mg/dl)" if raw_patient_row['fbs'] == 1 else "Normal (<=120 mg/dl)"
    restecg_val = {0.0: "Normal", 1.0: "ST-T Wave Abnormality", 2.0: "Left Ventricular Hypertrophy"}.get(raw_patient_row['restecg'], str(raw_patient_row['restecg']))
    exang_val = "Yes" if raw_patient_row['exang'] == 1 else "No"
    slope_val = {1.0: "Upsloping", 2.0: "Flat", 3.0: "Downsloping"}.get(raw_patient_row['slope'], str(raw_patient_row['slope']))
    thal_val = {3.0: "Normal", 6.0: "Fixed Defect", 7.0: "Reversible Defect"}.get(raw_patient_row['thal'], str(raw_patient_row['thal']))
    
    # HTML Content
    html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Clinical Diagnosis Report - Patient #{patient_idx}</title>
<style>
    body {{ font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; color: #333; margin: 40px; line-height: 1.5; }}
    .header {{ border-bottom: 3px solid #0056b3; padding-bottom: 20px; margin-bottom: 20px; }}
    .header h2 {{ margin: 0 0 10px 0; color: #0056b3; }}
    .header p {{ margin: 0; font-size: 14px; color: #555; }}
    .meta-table {{ width: 100%; border-collapse: collapse; margin-bottom: 25px; }}
    .meta-table td {{ padding: 8px 10px; border: 1px solid #ddd; font-size: 13px; }}
    .meta-table td.label {{ font-weight: bold; background-color: #f9f9f9; width: 25%; }}
    .diagnosis-box {{ padding: 15px; border-radius: 5px; margin-bottom: 25px; font-size: 18px; font-weight: bold; color: white; }}
    .section-title {{ border-bottom: 2px solid #ddd; padding-bottom: 6px; color: #0056b3; font-size: 16px; margin-top: 25px; margin-bottom: 12px; font-weight: bold; }}
    .card {{ padding: 10px 15px; border-radius: 4px; margin-bottom: 12px; border-left: 5px solid; }}
    .driver-card {{ background-color: #e8f4fd; border-left-color: #0056b3; border: 1px solid #d0e8fc; border-left-width: 5px; }}
    .abnormal-card {{ background-color: #fff9e6; border-left-color: #ffc107; border: 1px solid #ffeeba; border-left-width: 5px; }}
    .card h4 {{ margin: 0 0 6px 0; color: #222; font-size: 14px; }}
    .card p {{ margin: 3px 0; font-size: 12px; }}
    .interpretation {{ font-style: italic; color: #444; }}
    .footer {{ text-align: center; font-size: 11px; color: #999; border-top: 1px solid #eee; padding-top: 15px; margin-top: 40px; }}
</style>
</head>
<body>
    <div class="header">
        <h1>HEART DISEASE PREDICTION REPORT</h1>
        <p style="margin-bottom:10px;">Explainable Deep Learning Diagnosis Platform (CNN-LSTM-Attention + Grad-CAM)</p>
        <p><strong>Patient Name:</strong> {patient_idx} | <strong>Patient ID:</strong> {hospital_patient_id}</p>
    </div>
    
    <table class="meta-table">
        <tr>
            <td class="label">Patient Index</td>
            <td>#{patient_idx}</td>
            <td class="label">True Label (Ground Truth)</td>
            <td>{"Heart Disease" if true_label == 1 else ("Normal" if true_label == 0 else "N/A")}</td>
        </tr>
        <tr>
            <td class="label">Model Prediction Probability</td>
            <td>{predicted_prob:.2%}</td>
            <td class="label">Analysis Date</td>
            <td>{datetime.now().strftime('%Y-%m-%d %I:%M %p')}</td>
        </tr>
    </table>
    
    <div class="diagnosis-box {diag_class}">
        Model Recommendation: {diag}
    </div>
    
    <div class="section-title">Patient Clinical Profile</div>
    <table class="meta-table">
        <tr>
            <td class="label">Age</td>
            <td>{raw_patient_row['age']:.0f} years</td>
            <td class="label">Sex</td>
            <td>{sex_val}</td>
        </tr>
        <tr>
            <td class="label">Chest Pain Type (CP)</td>
            <td>{cp_val}</td>
            <td class="label">Resting Blood Pressure</td>
            <td>{raw_patient_row['trestbps']:.0f} mm Hg</td>
        </tr>
        <tr>
            <td class="label">Serum Cholesterol</td>
            <td>{raw_patient_row['chol']:.0f} mg/dl</td>
            <td class="label">Fasting Blood Sugar (FBS)</td>
            <td>{fbs_val}</td>
        </tr>
        <tr>
            <td class="label">Resting ECG</td>
            <td>{restecg_val}</td>
            <td class="label">Max Heart Rate (Thalach)</td>
            <td>{raw_patient_row['thalach']:.0f} bpm</td>
        </tr>
        <tr>
            <td class="label">Exercise Induced Angina</td>
            <td>{exang_val}</td>
            <td class="label">ST Depression (Oldpeak)</td>
            <td>{raw_patient_row['oldpeak']:.1f}</td>
        </tr>
        <tr>
            <td class="label">ST Slope</td>
            <td>{slope_val}</td>
            <td class="label">Vessels via Fluoroscopy (CA)</td>
            <td>{raw_patient_row['ca']:.0f}</td>
        </tr>
        <tr>
            <td class="label">Thalassemia (Thal)</td>
            <td>{thal_val}</td>
            <td class="label">Predicted Diagnosis Probability</td>
            <td><strong>{predicted_prob:.2%}</strong></td>
        </tr>
    </table>

    <div class="section-title">Patient-Friendly Diagnostic Narrative Summary</div>
    <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-left: 5px solid #0056b3; padding: 15px; border-radius: 6px; font-size: 13px; line-height: 1.6; color: #334155; margin-bottom: 25px;">
        {narrative_pdf_html}
    </div>

    <div class="section-title">{xai_title}</div>
    {primary_html}
    
    <div class="section-title">Secondary Clinical Abnormalities (Outside Top AI Focus)</div>
    {secondary_html}
    
    <div class="footer">
        Confidential Medical Report - For Professional Use Only - Powered by Explainable Heart Disease Prediction System
    </div>
</body>
</html>
"""
    
    os.makedirs(output_dir, exist_ok=True)
    html_path = os.path.join(output_dir, f"report_{patient_idx}.html")
    pdf_path = os.path.join(output_dir, f"report_{patient_idx}.pdf")
    
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
        
    print(f"HTML report compiled at: {html_path}")
    
    edge_paths = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
        "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
        "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
        "google-chrome",
        "msedge"
    ]
    edge_bin = None
    for p in edge_paths:
        if p == "msedge" or os.path.exists(p):
            edge_bin = p
            break
            
    if edge_bin:
        try:
            abs_html_path = os.path.abspath(html_path)
            abs_pdf_path = os.path.abspath(pdf_path)
            cmd = [
                edge_bin,
                "--headless",
                f"--print-to-pdf={abs_pdf_path}",
                abs_html_path
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            print(f"PDF compiled via Microsoft Edge headless: {pdf_path}")
            return f"report_{patient_idx}.pdf", clinical_details
        except Exception as e:
            print(f"Edge PDF conversion failed: {e}")
    else:
        print("Microsoft Edge not found. PDF report could not be generated.")
        
    return None, clinical_details
