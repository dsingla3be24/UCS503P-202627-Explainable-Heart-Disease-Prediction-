import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def compute_grad_cam_2d(model, sample_input, layer_name='conv2d_final'):
    """
    Computes Grad-CAM map for a target 2D Conv layer on a single sample,
    and returns a 1D averaged feature importance vector.
    """
    try:
        conv_layer = model.get_layer(layer_name)
    except ValueError:
        conv_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)]
        if not conv_layers:
            raise ValueError("No Conv2D layer found in the model.")
        conv_layer = conv_layers[-1]
        
    # Construct a sub-model mapping input to conv layer output and prediction logit
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[conv_layer.output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(sample_input)
        p = tf.clip_by_value(predictions[:, 0], 1e-7, 1.0 - 1e-7)
        logit = tf.math.log(p / (1.0 - p))
        
    # Gradients of prediction logit w.r.t. conv layer output: Shape (1, H, W, C)
    grads = tape.gradient(logit, conv_outputs)
    
    # Global average pooling of gradients: Shape (C,)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weighted sum of feature maps: Shape (H, W)
    conv_outputs_val = conv_outputs[0]
    grad_cam_2d = tf.reduce_sum(conv_outputs_val * pooled_grads, axis=-1)
    
    # ReLU to keep features that positively contribute to heart disease
    grad_cam_2d = tf.maximum(grad_cam_2d, 0.0)
    
    # Average 2D Grad-CAM heatmap to 1D feature importance (T,)
    grad_cam_1d = tf.reduce_mean(grad_cam_2d, axis=0)
    
    # Normalize to [0, 1] range
    # Normalize to [0, 1] range for 1D
    max_val = tf.reduce_max(grad_cam_1d)
    if max_val > 0.0:
        grad_cam_1d = grad_cam_1d / max_val
        
    # Normalize to [0, 1] range for 2D
    max_val_2d = tf.reduce_max(grad_cam_2d)
    if max_val_2d > 0.0:
        grad_cam_2d = grad_cam_2d / max_val_2d
        
    return grad_cam_1d.numpy(), grad_cam_2d.numpy()

def get_attention_weights(model, sample_input):
    """
    Extracts attention weights (alphas) from the custom TabularAttention layer.
    """
    try:
        att_layer = model.get_layer('tabular_attention')
        # The TabularAttention returns (context, alphas). Extract alphas (index 1)
        alphas_tensor = att_layer.output[1]
        
        # Build Keras model to output attention weights directly
        att_model = tf.keras.models.Model(inputs=model.input, outputs=alphas_tensor)
        alphas = att_model(sample_input) # Shape: (1, T, 1)
        return alphas.numpy()[0, :, 0] # Squeeze to (T,)
    except Exception as e:
        print(f"Attention weights extraction failed: {e}")
        return None

def compute_shap_values(model, train_data_1d, sample_input_1d):
    """
    Computes SHAP values for 1D tabular inputs using KernelExplainer.
    We build a wrapper function that converts 1D inputs to 2D GAF images,
    makes predictions, and returns them.
    """
    try:
        import shap
    except ImportError:
        print("shap is not installed. Skipping SHAP computation.")
        return None
        
    def model_predict_1d(X_1d):
        from src.preprocessing import batch_compute_gasf
        X_2d = batch_compute_gasf(X_1d)
        try:
            return model.predict(X_2d, verbose=0)
        except TypeError:
            return model.predict(X_2d)
        
    # Summarize background data to make it faster
    background = shap.kmeans(train_data_1d, 10)
    explainer = shap.KernelExplainer(model_predict_1d, background)
    
    # Calculate SHAP values for the sample
    shap_vals = explainer.shap_values(sample_input_1d, nsamples=100, silent=True)
    
    # Extract the SHAP values
    shap_vals = np.squeeze(np.asarray(shap_vals))
    return shap_vals

def aggregate_one_hot_weights(processed_weights, feature_mapping, original_features):
    """
    Aggregates weights from one-hot encoded columns back to the original 13 features.
    For numerical features, maps directly.
    For categorical features, sums the weights of its child dummy columns.
    Finally, normalizes the aggregated weights to sum to 1.0.
    """
    agg_weights = np.zeros(len(original_features))
    
    for i, orig_feat in enumerate(original_features):
        if orig_feat in feature_mapping:
            indices = feature_mapping[orig_feat]
            # Sum the weights of the one-hot columns corresponding to this feature.
            # float(np.sum(...)) keeps this robust if a weight is an array.
            feat_sum = float(np.sum([np.sum(processed_weights[idx]) for idx in indices]))
            agg_weights[i] = feat_sum
            
    # Normalize to prevent scale distortion
    sum_weights = np.sum(agg_weights)
    if sum_weights > 0:
        agg_weights = agg_weights / sum_weights
        
    return agg_weights

def calculate_xai_agreement_metrics(attention_weights, grad_cam_weights, k=4):
    """
    Computes Spearman Rank Correlation and Jaccard Similarity between 
    Attention and Grad-CAM feature importances.
    
    Args:
        attention_weights: numpy array of shape (N_features,)
        grad_cam_weights: numpy array of shape (N_features,)
        k: number of top features to consider for Jaccard Similarity (default: 4)
        
    Returns:
        dict: containing 'spearman_corr', 'spearman_pvalue', and 'jaccard_similarity'
    """
    if attention_weights is None or grad_cam_weights is None:
        return {
            'spearman_corr': 0.0,
            'spearman_pvalue': 1.0,
            'jaccard_similarity': 0.0
        }
        
    from scipy.stats import spearmanr
    
    # 1. Spearman Rank Correlation
    corr, pval = spearmanr(attention_weights, grad_cam_weights)
    
    # 2. Jaccard Similarity of top-k features
    top_k_att = set(np.argsort(attention_weights)[::-1][:k])
    top_k_gc = set(np.argsort(grad_cam_weights)[::-1][:k])
    
    intersection = len(top_k_att.intersection(top_k_gc))
    union = len(top_k_att.union(top_k_gc))
    jaccard = intersection / union if union > 0 else 0.0
    
    return {
        'spearman_corr': corr if not np.isnan(corr) else 0.0,
        'spearman_pvalue': pval if not np.isnan(pval) else 1.0,
        'jaccard_similarity': jaccard
    }

def plot_xai_comparison(original_features, attention_weights, grad_cam_weights, patient_idx, predicted_prob, true_label, shap_weights=None, save_path=None):
    """
    Plots a high-quality, side-by-side barplot comparison of Attention weights,
    Grad-CAM activations, and SHAP values, mapped back to the 13 original clinical features.
    """
    num_plots = 3 if shap_weights is not None else 2
    fig_width = 20 if num_plots == 3 else 14
    plt.figure(figsize=(fig_width, 6))
    
    # Sort original features by attention weight for cleaner visualization
    if attention_weights is not None:
        sort_idx = np.argsort(attention_weights)[::-1]
    else:
        sort_idx = np.argsort(grad_cam_weights)[::-1]
        
    sorted_features = [original_features[i] for i in sort_idx]
    
    # Plot 1: Attention Weights
    plt.subplot(1, num_plots, 1)
    if attention_weights is not None:
        sorted_att = attention_weights[sort_idx]
        sns.barplot(x=sorted_att, y=sorted_features, palette='plasma', hue=sorted_features, legend=False)
        plt.title('Attention Feature Importance', fontsize=12, fontweight='bold', pad=10)
        plt.xlabel('Importance (Normalized to sum to 1)', fontsize=10)
        plt.grid(axis='x', linestyle='--', alpha=0.5)
    else:
        plt.text(0.5, 0.5, "Attention weights are not available\nfor this architecture.", 
                 ha='center', va='center', fontsize=12, color='gray')
        plt.title('Attention Weights', fontsize=12, fontweight='bold')
        
    # Plot 2: Grad-CAM Feature Activations
    plt.subplot(1, num_plots, 2)
    sorted_gc = grad_cam_weights[sort_idx]
    sns.barplot(x=sorted_gc, y=sorted_features, palette='viridis', hue=sorted_features, legend=False)
    plt.title('Grad-CAM Feature Activations', fontsize=12, fontweight='bold', pad=10)
    plt.xlabel('Activation Strength (Normalized [0-1])', fontsize=10)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    
    # Plot 3: SHAP Feature Importance (if available)
    if shap_weights is not None:
        plt.subplot(1, num_plots, 3)
        sorted_shap = shap_weights[sort_idx]
        sns.barplot(x=sorted_shap, y=sorted_features, palette='rocket', hue=sorted_features, legend=False)
        plt.title('SHAP Feature Importance', fontsize=12, fontweight='bold', pad=10)
        plt.xlabel('SHAP Importance (Normalized absolute values)', fontsize=10)
        plt.grid(axis='x', linestyle='--', alpha=0.5)
    
    # Supertitle with patient diagnostic metadata
    diag_status = "Heart Disease" if true_label == 1 else "Normal"
    pred_status = "Heart Disease" if predicted_prob >= 0.5 else "Normal"
    plt.suptitle(
        f"Patient #{patient_idx} Explainability Profile\n"
        f"Actual Diagnosis: {diag_status} | Model Prediction: {pred_status} (Prob: {predicted_prob:.4f})",
        fontsize=14, fontweight='bold', color='#1a1a1a', y=1.02
    )
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()

def plot_importance_heatmap(original_features, weights_matrix, title, save_path=None):
    """
    Plots a feature importance heatmap across patient samples,
    using original feature labels.
    """
    plt.figure(figsize=(12, 6))
    sns.heatmap(
        weights_matrix,
        xticklabels=original_features,
        cmap='YlOrRd',
        cbar_kws={'label': 'Aggregated Importance Weight'}
    )
    plt.title(title, fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('Original Clinical Features', fontsize=11)
    plt.ylabel('Patient Sample Index (Test Split)', fontsize=11)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()

def plot_gaf_sample(gaf_matrix, feature_names, patient_idx, save_path=None):
    """
    Plots a beautiful 2D heatmap of a single patient's Gramian Angular Field (GAF).
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        gaf_matrix,
        xticklabels=feature_names,
        yticklabels=feature_names,
        cmap='rainbow', # vibrant, beautiful aesthetic
        cbar_kws={'label': 'Trigonometric Summation: cos(phi_i + phi_j)'}
    )
    plt.title(f'Gramian Angular Field (GASF) 2D Representation: Patient #{patient_idx}', fontsize=13, fontweight='bold', pad=15)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"GAF 2D plot saved for Patient #{patient_idx} at: {save_path}")
    else:
        plt.show()

import os
import json

def load_guidelines():
    """
    Loads the clinical guidelines schema from the external JSON file.
    """
    path = 'clinical_guidelines.json'
    if not os.path.exists(path):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(current_dir, '..', 'clinical_guidelines.json')
        
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load clinical_guidelines.json. Error: {e}")
        return {}

def generate_clinical_report(raw_patient_row, importance_weights, original_features, predicted_prob, shap_weights=None):
    """
    Generates a doctor-friendly clinical explanation report by dynamically evaluating
    patient values against the clinical guidelines configuration schema loaded from JSON.
    Splits findings into primary model drivers (based on consensus) and other detected abnormalities.
    """
    guidelines = load_guidelines()
    report = []
    diag = "HEART DISEASE PREDICTED" if predicted_prob >= 0.5 else "NORMAL / LOW RISK"
    report.append(f"=== CLINICAL EXPLANATION REPORT FOR DOCTOR ===")
    report.append(f"Model Diagnosis: {diag} (Probability: {predicted_prob:.2%})\n")
    
    # Pre-evaluate ALL features so we know which ones are abnormal
    all_evaluated = {}
    
    for feat_name in original_features:
        val = raw_patient_row[feat_name]
        guideline = guidelines.get(feat_name)
        
        ideal = "N/A"
        reasoning = f"Value is {val}."
        is_abnormal = False
        
        if guideline:
            # Build evaluation context
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
            context = {k: (v.item() if hasattr(v, 'item') else v) for k, v in context.items()}
            
            # Format ideal range
            ideal_tmpl = guideline.get('ideal_range_label', 'N/A')
            try:
                ideal = ideal_tmpl.format(**context)
            except Exception:
                ideal = ideal_tmpl
                
            # Find matching rule
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

    # 1. Primary Model Drivers (top 4 based on importance weight)
    if shap_weights is not None:
        report.append(f"Primary Model Drivers (AI Consensus: Grad-CAM + SHAP):")
        active_weights = (importance_weights + shap_weights) / 2.0
    else:
        report.append(f"Primary Model Drivers (AI Attention Focus):")
        active_weights = importance_weights
        
    sorted_idx = np.argsort(active_weights)[::-1]
    top_4_features = []
    
    for rank, idx in enumerate(sorted_idx[:4], 1):
        feat_name = original_features[idx]
        top_4_features.append(feat_name)
        eval_data = all_evaluated[feat_name]
        
        report.append(f"{rank}. Feature: {feat_name.upper()}")
        report.append(f"   - Patient Value: {eval_data['value']} | Ideal Clinical Range: {eval_data['ideal']}")
        report.append(f"   - Clinical Interpretation: {eval_data['reasoning']}")
        report.append("")
        
    # 2. Secondary Clinical Abnormalities (out-of-bounds features not in the top 4)
    secondary_abnormalities = []
    for feat_name in original_features:
        if feat_name not in top_4_features and all_evaluated[feat_name]['abnormal']:
            secondary_abnormalities.append(feat_name)
            
            
    report.append(f"Secondary Clinical Abnormalities Detected (Outside Top AI Drivers):")
    if secondary_abnormalities:
        for rank, feat_name in enumerate(secondary_abnormalities, 1):
            eval_data = all_evaluated[feat_name]
            report.append(f"{rank}. Feature: {feat_name.upper()}")
            report.append(f"   - Patient Value: {eval_data['value']} | Ideal Clinical Range: {eval_data['ideal']}")
            report.append(f"   - Clinical Interpretation: {eval_data['reasoning']}")
            report.append("")
    else:
        report.append("   None detected.\n")
        
    return "\n".join(report)

def generate_patient_pdf_report(raw_patient_row, importance_weights, original_features, predicted_prob, patient_idx, results_dir, true_label=None, shap_weights=None):
    """
    Generates a beautiful HTML report with the clinical details and explanation,
    embeds the patient's local explainability PNG, and uses Microsoft Edge in 
    headless mode to convert it to a PDF file. Then opens the PDF.
    """
    import subprocess
    import os
    
    guidelines = load_guidelines()
    diag = "HEART DISEASE PREDICTED" if predicted_prob >= 0.5 else "NORMAL / LOW RISK"
    diag_class = "danger" if predicted_prob >= 0.5 else "success"
    
    xai_title = "Explainability Visualization (Attention vs. Grad-CAM vs. SHAP)" if shap_weights is not None else "Explainability Visualization (Attention vs. Grad-CAM)"
    xai_desc = "Figure 1: Tri-Perspective feature attribution showing global attention weight, patient-specific Grad-CAM, and SHAP contributions." if shap_weights is not None else "Figure 1: Dual-Perspective feature attribution showing global attention weight and patient-specific Grad-CAM contribution."
    primary_title = "Primary Diagnostic Drivers (AI Consensus: Grad-CAM + SHAP)" if shap_weights is not None else "Primary Diagnostic Drivers (AI Focus)"
    
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
        primary_html += f"""
        <div class="card driver-card">
            <h4>{rank}. {feat_name.upper()}</h4>
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
            secondary_html += f"""
            <div class="card abnormal-card">
                <h4>{sec_count}. {feat_name.upper()}</h4>
                <p><strong>Patient Value:</strong> {eval_data['value']} &nbsp;|&nbsp; <strong>Ideal Range:</strong> {eval_data['ideal']}</p>
                <p class="interpretation"><strong>Interpretation:</strong> {eval_data['reasoning']}</p>
            </div>
            """
            sec_count += 1
            
    if not secondary_html:
        secondary_html = "<p>No other abnormalities detected.</p>"

    # Map raw variables to readable text labels for the doctor
    sex_val = "Male" if raw_patient_row['sex'] == 0 else "Female"
    cp_val = {1.0: "Typical Angina", 2.0: "Atypical Angina", 3.0: "Non-anginal Pain", 4.0: "Asymptomatic"}.get(raw_patient_row['cp'], str(raw_patient_row['cp']))
    fbs_val = "Elevated (>120 mg/dl)" if raw_patient_row['fbs'] == 1 else "Normal (<=120 mg/dl)"
    restecg_val = {0.0: "Normal", 1.0: "ST-T Wave Abnormality", 2.0: "Left Ventricular Hypertrophy"}.get(raw_patient_row['restecg'], str(raw_patient_row['restecg']))
    exang_val = "Yes" if raw_patient_row['exang'] == 1 else "No"
    slope_val = {1.0: "Upsloping", 2.0: "Flat", 3.0: "Downsloping"}.get(raw_patient_row['slope'], str(raw_patient_row['slope']))
    thal_val = {3.0: "Normal", 6.0: "Fixed Defect", 7.0: "Reversible Defect"}.get(raw_patient_row['thal'], str(raw_patient_row['thal']))
    
    # Absolute path of the explainability image
    img_filename = f"patient_{patient_idx}_explainability.png"
    img_path = os.path.abspath(os.path.join(results_dir, img_filename)).replace("\\", "/")
    
    # Absolute path of the GAF heatmap image
    gaf_img_filename = f"patient_{patient_idx}_gaf.png"
    gaf_img_path = os.path.abspath(os.path.join(results_dir, gaf_img_filename)).replace("\\", "/")
    
    # HTML Content
    html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Clinical Diagnosis Report - Patient #{patient_idx}</title>
<style>
    body {{ font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; color: #333; margin: 40px; line-height: 1.5; }}
    .header {{ border-bottom: 3px solid #0056b3; padding-bottom: 20px; margin-bottom: 20px; }}
    .header h1 {{ margin: 0; color: #0056b3; font-size: 26px; }}
    .header p {{ margin: 5px 0 0 0; color: #666; font-size: 14px; }}
    .meta-table {{ width: 100%; border-collapse: collapse; margin-bottom: 25px; }}
    .meta-table td {{ padding: 8px 10px; border: 1px solid #ddd; font-size: 13px; }}
    .meta-table td.label {{ font-weight: bold; background-color: #f9f9f9; width: 25%; }}
    .diagnosis-box {{ padding: 15px; border-radius: 5px; margin-bottom: 25px; font-size: 18px; font-weight: bold; }}
    .diagnosis-box.danger {{ background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }}
    .diagnosis-box.success {{ background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }}
    .section-title {{ border-bottom: 2px solid #ddd; padding-bottom: 6px; color: #0056b3; font-size: 16px; margin-top: 25px; margin-bottom: 12px; font-weight: bold; }}
    .card {{ padding: 10px 15px; border-radius: 4px; margin-bottom: 12px; border-left: 5px solid; }}
    .driver-card {{ background-color: #e8f4fd; border-left-color: #0056b3; border: 1px solid #d0e8fc; border-left-width: 5px; }}
    .abnormal-card {{ background-color: #fff9e6; border-left-color: #ffc107; border: 1px solid #ffeeba; border-left-width: 5px; }}
    .card h4 {{ margin: 0 0 6px 0; color: #222; font-size: 14px; }}
    .card p {{ margin: 3px 0; font-size: 12px; }}
    .interpretation {{ font-style: italic; color: #444; }}
    .xai-image {{ text-align: center; margin-top: 20px; margin-bottom: 20px; page-break-inside: avoid; }}
    .xai-image img {{ max-width: 90%; height: auto; border: 1px solid #ddd; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }}
    .footer {{ text-align: center; font-size: 11px; color: #999; border-top: 1px solid #eee; padding-top: 15px; margin-top: 40px; }}
</style>
</head>
<body>
    <div class="header">
        <h1>HEART DISEASE PREDICTION REPORT</h1>
        <p>Explainable Deep Learning Diagnosis Platform (CNN-LSTM-Attention + Grad-CAM)</p>
    </div>
    
    <table class="meta-table">
        <tr>
            <td class="label">Patient Index</td>
            <td>#{patient_idx}</td>
            <td class="label">True Label (Ground Truth)</td>
            <td>{"Heart Disease" if true_label == 1 else "Normal"}</td>
        </tr>
        <tr>
            <td class="label">Model Prediction Probability</td>
            <td>{predicted_prob:.2%}</td>
            <td class="label">Analysis Date</td>
            <td>2026-06-23</td>
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

    <div class="section-title">{primary_title}</div>
    {primary_html}
    
    <div class="section-title">Secondary Clinical Abnormalities (Outside Top AI Focus)</div>
    {secondary_html}
    
    <div style="page-break-before: always;"></div>
    
    <div class="section-title">{xai_title}</div>
    <div class="xai-image">
        <img src="{img_path}" alt="Explainability Chart">
        <p style="font-size: 11px; color: #666; margin-top: 6px;">{xai_desc}</p>
    </div>
    
    <div style="page-break-before: always;"></div>
    
    <div class="section-title">2D Gramian Angular Field (GASF) Heatmap</div>
    <div class="xai-image">
        <img src="{gaf_img_path}" alt="GAF Heatmap">
        <p style="font-size: 11px; color: #666; margin-top: 6px;">Figure 2: 2D feature-feature interaction texture mapping derived via Gramian Angular Summation Field conversion.</p>
    </div>
    
    <div class="footer">
        Confidential Medical Report - For Professional Use Only - Powered by Explainable Heart Disease Prediction System
    </div>
</body>
</html>
"""
    
    html_path = os.path.join(results_dir, f"patient_{patient_idx}_report.html")
    pdf_path = os.path.join(results_dir, f"patient_{patient_idx}_report.pdf")
    
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
        
    print(f"HTML report saved at: {html_path}")
    
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
            print(f"PDF report successfully generated via Edge at: {pdf_path}")
            try:
                os.startfile(abs_pdf_path)
            except Exception:
                pass
        except Exception as e:
            print(f"Edge PDF generation failed: {e}")
    else:
        print("Microsoft Edge not found. PDF report could not be generated.")



