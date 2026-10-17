import os
import sys

# Inject local lib directory if it exists
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'lib')))

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from src.preprocessing import preprocess_pipeline
from src.models import build_cnn, build_cnn_lstm, build_cnn_lstm_attention, compile_model
from src.train import train_and_evaluate_holdout, run_cross_validation
from src.explainability import (
    compute_grad_cam_2d, get_attention_weights, aggregate_one_hot_weights,
    plot_xai_comparison, plot_importance_heatmap, plot_gaf_sample,
    generate_clinical_report, generate_patient_pdf_report
)

def run_eda(file_path, save_dir='results'):
    """
    Performs Exploratory Data Analysis:
    1. Missing values & Duplicates analysis
    2. Target class distribution plot
    3. Correlation matrix heatmap of numerical columns
    4. Numerical feature distribution plots
    """
    print("\n=== Running Exploratory Data Analysis ===")
    df = pd.read_csv(file_path)
    
    # 1. Dataset overview
    print(f"Dataset Shape: {df.shape}")
    print("\nMissing Values Count per Column (original dataset):")
    # Clean string objects first
    eda_df = df.copy()
    for col in eda_df.columns:
        if eda_df[col].dtype == 'object':
            eda_df[col] = eda_df[col].astype(str).str.strip().replace({'?': np.nan, '': np.nan})
            eda_df[col] = pd.to_numeric(eda_df[col], errors='coerce')
    print(eda_df.isnull().sum())
    
    num_duplicates = eda_df.duplicated().sum()
    print(f"\nNumber of duplicate patient rows: {num_duplicates}")
    
    # Map target >= 1 to 1 for EDA target count plot
    eda_df['target'] = eda_df['target'].apply(lambda x: 1 if x >= 1 else 0)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot 1: Target Class Distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(data=eda_df, x='target', hue='target', palette='Set2', legend=False)
    plt.title('Target Class Distribution (0: Normal, 1: Heart Disease)', fontsize=12, fontweight='bold')
    plt.xlabel('Diagnosis Target')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'eda_class_distribution.png'), dpi=150)
    plt.close()
    
    # Plot 2: Correlation Matrix (numerical columns)
    num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    corr = eda_df[num_cols + ['target']].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
    plt.title('Correlation Matrix of Numerical Features', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'eda_correlation_matrix.png'), dpi=150)
    plt.close()
    
    # Plot 3: Numerical Feature Distributions split by Target
    plt.figure(figsize=(14, 8))
    for i, col in enumerate(num_cols):
        plt.subplot(2, 3, i+1)
        sns.histplot(data=eda_df, x=col, hue='target', kde=True, multiple='stack', palette='Set2')
        plt.title(f'{col.capitalize()} Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'eda_numerical_distributions.png'), dpi=150)
    plt.close()
    
    print(f"EDA Diagnostic plots successfully saved to '{save_dir}' directory.")

def run_tuning(data, max_trials=5, save_dir='results'):
    """
    Executes Keras Tuner hyperparameter search for the CNN-LSTM-Attention architecture.
    """
    print("\n=== Running Hyperparameter Tuning via Keras Tuner ===")
    try:
        import keras_tuner as kt
    except ImportError:
        print("keras-tuner is not installed. Skipping tuning and proceeding with default hyperparameters.")
        return None
        
    input_shape = (data['X_train'].shape[1], data['X_train'].shape[2], 1)
    
    # Define hyperparameter search space builder
    def model_builder_tuner(hp):
        model = build_cnn_lstm_attention(input_shape, hp=hp)
        learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-4])
        return compile_model(model, learning_rate=learning_rate)
        
    tuner = kt.RandomSearch(
        model_builder_tuner,
        objective='val_loss',
        max_trials=max_trials,
        input_shape=input_shape, # Optional but safe
        directory=os.path.join(save_dir, 'keras_tuner_dir'),
        project_name='heart_disease_optimization',
        overwrite=True
    )
    
    # Stop early if search stalls
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)]
    
    tuner.search(
        data['X_train'], data['y_train'],
        epochs=30,
        validation_data=(data['X_val'], data['y_val']),
        callbacks=callbacks,
        verbose=1
    )
    
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("\nBest Hyperparameters Found:")
    print(f"  CNN Filters: {best_hps.get('cnn_filters')}")
    print(f"  CNN Kernel Size: {best_hps.get('cnn_kernel')}")
    print(f"  LSTM Units: {best_hps.get('lstm_units')}")
    print(f"  Dropout Rate: {best_hps.get('dropout'):.2f}")
    print(f"  Dense Units: {best_hps.get('dense_units')}")
    print(f"  Learning Rate: {best_hps.get('learning_rate')}")
    
    return best_hps

def main():
    parser = argparse.ArgumentParser(description="Explainable Heart Disease Prediction Pipeline")
    parser.add_argument('--dataset', type=str, default='heart_disease_cleveland.csv', help='Filename of the CSV dataset')
    parser.add_argument('--tune', action='store_true', help='Run Keras Tuner hyperparameter search')
    parser.add_argument('--cv', action='store_true', help='Run 5-Fold Cross Validation')
    parser.add_argument('--epochs', type=int, default=80, help='Max number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
    parser.add_argument('--results_dir', type=str, default='results', help='Directory to save output files')
    parser.add_argument('--patient_idx', type=int, default=None, help='Index of the patient in the test set to explain separately')
    args = parser.parse_args()
    
    # Root base directory (dynamically resolved)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, args.dataset)
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return
        
    # Re-seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Clinical categorizations
    numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    
    # 2. Run Preprocessing Pipeline
    data = preprocess_pipeline(
        dataset_path,
        numerical_cols=numerical_cols,
        categorical_cols=categorical_cols,
        apply_smote=True,
        random_state=42
    )
    
    from src.preprocessing import load_and_clean_data
    df_clean = load_and_clean_data(dataset_path)
    original_features = data['original_features']
    feature_mapping = data['feature_mapping']

    # --- Mode 1: Run Separate Patient Explanation ---
    if args.patient_idx is not None:
        idx = args.patient_idx
        if idx < 0 or idx >= len(data['X_test']):
            print(f"Error: Patient index must be between 0 and {len(data['X_test']) - 1}.")
            return
            
        print(f"\n=== Running Separate Patient Explanation for Patient Index #{idx} ===")
        
        # Load or train the CNN-LSTM-Attention model
        input_shape = (data['X_train'].shape[1], data['X_train'].shape[2], 1)
        att_model = build_cnn_lstm_attention(input_shape)
        att_model = compile_model(att_model)
        weights_path = os.path.join(args.results_dir, "CNN_LSTM_Attention_best_weights.weights.h5")
        
        if os.path.exists(weights_path):
            print(f"Loading pre-trained weights from {weights_path}...")
            att_model.load_weights(weights_path)
        else:
            print("Pre-trained weights not found. Training CNN-LSTM-Attention model first to obtain weights...")
            def cnn_attention_builder(shape):
                return build_cnn_lstm_attention(shape)
            att_model, _, _ = train_and_evaluate_holdout(
                cnn_attention_builder, data, epochs=args.epochs, batch_size=args.batch_size, save_dir=args.results_dir
            )
            
        sample = data['X_test'][idx:idx+1]
        y_test_probs = att_model.predict(sample).flatten()
        prob = y_test_probs[0]
        true_lbl = data['y_test'][idx]
        
        # Calculate local explainability weights
        gc_weights_raw, _ = compute_grad_cam_2d(att_model, sample, layer_name='conv2d_final')
        att_weights_raw = get_attention_weights(att_model, sample)
        
        # Calculate SHAP values
        from src.explainability import compute_shap_values
        sample_1d = data['X_test_1d'][idx:idx+1]
        shap_weights_raw = compute_shap_values(att_model, data['X_train_1d'], sample_1d)
        
        gc_weights_agg = aggregate_one_hot_weights(gc_weights_raw, feature_mapping, original_features)
        att_weights_agg = None
        if att_weights_raw is not None:
            att_weights_agg = aggregate_one_hot_weights(att_weights_raw, feature_mapping, original_features)
            
        shap_weights_agg = None
        if shap_weights_raw is not None:
            shap_weights_abs = np.abs(shap_weights_raw)
            shap_weights_agg = aggregate_one_hot_weights(shap_weights_abs, feature_mapping, original_features)
            
        # Plot and save comparison
        # Setup separate subdirectory for this specific patient
        patient_dir = os.path.join(args.results_dir, f'patient_{idx}')
        os.makedirs(patient_dir, exist_ok=True)
        
        # Plot and save comparison
        save_path = os.path.join(patient_dir, f'patient_{idx}_explainability.png')
        plot_xai_comparison(
            original_features=original_features,
            attention_weights=att_weights_agg,
            grad_cam_weights=gc_weights_agg,
            patient_idx=idx,
            predicted_prob=prob,
            true_label=true_lbl,
            shap_weights=shap_weights_agg,
            save_path=save_path
        )
        print(f"Local explainability profile saved for patient #{idx} at: {save_path}")
        
        # Generate individual 2D GAF Heatmap for this patient
        gaf_path = os.path.join(patient_dir, f'patient_{idx}_gaf.png')
        plot_gaf_sample(sample[0].squeeze(), data['feature_names'], patient_idx=idx, save_path=gaf_path)
        
        # Generate doctor-friendly clinical report
        orig_idx = data['X_test_indices'][idx]
        raw_patient_row = df_clean.loc[orig_idx]
        clinical_rep = generate_clinical_report(raw_patient_row, gc_weights_agg, original_features, prob, shap_weights=shap_weights_agg)
        
        # Save clinical report as text file
        rep_path = os.path.join(patient_dir, f'patient_{idx}_clinical_explanation.txt')
        with open(rep_path, 'w', encoding='utf-8') as f:
            f.write(clinical_rep)
        print(f"Doctor-friendly explanation saved for patient #{idx} at: {rep_path}")
        print(clinical_rep)
        
        # Generate and open PDF report
        generate_patient_pdf_report(raw_patient_row, gc_weights_agg, original_features, prob, idx, patient_dir, true_lbl, shap_weights=shap_weights_agg)
        
        print("Separate Patient Explanation Complete.")
        return

    # --- Mode 2: Run Full Pipeline ---
    # 1. Run EDA
    run_eda(dataset_path, args.results_dir)
    
    # 2.5 Generate and Save a GAF (GASF) 2D Image Demonstration
    print("\n=== Generating 2D GAF Image Demonstration ===")
    gaf_demo_path = os.path.join(args.results_dir, 'gaf_representation_patient_0.png')
    patient_0_gasf = data['X_test'][0].squeeze()
    plot_gaf_sample(patient_0_gasf, data['feature_names'], patient_idx=0, save_path=gaf_demo_path)
    
    # 3. Optional Hyperparameter Tuning
    best_hps = None
    if args.tune:
        best_hps = run_tuning(data, max_trials=5, save_dir=args.results_dir)
        
    # Model builders
    def cnn_builder(shape):
        return build_cnn(shape, hp=best_hps)
    def cnn_lstm_builder(shape):
        return build_cnn_lstm(shape, hp=best_hps)
    def cnn_attention_builder(shape):
        return build_cnn_lstm_attention(shape, hp=best_hps)
        
    # 4. Train Holdout Models
    cnn_model, _, cnn_metrics = train_and_evaluate_holdout(
        cnn_builder, data, epochs=args.epochs, batch_size=args.batch_size, save_dir=args.results_dir
    )
    
    lstm_model, _, lstm_metrics = train_and_evaluate_holdout(
        cnn_lstm_builder, data, epochs=args.epochs, batch_size=args.batch_size, save_dir=args.results_dir
    )
    
    att_model, _, att_metrics = train_and_evaluate_holdout(
        cnn_attention_builder, data, epochs=args.epochs, batch_size=args.batch_size, save_dir=args.results_dir
    )
    
    # 5. Holdout Ablation Study comparison
    ablation_df = pd.DataFrame({
        'Model': ['CNN', 'CNN-LSTM', 'CNN-LSTM-Attention'],
        'Accuracy': [cnn_metrics['accuracy'], lstm_metrics['accuracy'], att_metrics['accuracy']],
        'Precision': [cnn_metrics['precision'], lstm_metrics['precision'], att_metrics['precision']],
        'Recall': [cnn_metrics['recall'], lstm_metrics['recall'], att_metrics['recall']],
        'F1-Score': [cnn_metrics['f1'], lstm_metrics['f1'], att_metrics['f1']],
        'ROC-AUC': [cnn_metrics['roc_auc'], lstm_metrics['roc_auc'], att_metrics['roc_auc']]
    })
    
    print("\n=== Holdout Ablation Study Summary ===")
    print(ablation_df.to_string(index=False))
    ablation_df.to_csv(os.path.join(args.results_dir, 'ablation_study_holdout.csv'), index=False)
    
    # Plot Ablation Comparison Barplot
    plt.figure(figsize=(10, 6))
    melted_df = pd.melt(ablation_df, id_vars='Model', var_name='Metric', value_name='Score')
    sns.barplot(data=melted_df, x='Metric', y='Score', hue='Model', palette='muted')
    plt.title('Ablation Study Comparison (Hold-out Test Set)', fontsize=13, fontweight='bold', pad=15)
    plt.ylim([0.0, 1.05])
    plt.ylabel('Score')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, 'ablation_study_comparison.png'), dpi=150)
    plt.close()
    
    # 6. Run Cross Validation (Optional)
    if args.cv:
        # Load raw dataset for CV
        df_full = pd.read_csv(dataset_path)
        feature_cols = [col for col in df_full.columns if col not in ['target', 'source']]
        X_df = df_full[feature_cols].copy()
        
        # Format object columns and clean
        for col in X_df.columns:
            if X_df[col].dtype == 'object':
                X_df[col] = pd.to_numeric(X_df[col].replace({'?': np.nan}), errors='coerce')
        X_df = X_df.fillna(X_df.median())
        y_full = df_full['target'].apply(lambda x: 1 if x >= 1 else 0)
        
        cnn_cv = run_cross_validation(cnn_builder, X_df, y_full, numerical_cols, categorical_cols, n_splits=5, epochs=args.epochs, batch_size=args.batch_size, save_dir=args.results_dir)
        lstm_cv = run_cross_validation(cnn_lstm_builder, X_df, y_full, numerical_cols, categorical_cols, n_splits=5, epochs=args.epochs, batch_size=args.batch_size, save_dir=args.results_dir)
        att_cv = run_cross_validation(cnn_attention_builder, X_df, y_full, numerical_cols, categorical_cols, n_splits=5, epochs=args.epochs, batch_size=args.batch_size, save_dir=args.results_dir)
        
        cv_ablation_df = pd.DataFrame({
            'Model': ['CNN', 'CNN-LSTM', 'CNN-LSTM-Attention'],
            'CV Accuracy': [cnn_cv['cv_accuracy'], lstm_cv['cv_accuracy'], att_cv['cv_accuracy']],
            'CV Precision': [cnn_cv['cv_precision'], lstm_cv['cv_precision'], att_cv['cv_precision']],
            'CV Recall': [cnn_cv['cv_recall'], lstm_cv['cv_recall'], att_cv['cv_recall']],
            'CV F1-Score': [cnn_cv['cv_f1'], lstm_cv['cv_f1'], att_cv['cv_f1']],
            'CV ROC-AUC': [cnn_cv['cv_roc_auc'], lstm_cv['cv_roc_auc'], att_cv['cv_roc_auc']]
        })
        print("\n=== 5-Fold Cross Validation Ablation Summary ===")
        print(cv_ablation_df.to_string(index=False))
        cv_ablation_df.to_csv(os.path.join(args.results_dir, 'ablation_study_cross_validation.csv'), index=False)
        
    # 7. Generate Explainable AI Profiles
    print("\n=== Generating Explainable AI Diagnoses ===")
    
    # Generate test set predictions for selection
    y_test_probs = att_model.predict(data['X_test']).flatten()
    y_test_true = data['y_test']
    
    # Select true positives (heart disease correctly classified) and true negatives (normal correctly classified)
    pos_indices = np.where((y_test_true == 1) & (y_test_probs >= 0.7))[0]
    neg_indices = np.where((y_test_true == 0) & (y_test_probs <= 0.3))[0]
    
    selected_indices = []
    if len(pos_indices) >= 3:
        selected_indices.extend(pos_indices[:3])
    elif len(pos_indices) > 0:
        selected_indices.extend(pos_indices)
        
    if len(neg_indices) >= 3:
        selected_indices.extend(neg_indices[:3])
    elif len(neg_indices) > 0:
        selected_indices.extend(neg_indices)
        
    print(f"Selected patient test indexes for diagnostic explainability profile: {selected_indices}")
    
    for idx in selected_indices:
        sample = data['X_test'][idx:idx+1] # Shape: (1, NumProcessedFeatures, 1)
        prob = y_test_probs[idx]
        true_lbl = y_test_true[idx]
        
        # Calculate local Grad-CAM w.r.t final conv layer
        gc_weights_raw, _ = compute_grad_cam_2d(att_model, sample, layer_name='conv2d_final')
        # Extract attention weights
        att_weights_raw = get_attention_weights(att_model, sample)
        
        # Calculate SHAP values
        from src.explainability import compute_shap_values
        sample_1d = data['X_test_1d'][idx:idx+1]
        shap_weights_raw = compute_shap_values(att_model, data['X_train_1d'], sample_1d)
        
        # Aggregate raw weights back to 13 original clinical features
        gc_weights_agg = aggregate_one_hot_weights(gc_weights_raw, feature_mapping, original_features)
        att_weights_agg = None
        if att_weights_raw is not None:
            att_weights_agg = aggregate_one_hot_weights(att_weights_raw, feature_mapping, original_features)
            
        shap_weights_agg = None
        if shap_weights_raw is not None:
            shap_weights_abs = np.abs(shap_weights_raw)
            shap_weights_agg = aggregate_one_hot_weights(shap_weights_abs, feature_mapping, original_features)
            
        # Setup separate subdirectory for this specific patient
        patient_dir = os.path.join(args.results_dir, f'patient_{idx}')
        os.makedirs(patient_dir, exist_ok=True)
        
        # Plot and save comparison
        save_path = os.path.join(patient_dir, f'patient_{idx}_explainability.png')
        plot_xai_comparison(
            original_features=original_features,
            attention_weights=att_weights_agg,
            grad_cam_weights=gc_weights_agg,
            patient_idx=idx,
            predicted_prob=prob,
            true_label=true_lbl,
            shap_weights=shap_weights_agg,
            save_path=save_path
        )
        print(f"Local explainability profile saved for patient #{idx} at: {save_path}")
        
        # Generate individual 2D GAF Heatmap for this patient
        gaf_path = os.path.join(patient_dir, f'patient_{idx}_gaf.png')
        plot_gaf_sample(sample[0].squeeze(), data['feature_names'], patient_idx=idx, save_path=gaf_path)
        
        # Generate doctor-friendly clinical report
        orig_idx = data['X_test_indices'][idx]
        raw_patient_row = df_clean.loc[orig_idx]
        clinical_rep = generate_clinical_report(raw_patient_row, gc_weights_agg, original_features, prob, shap_weights=shap_weights_agg)
        
        # Save clinical report as text file
        rep_path = os.path.join(patient_dir, f'patient_{idx}_clinical_explanation.txt')
        with open(rep_path, 'w', encoding='utf-8') as f:
            f.write(clinical_rep)
        print(f"Doctor-friendly explanation saved for patient #{idx} at: {rep_path}")
        print(clinical_rep)
        
        # Generate and open PDF report
        generate_patient_pdf_report(raw_patient_row, gc_weights_agg, original_features, prob, idx, patient_dir, true_lbl, shap_weights=shap_weights_agg)
        
        print("-" * 50)
        
    # Generate global Attention Heatmap for all test samples
    all_att_weights = []
    for i in range(len(data['X_test'])):
        sample = data['X_test'][i:i+1]
        att_w_raw = get_attention_weights(att_model, sample)
        if att_w_raw is not None:
            att_w_agg = aggregate_one_hot_weights(att_w_raw, feature_mapping, original_features)
            all_att_weights.append(att_w_agg)
            
    if all_att_weights:
        all_att_weights = np.array(all_att_weights)
        plot_importance_heatmap(
            original_features=original_features,
            weights_matrix=all_att_weights,
            title='Global Feature Importance Heatmap: Attention Weights (Test Set)',
            save_path=os.path.join(args.results_dir, 'global_attention_heatmap.png')
        )
        print("Global Attention Heatmap successfully saved.")
        
    # Generate global Grad-CAM Heatmap for all test samples
    all_gc_weights = []
    for i in range(len(data['X_test'])):
        sample = data['X_test'][i:i+1]
        gc_w_raw, _ = compute_grad_cam_2d(att_model, sample, layer_name='conv2d_final')
        gc_w_agg = aggregate_one_hot_weights(gc_w_raw, feature_mapping, original_features)
        all_gc_weights.append(gc_w_agg)
        
    if all_gc_weights:
        all_gc_weights = np.array(all_gc_weights)
        plot_importance_heatmap(
            original_features=original_features,
            weights_matrix=all_gc_weights,
            title='Global Feature Importance Heatmap: Grad-CAM Activations (Test Set)',
            save_path=os.path.join(args.results_dir, 'global_grad_cam_heatmap.png')
        )
        print("Global Grad-CAM Heatmap successfully saved.")
        
    print("\nPipeline execution complete! All plots and tables saved in 'results/' directory.")

if __name__ == '__main__':
    main()
