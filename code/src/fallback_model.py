import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.preprocessing import preprocess_pipeline, batch_compute_gasf

def aggregate_one_hot_weights(processed_weights, feature_mapping, original_features):
    """
    Aggregates one-hot encoded feature importances back to original features.
    Handles both scalar and array-type weight elements safely.
    """
    processed_weights = np.asarray(processed_weights).flatten()
    agg_weights = np.zeros(len(original_features))
    
    for i, orig_feat in enumerate(original_features):
        if orig_feat in feature_mapping:
            indices = feature_mapping[orig_feat]
            valid_indices = [idx for idx in indices if idx < len(processed_weights)]
            if valid_indices:
                feat_sum = float(np.sum([float(processed_weights[idx]) for idx in valid_indices]))
                agg_weights[i] = feat_sum
            
    # Normalize to prevent scale distortion
    sum_weights = np.sum(agg_weights)
    if sum_weights > 0:
        agg_weights = agg_weights / sum_weights
        
    return agg_weights

class ClinicalModelWrapper:
    """
    Wrapper that manages CNN-LSTM-Attention model training, predictions,
    and explainability (Grad-CAM, Attention, SHAP).
    
    Strictly uses the CNN-LSTM-Attention deep learning architecture.
    No RandomForest or other fallback classifiers are used.
    """
    def __init__(self, dataset_path="heart_disease_cleveland.csv"):
        self.dataset_path = dataset_path
        self.tf_model = None
        
        # Clinical categorizations
        self.numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        self.categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
        
        # Load and preprocess data
        print("Preprocessing dataset for training...")
        self.data_pipeline = preprocess_pipeline(
            self.dataset_path,
            numerical_cols=self.numerical_cols,
            categorical_cols=self.categorical_cols,
            apply_smote=True,
            random_state=42
        )
        self.processed_feature_names = self.data_pipeline['feature_names']
        self.original_features = self.data_pipeline['original_features']
        self.feature_mapping = self.data_pipeline['feature_mapping']
        
        # Initialize the CNN-LSTM-Attention model (uses mock TF emulator on this system)
        import tensorflow as tf
        tf.constant([1.0])  # Validate runtime
        print("TensorFlow runtime validated. Using CNN-LSTM-Attention network.")
        self._initialize_model()

    def _initialize_model(self):
        from src.models import build_cnn_lstm_attention, compile_model
        input_shape = (self.data_pipeline['X_train'].shape[1], self.data_pipeline['X_train'].shape[2], 1)
        self.tf_model = build_cnn_lstm_attention(input_shape)
        self.tf_model = compile_model(self.tf_model)
        
        # Look for best weights
        weights_path = os.path.join("results", "CNN_LSTM_Attention_best_weights.weights.h5")
        if os.path.exists(weights_path) and os.path.getsize(weights_path) > 100:
            print(f"Loading pre-trained CNN weights from {weights_path}...")
            self.tf_model.load_weights(weights_path)
        else:
            print("Pre-trained CNN weights not found/invalid. Training CNN model dynamically...")
            from src.train import train_and_evaluate_holdout
            def model_builder(shape):
                return build_cnn_lstm_attention(shape)
            self.tf_model, _, _ = train_and_evaluate_holdout(
                model_builder, self.data_pipeline, epochs=25, batch_size=16, save_dir="results"
            )
        
        # Calculate test metrics
        X_test_2d = self.data_pipeline['X_test']
        y_test = self.data_pipeline['y_test']
        y_pred_probs = self.tf_model.predict(X_test_2d, verbose=0).flatten()
        y_pred_labels = (y_pred_probs >= 0.5).astype(int)
        self.metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred_labels)),
            'precision': float(precision_score(y_test, y_pred_labels, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred_labels, zero_division=0)),
            'f1': float(f1_score(y_test, y_pred_labels, zero_division=0)),
            'roc_auc': float(roc_auc_score(y_test, y_pred_probs))
        }
            
    def preprocess_single_patient(self, raw_patient_dict):
        """
        Preprocesses a single patient dictionary to match the training set features exactly.
        """
        # Convert dictionary to DataFrame
        df = pd.DataFrame([raw_patient_dict])
        
        # Ensure all columns are numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # Median impute if there's any missing value in the input (using training data defaults)
        df_full_raw = pd.read_csv(self.dataset_path)
        for col in df.columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df_full_raw[col].median())
                
        # Scale numerical columns using ClinicalRangeScaler bounds
        scaler = self.data_pipeline['scaler']
        df_scaled_num = scaler.transform(df[self.numerical_cols])
        
        # One-hot encode categorical features
        df_cat = df[self.categorical_cols].astype(str)
        df_encoded_cat = pd.get_dummies(df_cat, columns=self.categorical_cols, prefix=self.categorical_cols, dtype=float)
        
        # Recombine
        X_processed = pd.concat([df_scaled_num, df_encoded_cat], axis=1)
        
        # Align with the columns of the training set (important for one-hot consistency)
        X_aligned = X_processed.reindex(columns=self.processed_feature_names, fill_value=0.0)
        
        return X_aligned.values # Shape (1, 22)

    def predict_proba(self, raw_patient_dict):
        """
        Predicts heart disease probability for a patient using CNN-LSTM-Attention.
        """
        X_proc_1d = self.preprocess_single_patient(raw_patient_dict)
        # Reshape 1D vector to GAF representation (1, 22, 22, 1)
        X_proc_2d = batch_compute_gasf(X_proc_1d)
        prob = self.tf_model.predict(X_proc_2d, verbose=0).flatten()[0]
        return float(prob)

    def explain(self, raw_patient_dict, predicted_prob):
        """
        Computes XAI local metrics (Attention, Grad-CAM, SHAP, and GAF)
        using the CNN-LSTM-Attention model.
        """
        X_proc_1d = self.preprocess_single_patient(raw_patient_dict)
        X_proc_2d = batch_compute_gasf(X_proc_1d)
        
        # Compute Grad-CAM
        from src.explainability import compute_grad_cam_2d, get_attention_weights, compute_shap_values
        
        gc_raw_1d, gc_raw_2d = compute_grad_cam_2d(self.tf_model, X_proc_2d, layer_name='conv2d_final')
        att_raw = get_attention_weights(self.tf_model, X_proc_2d)
        
        # SHAP calculations
        shap_raw = compute_shap_values(self.tf_model, self.data_pipeline['X_train_1d'], X_proc_1d)
        
        # Aggregate to original 13 features using local helper
        gc_agg = aggregate_one_hot_weights(gc_raw_1d, self.feature_mapping, self.original_features)
        att_agg = aggregate_one_hot_weights(att_raw, self.feature_mapping, self.original_features) if att_raw is not None else None
        
        if shap_raw is not None:
            shap_abs = np.abs(shap_raw)
            shap_agg = aggregate_one_hot_weights(shap_abs, self.feature_mapping, self.original_features)
        else:
            shap_agg = None
            
        gaf_matrix = X_proc_2d[0, :, :, 0]
            
        return {
            'shap': shap_agg,
            'attention': att_agg,
            'grad_cam': gc_agg,
            'grad_cam_2d': GAFMatrixToList(gc_raw_2d),
            'gaf': GAFMatrixToList(gaf_matrix)
        }

    def calculate_subgroup_metrics(self, raw_patient_dict):
        """
        Returns global model performance metrics. 
        Dynamic demographic filtering has been disabled per user request.
        """
        return self.metrics

def GAFMatrixToList(matrix):
    """
    Helper to convert numpy float matrix into JSON-serializable list of lists.
    """
    return [[float(val) for val in row] for row in matrix]
