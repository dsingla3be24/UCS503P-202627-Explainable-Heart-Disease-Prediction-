import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from sklearn.model_selection import StratifiedKFold
from src.preprocessing import ClinicalRangeScaler
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from src.models import compile_model

def get_callbacks(checkpoint_path, monitor='val_loss', patience=15, reduce_lr_patience=7):
    """
    Creates standard callbacks:
    1. EarlyStopping: avoids overfitting by halting training when validation performance stalls.
    2. ReduceLROnPlateau: decreases learning rate to fine-tune weights on training plateaus.
    3. ModelCheckpoint: saves the weights of the best performing epoch.
    """
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=reduce_lr_patience,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor=monitor,
            save_best_only=True,
            save_weights_only=True,
            verbose=0
        )
    ]
    return callbacks

def train_and_evaluate_holdout(model_builder, data, epochs=100, batch_size=32, save_dir='results'):
    """
    Trains a model on the hold-out train set, evaluates on validation set,
    and performs final metrics and plots generation on the holdout test set.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Instantiate and compile model
    input_shape = (data['X_train'].shape[1], data['X_train'].shape[2], 1)
    model = model_builder(input_shape)
    model = compile_model(model)
    
    # 2. Setup callbacks
    ckpt_name = f"{model.name}_best_weights.weights.h5"
    ckpt_path = os.path.join(save_dir, ckpt_name)
    callbacks = get_callbacks(ckpt_path, monitor='val_loss', patience=15, reduce_lr_patience=7)
    
    # 3. Calculate class weights (crucial if SMOTE is disabled)
    class_weights = None
    if not data['smote_applied']:
        classes = np.unique(data['y_train'])
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=data['y_train'])
        class_weights = dict(zip(classes, weights))
        print(f"Calculated training class weights: {class_weights}")
        
    # 4. Fit Model
    print(f"\n--- Training {model.name} (Hold-out Pipeline) ---")
    history = model.fit(
        data['X_train'], data['y_train'],
        validation_data=(data['X_val'], data['y_val']),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    # 5. Load the absolute best weights or save final weights if not exists
    if os.path.exists(ckpt_path):
        model.load_weights(ckpt_path)
        print(f"Loaded best weights from {ckpt_path}")
    else:
        model.save_weights(ckpt_path)
        print(f"Saved weights to {ckpt_path}")
        
    # 6. Evaluate on Test set
    y_pred_probs = model.predict(data['X_test']).flatten()
    y_pred_labels = (y_pred_probs >= 0.5).astype(int)
    y_true = data['y_test']
    
    # Compute metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred_labels),
        'precision': precision_score(y_true, y_pred_labels, zero_division=0),
        'recall': recall_score(y_true, y_pred_labels, zero_division=0),
        'f1': f1_score(y_true, y_pred_labels, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_pred_probs)
    }
    
    print(f"\n=== Holdout Evaluation Metrics: {model.name} ===")
    for k, v in metrics.items():
        print(f"  {k.upper()}: {v:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_labels, target_names=['Normal', 'Heart Disease']))
    
    # Save confusion matrix and ROC curves
    plot_evaluation_plots(model.name, y_true, y_pred_probs, y_pred_labels, save_dir)
    
    return model, history, metrics

def run_cross_validation(model_builder, X_df, y_series, numerical_cols, categorical_cols, n_splits=5, epochs=100, batch_size=32, save_dir='results'):
    """
    Performs Stratified K-Fold Cross Validation.
    Applies scaling, outlier capping, encoding, and SMOTE *inside* each fold 
    to completely prevent any data leakage.
    """
    os.makedirs(save_dir, exist_ok=True)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    cv_accuracies = []
    cv_precisions = []
    cv_recalls = []
    cv_f1s = []
    cv_aucs = []
    
    print(f"\n--- Running Stratified {n_splits}-Fold Cross Validation for: {model_builder.__name__} ---")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_df, y_series)):
        print(f"\n--- Fold {fold + 1}/{n_splits} ---")
        
        # 1. Split training and validation folds
        X_tr_raw, y_tr = X_df.iloc[train_idx].copy(), y_series.iloc[train_idx].values
        X_va_raw, y_va = X_df.iloc[val_idx].copy(), y_series.iloc[val_idx].values
        
        # 2. Fit scaler on training fold only, transform both folds
        scaler = ClinicalRangeScaler()
        X_tr_num_scaled = scaler.fit_transform(X_tr_raw[numerical_cols])
        X_va_num_scaled = scaler.transform(X_va_raw[numerical_cols])
        
        df_tr_num = pd.DataFrame(X_tr_num_scaled, columns=numerical_cols, index=X_tr_raw.index)
        df_va_num = pd.DataFrame(X_va_num_scaled, columns=numerical_cols, index=X_va_raw.index)
        
        # 3. One-hot encode categorical features (using pd.get_dummies)
        # Ensure we align columns in case one fold is missing a categorical level
        df_tr_cat = pd.get_dummies(X_tr_raw[categorical_cols].astype(str), prefix=categorical_cols, dtype=float)
        df_va_cat = pd.get_dummies(X_va_raw[categorical_cols].astype(str), prefix=categorical_cols, dtype=float)
        
        # Align column indexes
        df_tr_cat, df_va_cat = df_tr_cat.align(df_va_cat, join='left', axis=1, fill_value=0.0)
        df_va_cat = df_va_cat[df_tr_cat.columns] # Keep identical columns
        
        # Recombine
        X_tr_proc = pd.concat([df_tr_num, df_tr_cat], axis=1).values
        X_va_proc = pd.concat([df_va_num, df_va_cat], axis=1).values
        
        # 4. Apply SMOTE to the training fold
        smote_applied = False
        try:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=42)
            X_tr_res, y_tr_res = smote.fit_resample(X_tr_proc, y_tr)
            smote_applied = True
        except ImportError:
            X_tr_res, y_tr_res = X_tr_proc, y_tr
            
        # 5. Convert to 2D GAF (GASF) images: (samples, features, features, 1)
        from src.preprocessing import batch_compute_gasf
        X_tr_reshaped = batch_compute_gasf(X_tr_res)
        X_va_reshaped = batch_compute_gasf(X_va_proc)
        
        # 6. Build and compile model
        input_shape = (X_tr_reshaped.shape[1], X_tr_reshaped.shape[2], 1)
        model = model_builder(input_shape)
        model = compile_model(model)
        
        ckpt_path = os.path.join(save_dir, f"temp_cv_{model.name}_fold_{fold}.weights.h5")
        callbacks = get_callbacks(ckpt_path, monitor='val_loss', patience=10, reduce_lr_patience=5)
        
        class_weights = None
        if not smote_applied:
            classes = np.unique(y_tr_res)
            weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_tr_res)
            class_weights = dict(zip(classes, weights))
            
        # 7. Train Model
        model.fit(
            X_tr_reshaped, y_tr_res,
            validation_data=(X_va_reshaped, y_va),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=0
        )
        
        # 8. Load best weights and clean temporary checkpoint
        if os.path.exists(ckpt_path):
            model.load_weights(ckpt_path)
            try:
                os.remove(ckpt_path)
            except OSError:
                pass
                
        # 9. Predict and evaluate
        y_pred_probs = model.predict(X_va_reshaped).flatten()
        y_pred_labels = (y_pred_probs >= 0.5).astype(int)
        
        cv_accuracies.append(accuracy_score(y_va, y_pred_labels))
        cv_precisions.append(precision_score(y_va, y_pred_labels, zero_division=0))
        cv_recalls.append(recall_score(y_va, y_pred_labels, zero_division=0))
        cv_f1s.append(f1_score(y_va, y_pred_labels, zero_division=0))
        cv_aucs.append(roc_auc_score(y_va, y_pred_probs))
        
    metrics = {
        'cv_accuracy': np.mean(cv_accuracies),
        'cv_precision': np.mean(cv_precisions),
        'cv_recall': np.mean(cv_recalls),
        'cv_f1': np.mean(cv_f1s),
        'cv_roc_auc': np.mean(cv_aucs)
    }
    
    print(f"\n=== Average CV Metrics: {model_builder.__name__} ===")
    for k, v in metrics.items():
        print(f"  {k.upper()}: {v:.4f}")
        
    return metrics

def plot_evaluation_plots(model_name, y_true, y_pred_probs, y_pred_labels, save_dir):
    """
    Plots and saves:
    1. Confusion Matrix Heatmap
    2. ROC Curve
    """
    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Normal', 'Heart Disease'],
                yticklabels=['Normal', 'Heart Disease'],
                annot_kws={'size': 14, 'weight': 'bold'})
    plt.title(f'Confusion Matrix: {model_name}', fontsize=13, fontweight='bold', pad=15)
    plt.ylabel('True Diagnosis', fontsize=11)
    plt.xlabel('Predicted Diagnosis', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{model_name}_confusion_matrix.png"), dpi=150)
    plt.close()
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    auc_score = roc_auc_score(y_true, y_pred_probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=11)
    plt.ylabel('True Positive Rate', fontsize=11)
    plt.title(f'ROC Curve: {model_name}', fontsize=13, fontweight='bold', pad=15)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{model_name}_roc_curve.png"), dpi=150)
    plt.close()
