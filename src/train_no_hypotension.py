"""
Training Module for ICU Requirement Prediction - No Hypotension Model

This module handles:
- XGBoost model training with repeated stratified K-fold cross-validation
- Leakage-safe preprocessing (fit only on training folds)
- Model persistence
- Out-of-fold predictions

IMPORTANT: This model EXCLUDES Hypotension_Level feature for sensitivity analysis.
"""

import argparse
import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any

import xgboost as xgb
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score, f1_score,
    confusion_matrix, roc_curve, precision_recall_curve
)

# Configuration
RANDOM_SEED = 42
N_SPLITS = 5
N_REPEATS = 10

# Feature columns - EXCLUDING Hypotension_Level
FEATURE_COLUMNS = [
    "Type", "Neutropenia", "Focus_PneumResp", "UTI",
    "Focus_Bloodstream", "Focus_GI_Hepatobiliary", "Focus_SoftTissue",
    "Focus_NoneUnknown", "Comorb", "Mets_Binary", "Mets_Missing",
    "Line_Rx", "MASCC", "qSOFA", "Gender", "Age_Group"
]

# Explicit validation that hypotension is not in features
assert "Hypotension_Level" not in FEATURE_COLUMNS, "Hypotension_Level should NOT be in features!"
assert "Hypotension" not in " ".join(FEATURE_COLUMNS), "No hypotension-related features allowed!"

TARGET_COLUMN = "ICU_Requirement"

# XGBoost hyperparameters (same as base model for fair comparison)
XGBOOST_PARAMS = {
    "n_estimators": 100,
    "max_depth": 3,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": RANDOM_SEED,
    "use_label_encoder": False,
    "eval_metric": "logloss",
    "n_jobs": -1
}


def load_data(data_path: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load preprocessed data and extract features/target (excluding hypotension)."""
    df = pd.read_csv(data_path)
    
    # Validate hypotension is not used
    print("\n" + "="*60)
    print("NO-HYPOTENSION MODEL - Validation Check")
    print("="*60)
    print(f"Features used ({len(FEATURE_COLUMNS)} total):")
    for f in FEATURE_COLUMNS:
        print(f"  - {f}")
    print(f"\nHypotension_Level in features: {'Hypotension_Level' in FEATURE_COLUMNS}")
    print("="*60 + "\n")
    
    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values
    
    print(f"Loaded {len(df)} samples with {X.shape[1]} features")
    print(f"Target distribution: {np.bincount(y.astype(int))}")
    
    return df, X, y


def bootstrap_ci(values: np.ndarray, n_bootstrap: int = 1000, ci: float = 0.95) -> Tuple[float, float]:
    """Compute bootstrap confidence interval."""
    np.random.seed(RANDOM_SEED)
    bootstrapped = np.array([
        np.mean(np.random.choice(values, size=len(values), replace=True))
        for _ in range(n_bootstrap)
    ])
    lower = np.percentile(bootstrapped, (1 - ci) / 2 * 100)
    upper = np.percentile(bootstrapped, (1 + ci) / 2 * 100)
    return lower, upper


def train_with_cv(
    X: np.ndarray, 
    y: np.ndarray, 
    n_splits: int = N_SPLITS, 
    n_repeats: int = N_REPEATS
) -> Dict[str, Any]:
    """
    Train XGBoost with repeated stratified K-fold cross-validation.
    
    Returns out-of-fold predictions, metrics, and trained models.
    """
    print(f"\nRunning {n_repeats}-repeat {n_splits}-fold stratified cross-validation...")
    print(f"Total folds: {n_splits * n_repeats}")
    print(f"Model: NO-HYPOTENSION (excluding Hypotension_Level)")
    
    # Initialize cross-validator
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_SEED)
    
    # Storage for results
    all_fold_metrics = {
        "auroc": [],
        "auprc": [],
        "accuracy": [],
        "f1": []
    }
    
    # For aggregated OOF predictions (using last repeat for simplicity)
    oof_predictions = np.zeros(len(y))
    oof_probabilities = np.zeros(len(y))
    oof_counts = np.zeros(len(y))
    
    # Store models from the last repeat
    models = []
    
    fold_idx = 0
    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Train XGBoost
        model = xgb.XGBClassifier(**XGBOOST_PARAMS)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Predictions
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Calculate metrics
        auroc = roc_auc_score(y_val, y_pred_proba)
        auprc = average_precision_score(y_val, y_pred_proba)
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        
        all_fold_metrics["auroc"].append(auroc)
        all_fold_metrics["auprc"].append(auprc)
        all_fold_metrics["accuracy"].append(acc)
        all_fold_metrics["f1"].append(f1)
        
        # Accumulate OOF predictions
        oof_probabilities[val_idx] += y_pred_proba
        oof_counts[val_idx] += 1
        
        # Store model from last repeat
        if fold_idx >= (n_repeats - 1) * n_splits:
            models.append(model)
        
        fold_idx += 1
        
        if fold_idx % n_splits == 0:
            repeat_num = fold_idx // n_splits
            avg_auroc = np.mean(all_fold_metrics["auroc"][-n_splits:])
            print(f"  Repeat {repeat_num}/{n_repeats}: Mean AUROC = {avg_auroc:.4f}")
    
    # Average OOF probabilities
    oof_probabilities = oof_probabilities / oof_counts
    oof_predictions = (oof_probabilities >= 0.5).astype(int)
    
    # Compute summary metrics with CIs
    summary_metrics = {}
    for metric_name, values in all_fold_metrics.items():
        values = np.array(values)
        mean_val = np.mean(values)
        std_val = np.std(values)
        lower, upper = bootstrap_ci(values)
        
        summary_metrics[metric_name] = {
            "mean": float(mean_val),
            "std": float(std_val),
            "ci_lower": float(lower),
            "ci_upper": float(upper),
            "all_values": [float(v) for v in values]
        }
    
    # Overall confusion matrix from averaged OOF predictions
    cm = confusion_matrix(y, oof_predictions)
    summary_metrics["confusion_matrix"] = cm.tolist()
    
    print(f"\n--- Cross-Validation Results (NO HYPOTENSION) ---")
    for metric, vals in summary_metrics.items():
        if metric != "confusion_matrix":
            print(f"{metric.upper()}: {vals['mean']:.4f} (95% CI: [{vals['ci_lower']:.4f}, {vals['ci_upper']:.4f}])")
    
    print(f"\nConfusion Matrix (aggregated OOF):")
    print(f"  TN={cm[0,0]}, FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}, TP={cm[1,1]}")
    
    results = {
        "metrics": summary_metrics,
        "oof_predictions": oof_predictions,
        "oof_probabilities": oof_probabilities,
        "models": models,
        "feature_names": FEATURE_COLUMNS,
        "config": {
            "n_splits": n_splits,
            "n_repeats": n_repeats,
            "random_seed": RANDOM_SEED,
            "xgboost_params": XGBOOST_PARAMS,
            "model_variant": "no_hypotension",
            "excluded_features": ["Hypotension_Level"]
        }
    }
    
    return results


def compute_feature_importance(models: List[xgb.XGBClassifier], feature_names: List[str]) -> pd.DataFrame:
    """Compute gain-based feature importance averaged across models."""
    importance_dfs = []
    
    for model in models:
        importance = model.get_booster().get_score(importance_type='gain')
        # Convert feature names (f0, f1, ...) to actual names
        importance_mapped = {}
        for feat, score in importance.items():
            feat_idx = int(feat.replace('f', ''))
            if feat_idx < len(feature_names):
                importance_mapped[feature_names[feat_idx]] = score
        
        importance_dfs.append(pd.Series(importance_mapped))
    
    # Average across models
    importance_df = pd.concat(importance_dfs, axis=1).mean(axis=1).sort_values(ascending=False)
    importance_df = importance_df.reset_index()
    importance_df.columns = ['feature', 'importance']
    
    return importance_df


def train_final_model(X: np.ndarray, y: np.ndarray) -> xgb.XGBClassifier:
    """Train final model on all data for deployment (optional)."""
    print("\nTraining final model on all data (no hypotension)...")
    model = xgb.XGBClassifier(**XGBOOST_PARAMS)
    model.fit(X, y, verbose=False)
    return model


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train XGBoost for ICU Prediction (No Hypotension)")
    parser.add_argument("--data-path", type=str, default="./data/processed_data.csv")
    parser.add_argument("--output-dir", type=str, default="./output/no_hypotension")
    parser.add_argument("--n-splits", type=int, default=N_SPLITS)
    parser.add_argument("--n-repeats", type=int, default=N_REPEATS)
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    df, X, y = load_data(args.data_path)
    
    # Train with cross-validation
    results = train_with_cv(X, y, n_splits=args.n_splits, n_repeats=args.n_repeats)
    
    # Compute feature importance
    importance_df = compute_feature_importance(results["models"], results["feature_names"])
    print("\n--- Feature Importance (Gain) - No Hypotension Model ---")
    print(importance_df.to_string(index=False))
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    metrics_output = {
        "cv_metrics": results["metrics"],
        "config": results["config"]
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics_output, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")
    
    # Save OOF predictions
    oof_df = df.copy()
    oof_df["oof_probability"] = results["oof_probabilities"]
    oof_df["oof_prediction"] = results["oof_predictions"]
    oof_path = os.path.join(args.output_dir, "cv_predictions.csv")
    oof_df.to_csv(oof_path, index=False)
    print(f"OOF predictions saved to: {oof_path}")
    
    # Save feature importance
    importance_path = os.path.join(args.output_dir, "feature_importance.csv")
    importance_df.to_csv(importance_path, index=False)
    print(f"Feature importance saved to: {importance_path}")
    
    # Train and save final model
    final_model = train_final_model(X, y)
    model_path = os.path.join(args.output_dir, "xgboost_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(final_model, f)
    print(f"Final model saved to: {model_path}")
    
    # Save validation summary
    validation_path = os.path.join(args.output_dir, "validation_summary.txt")
    with open(validation_path, 'w') as f:
        f.write("NO-HYPOTENSION MODEL - Validation Summary\n")
        f.write("="*50 + "\n\n")
        f.write("1. Hypotension feature excluded: YES\n")
        f.write(f"   Features used: {len(FEATURE_COLUMNS)}\n")
        f.write(f"   Hypotension_Level in features: {'Hypotension_Level' in FEATURE_COLUMNS}\n\n")
        f.write("2. Metrics from OOF predictions: YES\n")
        f.write("   All metrics computed from out-of-fold predictions\n\n")
        f.write("3. Preprocessing leakage: NONE\n")
        f.write("   No preprocessing applied (features already encoded in data)\n")
        f.write("   Model trained only on training folds\n\n")
        f.write(f"4. Feature list:\n")
        for f_name in FEATURE_COLUMNS:
            f.write(f"   - {f_name}\n")
    print(f"Validation summary saved to: {validation_path}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE - NO HYPOTENSION MODEL")
    print("="*60)
    
    return results


if __name__ == "__main__":
    main()
