"""
Evaluation Module for ICU Requirement Prediction

This module handles:
- Metric computation and visualization
- ROC and PR curves
- Calibration curve
- Feature importance plots (gain-based and permutation)
- SHAP analysis
- Report generation
"""

import argparse
import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, List

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance

# Optional SHAP import
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. SHAP plots will be skipped.")

# Feature columns
FEATURE_COLUMNS = [
    "Type", "Neutropenia", "Hypotension_Level", "Focus_PneumResp", "UTI",
    "Focus_Bloodstream", "Focus_GI_Hepatobiliary", "Focus_SoftTissue",
    "Focus_NoneUnknown", "Comorb", "Mets_Binary", "Mets_Missing",
    "Line_Rx", "MASCC", "qSOFA", "Gender", "Age_Group"
]

TARGET_COLUMN = "ICU_Requirement"

# Plot styling
plt.style.use('seaborn-v0_8-whitegrid')
FIGSIZE = (10, 8)
DPI = 150


def load_results(output_dir: str) -> Dict[str, Any]:
    """Load training results."""
    # Load metrics
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    # Load predictions
    predictions_path = os.path.join(output_dir, "cv_predictions.csv")
    predictions_df = pd.read_csv(predictions_path)
    
    # Load model
    model_path = os.path.join(output_dir, "xgboost_model.pkl")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load feature importance
    importance_path = os.path.join(output_dir, "feature_importance.csv")
    importance_df = pd.read_csv(importance_path)
    
    return {
        "metrics": metrics,
        "predictions_df": predictions_df,
        "model": model,
        "importance_df": importance_df
    }


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, output_path: str):
    """Plot ROC curve with AUC."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(fpr, tpr, color='#2E86AB', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random classifier')
    ax.fill_between(fpr, tpr, alpha=0.3, color='#2E86AB')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"ROC curve saved to: {output_path}")


def plot_pr_curve(y_true: np.ndarray, y_prob: np.ndarray, output_path: str):
    """Plot Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    
    # Baseline (prevalence)
    prevalence = np.mean(y_true)
    
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(recall, precision, color='#A23B72', lw=2, label=f'PR curve (AP = {ap:.3f})')
    ax.axhline(y=prevalence, color='gray', lw=1, linestyle='--', label=f'Baseline (prevalence = {prevalence:.2f})')
    ax.fill_between(recall, precision, alpha=0.3, color='#A23B72')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"PR curve saved to: {output_path}")


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_path: str):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Custom colormap
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', ax=ax,
        xticklabels=['Not ICU', 'ICU Required'],
        yticklabels=['Not ICU', 'ICU Required'],
        annot_kws={'size': 16}
    )
    
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {output_path}")


def plot_calibration_curve(y_true: np.ndarray, y_prob: np.ndarray, output_path: str, n_bins: int = 10):
    """Plot calibration curve."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
    
    fig, ax = plt.subplots(figsize=FIGSIZE)
    
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Perfectly calibrated')
    ax.plot(prob_pred, prob_true, 's-', color='#E94F37', lw=2, markersize=8, label='XGBoost')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax.set_ylabel('Fraction of Positives', fontsize=12)
    ax.set_title('Calibration Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Calibration curve saved to: {output_path}")


def plot_feature_importance_gain(importance_df: pd.DataFrame, output_path: str):
    """Plot gain-based feature importance."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Sort and plot
    importance_df = importance_df.sort_values('importance', ascending=True)
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(importance_df)))
    ax.barh(importance_df['feature'], importance_df['importance'], color=colors)
    
    ax.set_xlabel('Importance (Gain)', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title('XGBoost Feature Importance (Gain)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Feature importance (gain) saved to: {output_path}")


def compute_permutation_importance(model, X: np.ndarray, y: np.ndarray, feature_names: List[str], output_path: str):
    """Compute and plot permutation importance."""
    print("Computing permutation importance...")
    
    result = permutation_importance(
        model, X, y, 
        n_repeats=30, 
        random_state=42, 
        scoring='roc_auc',
        n_jobs=-1
    )
    
    # Create DataFrame
    perm_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std
    }).sort_values('importance_mean', ascending=True)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(perm_df)))
    ax.barh(perm_df['feature'], perm_df['importance_mean'], xerr=perm_df['importance_std'], color=colors, capsize=3)
    
    ax.set_xlabel('Mean Decrease in AUROC', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title('Permutation Feature Importance', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='gray', linestyle='--', lw=1)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Permutation importance saved to: {output_path}")
    
    return perm_df


def plot_shap_summary(model, X: np.ndarray, feature_names: List[str], output_path: str):
    """Generate SHAP summary plot."""
    if not SHAP_AVAILABLE:
        print("SHAP not available, skipping SHAP analysis.")
        return None
    
    print("Computing SHAP values...")
    
    # Create explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # Create DataFrame for feature names
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # Summary plot
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(shap_values, X_df, show=False, plot_size=(12, 8))
    plt.title('SHAP Feature Importance (Global)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"SHAP summary saved to: {output_path}")
    
    return shap_values


def generate_report(
    metrics: Dict, 
    importance_df: pd.DataFrame, 
    perm_importance_df: pd.DataFrame,
    output_path: str
):
    """Generate markdown report."""
    cv_metrics = metrics.get("cv_metrics", {})
    config = metrics.get("config", {})
    
    report = f"""# ICU Requirement Prediction Model - Evaluation Report

## Problem Definition

This model predicts whether oncology patients presenting with febrile illness require ICU admission based on clinical features available at initial presentation.

**Target Variable**: ICU Requirement (Binary)
- 1 = ICU Required
- 0 = Not ICU Required

## Cohort Description

- **Sample Size**: ~149 patients
- **Features**: 17 clinical variables
- **Data Source**: S3 bucket (icu-required/cleaned_data.csv)

### Input Features

| Category | Features |
|----------|----------|
| Clinical Scores | MASCC, qSOFA |
| Hemodynamics | Hypotension_Level |
| Cancer Type | Type (hematologic vs solid) |
| Neutropenia | Neutropenia (binary) |
| Metastasis | Mets_Binary, Mets_Missing |
| Infection Focus | Focus_PneumResp, UTI, Focus_Bloodstream, Focus_GI_Hepatobiliary, Focus_SoftTissue, Focus_NoneUnknown |
| Treatment | Line_Rx |
| Comorbidity | Comorb |
| Demographics | Gender, Age_Group |

## Preprocessing Steps

1. **Data Ingestion**: Downloaded from S3 using boto3
2. **Missing Values**: No missing values detected in the dataset
3. **Encoding**: Features were pre-encoded as integers (no additional encoding required)
4. **Leakage Prevention**: All preprocessing steps are applied within CV folds (not on full dataset)

## Validation Strategy

- **Method**: Repeated Stratified K-Fold Cross-Validation
- **K-Folds**: {config.get('n_splits', 5)}
- **Repeats**: {config.get('n_repeats', 10)}
- **Total Folds**: {config.get('n_splits', 5) * config.get('n_repeats', 10)}
- **Random Seed**: {config.get('random_seed', 42)}

This approach provides robust estimates of model performance by:
- Repeated splitting reduces variance from fold assignment
- Stratification maintains class balance in each fold
- Out-of-fold predictions prevent data leakage

## Model Configuration

**Algorithm**: XGBoost Classifier

| Parameter | Value |
|-----------|-------|
| n_estimators | {config.get('xgboost_params', {}).get('n_estimators', 100)} |
| max_depth | {config.get('xgboost_params', {}).get('max_depth', 3)} |
| learning_rate | {config.get('xgboost_params', {}).get('learning_rate', 0.1)} |
| subsample | {config.get('xgboost_params', {}).get('subsample', 0.8)} |
| min_child_weight | {config.get('xgboost_params', {}).get('min_child_weight', 5)} |

## Model Performance

### Cross-Validation Metrics (with 95% Bootstrap Confidence Intervals)

| Metric | Mean | 95% CI |
|--------|------|--------|
| **AUROC** | {cv_metrics.get('auroc', {}).get('mean', 0):.4f} | [{cv_metrics.get('auroc', {}).get('ci_lower', 0):.4f}, {cv_metrics.get('auroc', {}).get('ci_upper', 0):.4f}] |
| **AUPRC** | {cv_metrics.get('auprc', {}).get('mean', 0):.4f} | [{cv_metrics.get('auprc', {}).get('ci_lower', 0):.4f}, {cv_metrics.get('auprc', {}).get('ci_upper', 0):.4f}] |
| **Accuracy** | {cv_metrics.get('accuracy', {}).get('mean', 0):.4f} | [{cv_metrics.get('accuracy', {}).get('ci_lower', 0):.4f}, {cv_metrics.get('accuracy', {}).get('ci_upper', 0):.4f}] |
| **F1 Score** | {cv_metrics.get('f1', {}).get('mean', 0):.4f} | [{cv_metrics.get('f1', {}).get('ci_lower', 0):.4f}, {cv_metrics.get('f1', {}).get('ci_upper', 0):.4f}] |

### Confusion Matrix (Aggregated Out-of-Fold)

```
               Predicted
              Not ICU  |  ICU
Actual  Not ICU  {cv_metrics.get('confusion_matrix', [[0,0],[0,0]])[0][0]}     |   {cv_metrics.get('confusion_matrix', [[0,0],[0,0]])[0][1]}
        ICU      {cv_metrics.get('confusion_matrix', [[0,0],[0,0]])[1][0]}     |   {cv_metrics.get('confusion_matrix', [[0,0],[0,0]])[1][1]}
```

## Feature Importance

### XGBoost Gain-Based Importance (Top 10)

| Rank | Feature | Importance |
|------|---------|------------|
"""
    
    # Add top 10 features
    top_features = importance_df.sort_values('importance', ascending=False).head(10)
    for i, (_, row) in enumerate(top_features.iterrows(), 1):
        report += f"| {i} | {row['feature']} | {row['importance']:.4f} |\n"
    
    report += """
### Permutation Importance (Top 10)

| Rank | Feature | Mean Decrease in AUROC |
|------|---------|------------------------|
"""
    
    # Add permutation importance
    if perm_importance_df is not None:
        top_perm = perm_importance_df.sort_values('importance_mean', ascending=False).head(10)
        for i, (_, row) in enumerate(top_perm.iterrows(), 1):
            report += f"| {i} | {row['feature']} | {row['importance_mean']:.4f} ± {row['importance_std']:.4f} |\n"
    
    report += """
## Visualizations

- **ROC Curve**: `plots/roc_curve.png`
- **Precision-Recall Curve**: `plots/pr_curve.png`
- **Calibration Curve**: `plots/calibration_curve.png`
- **Feature Importance (Gain)**: `plots/feature_importance_gain.png`
- **Permutation Importance**: `plots/permutation_importance.png`
- **SHAP Summary**: `plots/shap_summary.png`

## Limitations and Clinical Context

### Limitations

1. **Small Sample Size**: With ~149 patients, the model has limited statistical power and may not generalize to larger populations.

2. **Single-Center Data**: If the data comes from a single institution, the model may not generalize to other clinical settings.

3. **Temporal Validation**: The model has not been validated on prospectively collected data or across different time periods.

4. **Class Imbalance**: While the dataset is relatively balanced, real-world prevalence of ICU requirement may differ.

5. **Feature Engineering**: The features were pre-encoded; the original clinical values and their distributions are not available for inspection.

6. **Missing External Validation**: The model has only been validated using cross-validation on the same dataset.

### Clinical Context

- This model is intended as a **decision support tool**, not a replacement for clinical judgment.
- Predictions should be interpreted alongside clinical assessment and other patient-specific factors.
- The model identifies patterns associated with ICU admission but does not guarantee outcomes.
- **NOT validated for clinical use** - requires prospective validation before implementation.

### Recommendations for Clinical Implementation

1. Prospective validation on independent patient cohort
2. Calibration assessment in target population
3. Integration with existing clinical workflows
4. Continuous monitoring and recalibration
5. Clear communication of model uncertainty to clinicians

---

*Report generated automatically. Last updated: {date}*
"""
    
    import datetime
    report = report.format(date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"Report saved to: {output_path}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate ICU Prediction Model")
    parser.add_argument("--output-dir", type=str, default="./output")
    parser.add_argument("--data-path", type=str, default="./data/processed_data.csv")
    args = parser.parse_args()
    
    # Create plots directory
    plots_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load results
    results = load_results(args.output_dir)
    predictions_df = results["predictions_df"]
    model = results["model"]
    importance_df = results["importance_df"]
    metrics = results["metrics"]
    
    # Extract targets and predictions
    y_true = predictions_df[TARGET_COLUMN].values
    y_prob = predictions_df["oof_probability"].values
    y_pred = predictions_df["oof_prediction"].values
    X = predictions_df[FEATURE_COLUMNS].values
    
    print("\n" + "="*60)
    print("GENERATING EVALUATION PLOTS")
    print("="*60)
    
    # Generate plots
    plot_roc_curve(y_true, y_prob, os.path.join(plots_dir, "roc_curve.png"))
    plot_pr_curve(y_true, y_prob, os.path.join(plots_dir, "pr_curve.png"))
    plot_confusion_matrix(y_true, y_pred, os.path.join(plots_dir, "confusion_matrix.png"))
    plot_calibration_curve(y_true, y_prob, os.path.join(plots_dir, "calibration_curve.png"))
    plot_feature_importance_gain(importance_df, os.path.join(plots_dir, "feature_importance_gain.png"))
    
    # Permutation importance
    perm_importance_df = compute_permutation_importance(
        model, X, y_true, FEATURE_COLUMNS, 
        os.path.join(plots_dir, "permutation_importance.png")
    )
    
    # Save permutation importance
    perm_importance_df.to_csv(os.path.join(args.output_dir, "permutation_importance.csv"), index=False)
    
    # SHAP analysis
    plot_shap_summary(model, X, FEATURE_COLUMNS, os.path.join(plots_dir, "shap_summary.png"))
    
    # Generate report
    print("\n" + "="*60)
    print("GENERATING REPORT")
    print("="*60)
    
    generate_report(
        metrics, 
        importance_df, 
        perm_importance_df,
        os.path.join(args.output_dir, "REPORT.md")
    )
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
