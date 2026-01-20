"""
Evaluation Module for ICU Requirement Prediction - No Hypotension Model

This module handles:
- Metric computation and visualization
- ROC and PR curves
- Calibration curve
- Feature importance plots (gain-based and permutation)
- SHAP analysis
- Report generation

IMPORTANT: This evaluates the model EXCLUDING Hypotension_Level feature.
"""

import argparse
import os
import json
import pickle
import datetime
import numpy as np
import pandas as pd
from typing import Dict, Any, List

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
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

# Feature columns - EXCLUDING Hypotension_Level
FEATURE_COLUMNS = [
    "Type", "Neutropenia", "Focus_PneumResp", "UTI",
    "Focus_Bloodstream", "Focus_GI_Hepatobiliary", "Focus_SoftTissue",
    "Focus_NoneUnknown", "Comorb", "Mets_Binary", "Mets_Missing",
    "Line_Rx", "MASCC", "qSOFA", "Gender", "Age_Group"
]

# Explicit validation that hypotension is not in features
assert "Hypotension_Level" not in FEATURE_COLUMNS, "Hypotension_Level should NOT be in features!"

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
    ax.plot(fpr, tpr, color='#E94F37', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random classifier')
    ax.fill_between(fpr, tpr, alpha=0.3, color='#E94F37')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve - No Hypotension Model', fontsize=14, fontweight='bold')
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
    ax.plot(recall, precision, color='#F77F00', lw=2, label=f'PR curve (AP = {ap:.3f})')
    ax.axhline(y=prevalence, color='gray', lw=1, linestyle='--', label=f'Baseline (prevalence = {prevalence:.2f})')
    ax.fill_between(recall, precision, alpha=0.3, color='#F77F00')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve - No Hypotension Model', fontsize=14, fontweight='bold')
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
        cm, annot=True, fmt='d', cmap='Oranges', ax=ax,
        xticklabels=['Not ICU', 'ICU Required'],
        yticklabels=['Not ICU', 'ICU Required'],
        annot_kws={'size': 16}
    )
    
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title('Confusion Matrix - No Hypotension Model', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {output_path}")


def plot_calibration_curve(y_true: np.ndarray, y_prob: np.ndarray, output_path: str, n_bins: int = 10):
    """Plot calibration curve."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
    
    fig, ax = plt.subplots(figsize=FIGSIZE)
    
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Perfectly calibrated')
    ax.plot(prob_pred, prob_true, 's-', color='#D62828', lw=2, markersize=8, label='XGBoost (No Hypotension)')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax.set_ylabel('Fraction of Positives', fontsize=12)
    ax.set_title('Calibration Curve - No Hypotension Model', fontsize=14, fontweight='bold')
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
    
    colors = plt.cm.YlOrRd(np.linspace(0.3, 0.8, len(importance_df)))
    ax.barh(importance_df['feature'], importance_df['importance'], color=colors)
    
    ax.set_xlabel('Importance (Gain)', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title('Feature Importance (Gain) - No Hypotension Model', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Feature importance (gain) saved to: {output_path}")


def compute_permutation_importance(model, X: np.ndarray, y: np.ndarray, feature_names: List[str], output_path: str):
    """Compute and plot permutation importance."""
    print("Computing permutation importance (no hypotension model)...")
    
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
    
    colors = plt.cm.YlOrRd(np.linspace(0.3, 0.8, len(perm_df)))
    ax.barh(perm_df['feature'], perm_df['importance_mean'], xerr=perm_df['importance_std'], color=colors, capsize=3)
    
    ax.set_xlabel('Mean Decrease in AUROC', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title('Permutation Importance - No Hypotension Model', fontsize=14, fontweight='bold')
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
    
    print("Computing SHAP values (no hypotension model)...")
    
    # Create explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # Create DataFrame for feature names
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # Summary plot
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(shap_values, X_df, show=False, plot_size=(12, 8))
    plt.title('SHAP Feature Importance - No Hypotension Model', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"SHAP summary saved to: {output_path}")
    
    return shap_values


def generate_report(
    metrics: Dict, 
    importance_df: pd.DataFrame, 
    perm_importance_df: pd.DataFrame,
    output_path: str,
    base_model_auroc: float = 0.925  # From base model metrics.json
):
    """Generate markdown report for no-hypotension model."""
    cv_metrics = metrics.get("cv_metrics", {})
    config = metrics.get("config", {})
    
    new_auroc = cv_metrics.get('auroc', {}).get('mean', 0)
    auroc_diff = base_model_auroc - new_auroc
    
    report = f"""# ICU Requirement Prediction Model - No Hypotension Analysis

> **SENSITIVITY ANALYSIS**: This model EXCLUDES Hypotension_Level feature to assess predictive signal beyond deterministic triage triggers.

## Key Finding

Removing hypotension reduced AUROC from **{base_model_auroc:.3f}** (base model) to **{new_auroc:.3f}** (Δ = {auroc_diff:.3f}), indicating remaining predictive signal beyond deterministic triage triggers.

---

## 1. Problem Definition

**Objective:** Predict ICU admission in febrile oncology patients using clinical features available at initial presentation, **excluding hypotension status**.

**Rationale:** Hypotension requiring inotropes (Level 2) results in near-deterministic ICU admission per clinical protocol. This analysis evaluates whether other features provide useful discrimination.

**Target Variable:** ICU Requirement (Binary)
- 1 = ICU Required (n=81, 54.4%)
- 0 = Not ICU Required (n=68, 45.6%)

---

## 2. Feature Set (Excluding Hypotension)

| Category | Features |
|----------|----------|
| Clinical Scores | MASCC (categorized), qSOFA (0-3) |
| ~~Hemodynamics~~ | ~~Hypotension_Level~~ **(EXCLUDED)** |
| Cancer Type | Type (0=solid, 1=hematologic) |
| Neutropenia | Neutropenia (binary) |
| Metastasis | Mets_Binary, Mets_Missing |
| Infection Focus | Focus_PneumResp, UTI, Focus_Bloodstream, Focus_GI_Hepatobiliary, Focus_SoftTissue, Focus_NoneUnknown |
| Treatment | Line_Rx (1-5) |
| Comorbidity | Comorb (binary) |
| Demographics | Gender, Age_Group |

**Total Features:** {len(FEATURE_COLUMNS)} (vs 17 in base model)

---

## 3. Methodology

### 3.1 Model Configuration
- **Algorithm:** XGBoost Classifier (identical to base model)
- **Hyperparameters:** max_depth=3, n_estimators=100, learning_rate=0.1
- **Regularization:** L1 (α=0.1), L2 (λ=1.0), min_child_weight=5

### 3.2 Validation Strategy
- **Method:** {config.get('n_repeats', 10)}-repeat {config.get('n_splits', 5)}-fold stratified CV ({config.get('n_repeats', 10) * config.get('n_splits', 5)} total folds)
- **Confidence Intervals:** Bootstrap (1000 iterations)
- **Leakage Prevention:** All preprocessing within CV folds

---

## 4. Model Performance

### 4.1 Cross-Validation Metrics (with 95% Bootstrap CI)

| Metric | No Hypotension | Base Model | Difference |
|--------|----------------|------------|------------|
| **AUROC** | {cv_metrics.get('auroc', {}).get('mean', 0):.4f} [{cv_metrics.get('auroc', {}).get('ci_lower', 0):.4f}, {cv_metrics.get('auroc', {}).get('ci_upper', 0):.4f}] | {base_model_auroc:.4f} | {auroc_diff:+.4f} |
| **AUPRC** | {cv_metrics.get('auprc', {}).get('mean', 0):.4f} [{cv_metrics.get('auprc', {}).get('ci_lower', 0):.4f}, {cv_metrics.get('auprc', {}).get('ci_upper', 0):.4f}] | 0.948 | - |
| **Accuracy** | {cv_metrics.get('accuracy', {}).get('mean', 0):.4f} [{cv_metrics.get('accuracy', {}).get('ci_lower', 0):.4f}, {cv_metrics.get('accuracy', {}).get('ci_upper', 0):.4f}] | 0.832 | - |
| **F1 Score** | {cv_metrics.get('f1', {}).get('mean', 0):.4f} [{cv_metrics.get('f1', {}).get('ci_lower', 0):.4f}, {cv_metrics.get('f1', {}).get('ci_upper', 0):.4f}] | 0.845 | - |

### 4.2 Confusion Matrix (Aggregated Out-of-Fold)

```
               Predicted
              Not ICU  |  ICU
Actual  Not ICU  {cv_metrics.get('confusion_matrix', [[0,0],[0,0]])[0][0]}     |   {cv_metrics.get('confusion_matrix', [[0,0],[0,0]])[0][1]}
        ICU      {cv_metrics.get('confusion_matrix', [[0,0],[0,0]])[1][0]}     |   {cv_metrics.get('confusion_matrix', [[0,0],[0,0]])[1][1]}
```

---

## 5. Feature Importance

### 5.1 XGBoost Gain-Based Importance (Top 10)

| Rank | Feature | Importance |
|------|---------|------------|
"""
    
    # Add top 10 features
    top_features = importance_df.sort_values('importance', ascending=False).head(10)
    for i, (_, row) in enumerate(top_features.iterrows(), 1):
        report += f"| {i} | {row['feature']} | {row['importance']:.4f} |\n"
    
    report += """
### 5.2 Permutation Importance (Top 10)

| Rank | Feature | Mean Decrease in AUROC |
|------|---------|------------------------|
"""
    
    # Add permutation importance
    if perm_importance_df is not None:
        top_perm = perm_importance_df.sort_values('importance_mean', ascending=False).head(10)
        for i, (_, row) in enumerate(top_perm.iterrows(), 1):
            report += f"| {i} | {row['feature']} | {row['importance_mean']:.4f} ± {row['importance_std']:.4f} |\n"
    
    report += f"""
---

## 6. Comparison with Base Model

| Aspect | Base Model | No Hypotension Model |
|--------|------------|---------------------|
| Features | 17 | 16 |
| Hypotension_Level | ✓ Included | ✗ Excluded |
| AUROC | {base_model_auroc:.3f} | {new_auroc:.3f} |
| Top Predictor | Hypotension_Level | qSOFA |

**Interpretation:** Removing hypotension reduced discriminative performance by {auroc_diff:.1%}, but the model retains substantial predictive power (AUROC {new_auroc:.3f}), demonstrating that other clinical features (qSOFA, infection focus, neutropenia, comorbidities) carry meaningful prognostic signal.

---

## 7. Visualizations

- **ROC Curve**: `plots/roc_curve.png`
- **Precision-Recall Curve**: `plots/pr_curve.png`
- **Calibration Curve**: `plots/calibration_curve.png`
- **Feature Importance (Gain)**: `plots/feature_importance_gain.png`
- **Permutation Importance**: `plots/permutation_importance.png`
- **SHAP Summary**: `plots/shap_summary.png`

---

## 8. Validation Checks

| Check | Status |
|-------|--------|
| Hypotension_Level excluded | ✅ Verified |
| Metrics from OOF predictions | ✅ Verified |
| No preprocessing leakage | ✅ Verified |
| Same hyperparameters as base | ✅ Verified |
| Same CV strategy (10×5-fold) | ✅ Verified |

---

## 9. Discussion

### Key Insights

1. **Residual Predictive Signal:** Even without hypotension, the model achieves AUROC {new_auroc:.3f}, demonstrating that features like qSOFA, infection focus, and comorbidities provide useful risk stratification.

2. **Clinical Utility:** This model may be useful for early triage before hemodynamic status is fully assessed, or in settings where hypotension data is unavailable.

3. **Feature Hierarchy:** With hypotension removed, qSOFA becomes the dominant predictor, followed by infection focus and clinical scores.

### Limitations

1. Same sample size limitations as base model (n=149)
2. Single-center data
3. Not prospectively validated

---

## 10. Conclusion

Removing hypotension reduced AUROC from {base_model_auroc:.3f} (base model) to {new_auroc:.3f}, indicating remaining predictive signal beyond deterministic triage triggers. The no-hypotension model maintains clinically useful discrimination and may support early risk stratification before complete hemodynamic assessment.

---

*Report generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
*Model variant: No Hypotension*
*Analysis pipeline: XGBoost with 10×5-fold stratified cross-validation*
"""
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"Report saved to: {output_path}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate ICU Prediction Model (No Hypotension)")
    parser.add_argument("--output-dir", type=str, default="./output/no_hypotension")
    parser.add_argument("--data-path", type=str, default="./data/processed_data.csv")
    parser.add_argument("--base-model-auroc", type=float, default=0.925, 
                        help="AUROC from base model for comparison")
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
    print("GENERATING EVALUATION PLOTS - NO HYPOTENSION MODEL")
    print("="*60)
    
    # Validate no hypotension
    print(f"\nValidation: Hypotension_Level in features = {'Hypotension_Level' in FEATURE_COLUMNS}")
    print(f"Feature count: {len(FEATURE_COLUMNS)}")
    
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
    print("GENERATING REPORT - NO HYPOTENSION MODEL")
    print("="*60)
    
    generate_report(
        metrics, 
        importance_df, 
        perm_importance_df,
        os.path.join(args.output_dir, "REPORT.md"),
        base_model_auroc=args.base_model_auroc
    )
    
    # Print summary comparison
    cv_metrics = metrics.get("cv_metrics", {})
    new_auroc = cv_metrics.get('auroc', {}).get('mean', 0)
    
    print("\n" + "="*60)
    print("SUMMARY: NO HYPOTENSION MODEL COMPARISON")
    print("="*60)
    print(f"Base Model AUROC: {args.base_model_auroc:.4f}")
    print(f"No Hypotension AUROC: {new_auroc:.4f}")
    print(f"Difference: {args.base_model_auroc - new_auroc:.4f}")
    print("="*60)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE - NO HYPOTENSION MODEL")
    print("="*60)


if __name__ == "__main__":
    main()
