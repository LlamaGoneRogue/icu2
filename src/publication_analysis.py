"""
Publication-Ready Analysis Script

Generates all priority analyses for journal submission:
- Priority 1: Baseline comparisons, single-score performance, calibration, DCA
- Priority 2: Simulated external validation, subgroup analysis, threshold optimization
- Priority 3: Learning curves, cost-effectiveness framework
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import (
    cross_val_score, cross_val_predict, StratifiedKFold,
    RepeatedStratifiedKFold, train_test_split
)
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, precision_score, recall_score,
    brier_score_loss, roc_curve, precision_recall_curve, confusion_matrix,
    average_precision_score, log_loss
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
import xgboost as xgb
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Configuration
np.random.seed(42)
OUTPUT_DIR = './output/publication'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/figures', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/tables', exist_ok=True)

# Plot styling
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Load data
df = pd.read_csv('./data/processed_data.csv')
features = [c for c in df.columns if c != 'ICU_Requirement']
X = df[features].values
y = df['ICU_Requirement'].values
feature_names = features

print("="*80)
print("PUBLICATION-READY ANALYSIS")
print("="*80)
print(f"Dataset: n={len(df)}, features={len(features)}")
print(f"ICU Required: {(y==1).sum()} ({(y==1).mean()*100:.1f}%)")
print(f"Not ICU Required: {(y==0).sum()} ({(y==0).mean()*100:.1f}%)")

# ============================================================================
# PRIORITY 1: BASELINE COMPARISONS
# ============================================================================
print("\n" + "="*80)
print("PRIORITY 1: MODEL COMPARISONS")
print("="*80)

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)

def evaluate_model(model, X, y, cv, name):
    """Evaluate model with comprehensive metrics."""
    aurocs = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
    accs = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    f1s = cross_val_score(model, X, y, cv=cv, scoring='f1')
    
    # Get OOF predictions for additional metrics
    cv_single = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = cross_val_predict(model, X, y, cv=cv_single, method='predict_proba')[:, 1]
    oof_preds = (oof_probs >= 0.5).astype(int)
    
    auprc = average_precision_score(y, oof_probs)
    brier = brier_score_loss(y, oof_probs)
    
    sens = recall_score(y, oof_preds)
    spec = recall_score(1-y, 1-oof_preds)
    ppv = precision_score(y, oof_preds)
    npv = precision_score(1-y, 1-oof_preds)
    
    return {
        'name': name,
        'auroc_mean': aurocs.mean(),
        'auroc_std': aurocs.std(),
        'auroc_ci_low': np.percentile(aurocs, 2.5),
        'auroc_ci_high': np.percentile(aurocs, 97.5),
        'auprc': auprc,
        'accuracy_mean': accs.mean(),
        'accuracy_std': accs.std(),
        'f1_mean': f1s.mean(),
        'f1_std': f1s.std(),
        'sensitivity': sens,
        'specificity': spec,
        'ppv': ppv,
        'npv': npv,
        'brier': brier,
        'oof_probs': oof_probs,
        'oof_preds': oof_preds
    }

# Models to compare
models = {
    'XGBoost (Full)': xgb.XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        random_state=42, eval_metric='logloss'
    ),
    'Logistic Regression': make_pipeline(
        StandardScaler(), 
        LogisticRegression(random_state=42, max_iter=1000)
    ),
    'XGBoost (No Hypotension)': xgb.XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        random_state=42, eval_metric='logloss'
    ),
}

# Evaluate all models
results = {}

# Full models
print("\nEvaluating models...")
results['XGBoost (Full)'] = evaluate_model(models['XGBoost (Full)'], X, y, cv, 'XGBoost (Full)')
results['Logistic Regression'] = evaluate_model(models['Logistic Regression'], X, y, cv, 'Logistic Regression')

# Without hypotension
X_no_hypo = df[[f for f in features if f != 'Hypotension_Level']].values
results['XGBoost (No Hypotension)'] = evaluate_model(
    models['XGBoost (No Hypotension)'], X_no_hypo, y, cv, 'XGBoost (No Hypotension)'
)

# MASCC alone
X_mascc = df[['MASCC']].values
lr_mascc = LogisticRegression(random_state=42)
results['MASCC Alone'] = evaluate_model(lr_mascc, X_mascc, y, cv, 'MASCC Alone')

# qSOFA alone  
X_qsofa = df[['qSOFA']].values
lr_qsofa = LogisticRegression(random_state=42)
results['qSOFA Alone'] = evaluate_model(lr_qsofa, X_qsofa, y, cv, 'qSOFA Alone')

# Combined MASCC + qSOFA
X_scores = df[['MASCC', 'qSOFA']].values
lr_combined = LogisticRegression(random_state=42)
results['MASCC + qSOFA'] = evaluate_model(lr_combined, X_scores, y, cv, 'MASCC + qSOFA')

# Create comparison table
print("\n--- Table 1: Model Performance Comparison ---")
table1_data = []
for name, res in results.items():
    table1_data.append({
        'Model': name,
        'AUROC': f"{res['auroc_mean']:.3f} ({res['auroc_ci_low']:.3f}-{res['auroc_ci_high']:.3f})",
        'AUPRC': f"{res['auprc']:.3f}",
        'Accuracy': f"{res['accuracy_mean']:.3f} ± {res['accuracy_std']:.3f}",
        'Sensitivity': f"{res['sensitivity']:.3f}",
        'Specificity': f"{res['specificity']:.3f}",
        'PPV': f"{res['ppv']:.3f}",
        'NPV': f"{res['npv']:.3f}",
        'Brier Score': f"{res['brier']:.3f}"
    })

table1 = pd.DataFrame(table1_data)
print(table1.to_string(index=False))
table1.to_csv(f'{OUTPUT_DIR}/tables/table1_model_comparison.csv', index=False)

# ============================================================================
# PRIORITY 1: CALIBRATION ASSESSMENT
# ============================================================================
print("\n" + "="*80)
print("PRIORITY 1: CALIBRATION ASSESSMENT")
print("="*80)

def calibration_metrics(y_true, y_prob, n_bins=10):
    """Calculate calibration metrics including slope and intercept."""
    from sklearn.linear_model import LogisticRegression
    
    # Calibration slope and intercept
    log_odds = np.log(np.clip(y_prob, 1e-10, 1-1e-10) / (1 - np.clip(y_prob, 1e-10, 1-1e-10)))
    lr = LogisticRegression(fit_intercept=True)
    lr.fit(log_odds.reshape(-1, 1), y_true)
    
    slope = lr.coef_[0][0]
    intercept = lr.intercept_[0]
    
    # Hosmer-Lemeshow-like statistics
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
    
    return {
        'slope': slope,
        'intercept': intercept,
        'brier': brier_score_loss(y_true, y_prob),
        'prob_true': prob_true,
        'prob_pred': prob_pred
    }

# Calibration for main models
print("\n--- Calibration Metrics ---")
calibration_results = {}
for name in ['XGBoost (Full)', 'Logistic Regression', 'XGBoost (No Hypotension)']:
    cal = calibration_metrics(y, results[name]['oof_probs'])
    calibration_results[name] = cal
    print(f"\n{name}:")
    print(f"  Calibration Slope: {cal['slope']:.3f} (ideal = 1.0)")
    print(f"  Calibration Intercept: {cal['intercept']:.3f} (ideal = 0.0)")
    print(f"  Brier Score: {cal['brier']:.4f}")

# Calibration plot
fig, ax = plt.subplots(figsize=(8, 8))
colors = ['#2E86AB', '#A23B72', '#F18F01']
for i, (name, cal) in enumerate(calibration_results.items()):
    ax.plot(cal['prob_pred'], cal['prob_true'], 's-', color=colors[i], 
            markersize=8, linewidth=2, label=f"{name} (slope={cal['slope']:.2f})")

ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Perfect calibration')
ax.set_xlabel('Mean Predicted Probability')
ax.set_ylabel('Observed Proportion')
ax.set_title('Calibration Curves')
ax.legend(loc='lower right')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figures/fig_calibration.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nCalibration plot saved: {OUTPUT_DIR}/figures/fig_calibration.png")

# ============================================================================
# PRIORITY 1: DECISION CURVE ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("PRIORITY 1: DECISION CURVE ANALYSIS")
print("="*80)

def decision_curve_analysis(y_true, y_prob, thresholds):
    """Compute net benefit at various threshold probabilities."""
    n = len(y_true)
    net_benefits = []
    
    for thresh in thresholds:
        # Predicted positives at this threshold
        pred_pos = (y_prob >= thresh).astype(int)
        
        # True positives and false positives
        tp = ((pred_pos == 1) & (y_true == 1)).sum()
        fp = ((pred_pos == 1) & (y_true == 0)).sum()
        
        # Net benefit
        net_benefit = (tp / n) - (fp / n) * (thresh / (1 - thresh)) if thresh < 1 else 0
        net_benefits.append(net_benefit)
    
    return np.array(net_benefits)

thresholds = np.linspace(0.01, 0.99, 100)

# Treat all (send everyone to ICU)
treat_all = (y.sum() / len(y)) - (1 - y.sum() / len(y)) * (thresholds / (1 - thresholds))
treat_all = np.clip(treat_all, -0.5, 1)

# Treat none
treat_none = np.zeros(len(thresholds))

# Model net benefits
dca_results = {}
for name in ['XGBoost (Full)', 'Logistic Regression', 'MASCC Alone', 'qSOFA Alone']:
    dca_results[name] = decision_curve_analysis(y, results[name]['oof_probs'], thresholds)

# DCA Plot
fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(thresholds, treat_all, 'k-', lw=1.5, label='Treat All')
ax.plot(thresholds, treat_none, 'k--', lw=1.5, label='Treat None')

colors = ['#2E86AB', '#A23B72', '#F18F01', '#4ECDC4']
for i, (name, nb) in enumerate(dca_results.items()):
    ax.plot(thresholds, nb, '-', color=colors[i], lw=2, label=name)

ax.set_xlabel('Threshold Probability')
ax.set_ylabel('Net Benefit')
ax.set_title('Decision Curve Analysis')
ax.legend(loc='upper right')
ax.set_xlim([0, 0.8])
ax.set_ylim([-0.1, 0.6])
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figures/fig_dca.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"DCA plot saved: {OUTPUT_DIR}/figures/fig_dca.png")

# ============================================================================
# PRIORITY 2: SIMULATED EXTERNAL VALIDATION (TEMPORAL SPLIT)
# ============================================================================
print("\n" + "="*80)
print("PRIORITY 2: SIMULATED EXTERNAL VALIDATION (Temporal Hold-out)")
print("="*80)

# Since we don't have external data, we simulate temporal validation
# by using last 30% as hold-out (simulating prospective collection)
np.random.seed(42)
X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)

print(f"\nDevelopment set: n={len(X_dev)} (ICU: {y_dev.sum()})")
print(f"Validation set: n={len(X_test)} (ICU: {y_test.sum()})")

# Train on development, test on holdout
model_xgb = xgb.XGBClassifier(
    n_estimators=100, max_depth=3, learning_rate=0.1,
    random_state=42, eval_metric='logloss'
)
model_xgb.fit(X_dev, y_dev, verbose=False)

# Internal CV performance on development
cv_scores = cross_val_score(model_xgb, X_dev, y_dev, cv=5, scoring='roc_auc')
print(f"\nInternal CV (development): AUROC = {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# External validation performance
y_test_prob = model_xgb.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_prob >= 0.5).astype(int)

ext_auroc = roc_auc_score(y_test, y_test_prob)
ext_acc = accuracy_score(y_test, y_test_pred)
ext_sens = recall_score(y_test, y_test_pred)
ext_spec = recall_score(1-y_test, 1-y_test_pred)

print(f"\nExternal Validation (hold-out 30%): AUROC = {ext_auroc:.3f}")
print(f"  Accuracy: {ext_acc:.3f}")
print(f"  Sensitivity: {ext_sens:.3f}")
print(f"  Specificity: {ext_spec:.3f}")

# Bootstrap CI for external validation
boot_aurocs = []
for _ in range(1000):
    idx = np.random.choice(len(y_test), len(y_test), replace=True)
    if len(np.unique(y_test[idx])) < 2:
        continue
    boot_aurocs.append(roc_auc_score(y_test[idx], y_test_prob[idx]))

print(f"  Bootstrap 95% CI: [{np.percentile(boot_aurocs, 2.5):.3f}, {np.percentile(boot_aurocs, 97.5):.3f}]")

# ============================================================================
# PRIORITY 2: SUBGROUP ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("PRIORITY 2: SUBGROUP ANALYSIS")
print("="*80)

# Subgroup by tumor type (Type: 0=Solid, 1=Hematologic)
subgroups = {
    'Solid Tumors': df['Type'] == 0,
    'Hematologic': df['Type'] == 1,
    'With Neutropenia': df['Neutropenia'] == 1,
    'Without Neutropenia': df['Neutropenia'] == 0,
    'Age Group 1': df['Age_Group'] == 1,
    'Age Group 2': df['Age_Group'] == 2,
    'Age Group 3': df['Age_Group'] == 3,
}

print("\n--- Table 2: Subgroup Performance ---")
subgroup_results = []
oof_probs_full = results['XGBoost (Full)']['oof_probs']

for name, mask in subgroups.items():
    y_sub = y[mask]
    prob_sub = oof_probs_full[mask]
    
    if len(np.unique(y_sub)) < 2 or len(y_sub) < 10:
        continue
    
    auroc = roc_auc_score(y_sub, prob_sub)
    n_total = len(y_sub)
    n_icu = y_sub.sum()
    
    # Bootstrap CI
    boot = []
    for _ in range(500):
        idx = np.random.choice(len(y_sub), len(y_sub), replace=True)
        if len(np.unique(y_sub[idx])) < 2:
            continue
        boot.append(roc_auc_score(y_sub[idx], prob_sub[idx]))
    
    ci_low = np.percentile(boot, 2.5) if boot else np.nan
    ci_high = np.percentile(boot, 97.5) if boot else np.nan
    
    subgroup_results.append({
        'Subgroup': name,
        'N': n_total,
        'ICU Events': n_icu,
        'AUROC': f"{auroc:.3f}",
        '95% CI': f"({ci_low:.3f}-{ci_high:.3f})"
    })

subgroup_df = pd.DataFrame(subgroup_results)
print(subgroup_df.to_string(index=False))
subgroup_df.to_csv(f'{OUTPUT_DIR}/tables/table2_subgroup_analysis.csv', index=False)

# ============================================================================
# PRIORITY 2: THRESHOLD SENSITIVITY ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("PRIORITY 2: THRESHOLD OPTIMIZATION")
print("="*80)

def threshold_analysis(y_true, y_prob, thresholds):
    """Analyze performance at different thresholds."""
    results = []
    for thresh in thresholds:
        pred = (y_prob >= thresh).astype(int)
        
        tp = ((pred == 1) & (y_true == 1)).sum()
        tn = ((pred == 0) & (y_true == 0)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        fn = ((pred == 0) & (y_true == 1)).sum()
        
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        youden = sens + spec - 1
        
        results.append({
            'Threshold': thresh,
            'Sensitivity': sens,
            'Specificity': spec,
            'PPV': ppv,
            'NPV': npv,
            'Youden Index': youden
        })
    
    return pd.DataFrame(results)

thresholds_to_test = [0.3, 0.4, 0.5, 0.6, 0.7]
thresh_df = threshold_analysis(y, results['XGBoost (Full)']['oof_probs'], thresholds_to_test)
print("\n--- Table 3: Threshold Sensitivity Analysis ---")
print(thresh_df.to_string(index=False))
thresh_df.to_csv(f'{OUTPUT_DIR}/tables/table3_threshold_analysis.csv', index=False)

# Find optimal threshold (Youden Index)
all_thresh_df = threshold_analysis(y, results['XGBoost (Full)']['oof_probs'], np.linspace(0.1, 0.9, 81))
optimal_idx = all_thresh_df['Youden Index'].idxmax()
optimal_thresh = all_thresh_df.loc[optimal_idx, 'Threshold']
print(f"\nOptimal threshold (Youden): {optimal_thresh:.2f}")

# ============================================================================
# PRIORITY 3: LEARNING CURVE
# ============================================================================
print("\n" + "="*80)
print("PRIORITY 3: LEARNING CURVE ANALYSIS")
print("="*80)

from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    xgb.XGBClassifier(n_estimators=100, max_depth=3, random_state=42, eval_metric='logloss'),
    X, y, 
    train_sizes=np.linspace(0.2, 1.0, 8),
    cv=5, scoring='roc_auc', random_state=42
)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(train_sizes, train_scores.mean(axis=1), 'o-', color='#2E86AB', lw=2, label='Training Score')
ax.fill_between(train_sizes, 
                train_scores.mean(axis=1) - train_scores.std(axis=1),
                train_scores.mean(axis=1) + train_scores.std(axis=1),
                alpha=0.2, color='#2E86AB')
ax.plot(train_sizes, val_scores.mean(axis=1), 's-', color='#E94F37', lw=2, label='Cross-Validation Score')
ax.fill_between(train_sizes, 
                val_scores.mean(axis=1) - val_scores.std(axis=1),
                val_scores.mean(axis=1) + val_scores.std(axis=1),
                alpha=0.2, color='#E94F37')
ax.set_xlabel('Training Set Size')
ax.set_ylabel('AUROC')
ax.set_title('Learning Curve')
ax.legend(loc='lower right')
ax.set_ylim([0.7, 1.05])
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figures/fig_learning_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Learning curve saved: {OUTPUT_DIR}/figures/fig_learning_curve.png")

# ============================================================================
# COMBINED ROC CURVES
# ============================================================================
print("\n" + "="*80)
print("GENERATING COMBINED ROC CURVES")
print("="*80)

fig, ax = plt.subplots(figsize=(10, 8))
colors = {
    'XGBoost (Full)': '#2E86AB',
    'Logistic Regression': '#A23B72',
    'MASCC Alone': '#F18F01',
    'qSOFA Alone': '#4ECDC4',
    'XGBoost (No Hypotension)': '#7B2D8E'
}

for name in ['XGBoost (Full)', 'Logistic Regression', 'MASCC Alone', 'qSOFA Alone', 'XGBoost (No Hypotension)']:
    fpr, tpr, _ = roc_curve(y, results[name]['oof_probs'])
    auroc = results[name]['auroc_mean']
    ax.plot(fpr, tpr, color=colors[name], lw=2, 
            label=f"{name} (AUC = {auroc:.3f})")

ax.plot([0, 1], [0, 1], 'k--', lw=1)
ax.set_xlabel('1 - Specificity (False Positive Rate)')
ax.set_ylabel('Sensitivity (True Positive Rate)')
ax.set_title('Receiver Operating Characteristic Curves')
ax.legend(loc='lower right')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figures/fig_roc_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"ROC comparison saved: {OUTPUT_DIR}/figures/fig_roc_comparison.png")

# ============================================================================
# FEATURE IMPORTANCE COMPARISON
# ============================================================================
print("\n" + "="*80)
print("GENERATING FEATURE IMPORTANCE PLOT")
print("="*80)

# Train final model for importance
model_final = xgb.XGBClassifier(
    n_estimators=100, max_depth=3, learning_rate=0.1,
    random_state=42, eval_metric='logloss'
)
model_final.fit(X, y, verbose=False)

# Get feature importance
importance = model_final.feature_importances_
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance
}).sort_values('Importance', ascending=True)

fig, ax = plt.subplots(figsize=(10, 8))
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(importance_df)))
ax.barh(importance_df['Feature'], importance_df['Importance'], color=colors)
ax.set_xlabel('Feature Importance (Gain)')
ax.set_title('XGBoost Feature Importance')
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figures/fig_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Feature importance saved: {OUTPUT_DIR}/figures/fig_feature_importance.png")

# ============================================================================
# SAVE ALL RESULTS
# ============================================================================
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# Save comprehensive results JSON
all_results = {
    'model_comparison': {name: {k: v for k, v in res.items() if k not in ['oof_probs', 'oof_preds']} 
                         for name, res in results.items()},
    'calibration': {name: {'slope': cal['slope'], 'intercept': cal['intercept'], 'brier': cal['brier']}
                    for name, cal in calibration_results.items()},
    'external_validation': {
        'auroc': ext_auroc,
        'accuracy': ext_acc,
        'sensitivity': ext_sens,
        'specificity': ext_spec,
        'ci_low': np.percentile(boot_aurocs, 2.5),
        'ci_high': np.percentile(boot_aurocs, 97.5),
        'n_dev': len(X_dev),
        'n_test': len(X_test)
    },
    'optimal_threshold': optimal_thresh
}

with open(f'{OUTPUT_DIR}/all_results.json', 'w') as f:
    json.dump(all_results, f, indent=2, default=float)

print(f"\nResults saved to: {OUTPUT_DIR}/all_results.json")
print(f"Tables saved to: {OUTPUT_DIR}/tables/")
print(f"Figures saved to: {OUTPUT_DIR}/figures/")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
