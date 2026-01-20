# ICU Requirement Prediction - Combined Model Analysis Report

## Executive Summary

This report presents a comprehensive analysis of two XGBoost models for predicting ICU admission in febrile oncology patients:

1. **Base Model** — Full feature set including hypotension status
2. **No-Hypotension Model** — Sensitivity analysis excluding hypotension

| Model | AUROC (95% CI) | AUPRC | Accuracy | F1 |
|-------|----------------|-------|----------|-----|
| **Base Model** | **0.925** (0.913-0.936) | **0.948** | **0.832** | **0.845** |
| No-Hypotension | 0.821 (0.803-0.839) | 0.859 | 0.773 | 0.797 |
| Δ (Difference) | -0.104 | -0.089 | -0.059 | -0.048 |

> **Key Finding**: Removing hypotension reduced AUROC from 0.925 to 0.821, indicating remaining predictive signal beyond deterministic triage triggers.

---

## 1. Study Overview

### 1.1 Objective
Predict whether oncology patients presenting with febrile illness require ICU admission using clinical features available at initial presentation.

### 1.2 Rationale for Dual Model Analysis
Hypotension requiring inotropes (Level 2) results in near-deterministic ICU admission per clinical protocol. The no-hypotension model assesses whether remaining features provide useful discrimination independent of this triage trigger.

### 1.3 Cohort
- **Sample Size**: 149 patients
- **ICU Required**: 81 (54.4%)
- **Not ICU Required**: 68 (45.6%)

---

## 2. Feature Sets

### 2.1 Comparison

| Feature | Base Model | No-Hypotension |
|---------|:----------:|:--------------:|
| **Hypotension_Level** | ✓ | ✗ |
| qSOFA | ✓ | ✓ |
| MASCC | ✓ | ✓ |
| Type | ✓ | ✓ |
| Neutropenia | ✓ | ✓ |
| Mets_Binary | ✓ | ✓ |
| Mets_Missing | ✓ | ✓ |
| Comorb | ✓ | ✓ |
| Line_Rx | ✓ | ✓ |
| Focus_PneumResp | ✓ | ✓ |
| UTI | ✓ | ✓ |
| Focus_Bloodstream | ✓ | ✓ |
| Focus_GI_Hepatobiliary | ✓ | ✓ |
| Focus_SoftTissue | ✓ | ✓ |
| Focus_NoneUnknown | ✓ | ✓ |
| Gender | ✓ | ✓ |
| Age_Group | ✓ | ✓ |
| **Total Features** | **17** | **16** |

### 2.2 Feature Categories

| Category | Features |
|----------|----------|
| Clinical Scores | MASCC, qSOFA |
| Hemodynamics | Hypotension_Level (base only) |
| Cancer Characteristics | Type, Mets_Binary, Mets_Missing |
| Infection Focus | Focus_PneumResp, UTI, Focus_Bloodstream, Focus_GI_Hepatobiliary, Focus_SoftTissue, Focus_NoneUnknown |
| Clinical Status | Neutropenia, Comorb, Line_Rx |
| Demographics | Gender, Age_Group |

---

## 3. Methodology

### 3.1 Model Configuration

Both models used identical XGBoost configuration for fair comparison:

| Parameter | Value |
|-----------|-------|
| Algorithm | XGBoost Classifier |
| n_estimators | 100 |
| max_depth | 3 |
| learning_rate | 0.1 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| min_child_weight | 5 |
| reg_alpha (L1) | 0.1 |
| reg_lambda (L2) | 1.0 |

### 3.2 Validation Strategy

| Aspect | Configuration |
|--------|---------------|
| Method | Repeated Stratified K-Fold |
| K-Folds | 5 |
| Repeats | 10 |
| Total Evaluations | 50 |
| Confidence Intervals | Bootstrap (1000 iterations) |
| Random Seed | 42 |

### 3.3 Leakage Prevention
- All metrics computed from out-of-fold predictions
- No preprocessing steps requiring fit (data pre-encoded)
- Model trained only on training folds per split

---

## 4. Performance Comparison

### 4.1 Primary Metrics

| Metric | Base Model | No-Hypotension | Δ | % Change |
|--------|------------|----------------|---|----------|
| **AUROC** | 0.9248 | 0.8207 | -0.1041 | -11.3% |
| **AUPRC** | 0.9483 | 0.8588 | -0.0895 | -9.4% |
| **Accuracy** | 0.8317 | 0.7727 | -0.0590 | -7.1% |
| **F1 Score** | 0.8450 | 0.7966 | -0.0484 | -5.7% |

### 4.2 95% Confidence Intervals

| Metric | Base Model CI | No-Hypotension CI |
|--------|---------------|-------------------|
| AUROC | [0.913, 0.936] | [0.803, 0.839] |
| AUPRC | [0.941, 0.955] | [0.841, 0.876] |
| Accuracy | [0.817, 0.845] | [0.757, 0.790] |
| F1 | [0.831, 0.857] | [0.781, 0.812] |

### 4.3 Confusion Matrices (Out-of-Fold Aggregated)

#### Base Model
```
               Predicted
              Not ICU  |  ICU
Actual  Not ICU  55    |   13
        ICU      12    |   69

Sensitivity: 0.852 (69/81)
Specificity: 0.809 (55/68)
PPV: 0.841 (69/82)
NPV: 0.821 (55/67)
```

#### No-Hypotension Model
```
               Predicted
              Not ICU  |  ICU
Actual  Not ICU  50    |   18
        ICU      13    |   68

Sensitivity: 0.840 (68/81)
Specificity: 0.735 (50/68)
PPV: 0.791 (68/86)
NPV: 0.794 (50/63)
```

### 4.4 Performance Interpretation

| Aspect | Base Model | No-Hypotension | Interpretation |
|--------|------------|----------------|----------------|
| Discrimination | Excellent (>0.9) | Good (>0.8) | Both clinically useful |
| Sensitivity | 85.2% | 84.0% | Minimal loss |
| Specificity | 80.9% | 73.5% | Moderate reduction |
| PPV | 84.1% | 79.1% | Slight decrease |
| False Positives | 13 | 18 | +5 additional |
| False Negatives | 12 | 13 | +1 additional |

---

## 5. Feature Importance Comparison

### 5.1 Gain-Based Importance (Top 10)

| Rank | Base Model | Importance | No-Hypotension | Importance |
|------|------------|------------|----------------|------------|
| 1 | **Hypotension_Level** | — | **qSOFA** | 10.25 |
| 2 | qSOFA | — | Mets_Binary | 2.67 |
| 3 | Mets_Binary | — | Comorb | 1.40 |
| 4 | Comorb | — | UTI | 1.11 |
| 5 | Focus_PneumResp | — | Age_Group | 0.87 |
| 6 | UTI | — | Type | 0.66 |
| 7 | Age_Group | — | Line_Rx | 0.58 |
| 8 | Type | — | Neutropenia | 0.55 |
| 9 | Line_Rx | — | Gender | 0.54 |
| 10 | Neutropenia | — | Focus_PneumResp | 0.48 |

### 5.2 Permutation Importance (No-Hypotension Model)

| Rank | Feature | Mean Decrease in AUROC |
|------|---------|------------------------|
| 1 | **qSOFA** | 0.251 ± 0.031 |
| 2 | Mets_Binary | 0.039 ± 0.013 |
| 3 | Comorb | 0.020 ± 0.008 |
| 4 | Age_Group | 0.012 ± 0.008 |
| 5 | Type | 0.009 ± 0.006 |

### 5.3 Feature Importance Insights

1. **Hypotension dominates the base model** — When included, it is the strongest predictor
2. **qSOFA becomes dominant without hypotension** — With permutation importance of 0.25, qSOFA accounts for most of the remaining signal
3. **Metastatic status and comorbidities** — Provide secondary predictive value
4. **Infection focus** — Contributes modestly but consistently

---

## 6. Clinical Interpretation

### 6.1 Why Hypotension Matters
- Patients with Hypotension_Level=2 (requiring inotropes) have near-deterministic ICU admission
- This reflects clinical protocol, not model insight
- Removing hypotension reveals underlying biological risk factors

### 6.2 Residual Predictive Signal
Even without hypotension, the model achieves AUROC 0.821, indicating:
- **qSOFA** captures physiologic derangement (respiratory rate, mentation, blood pressure already incorporated)
- **Metastatic disease** reflects tumor burden and immunocompromise
- **Comorbidities** indicate baseline vulnerability
- **Age** correlates with physiologic reserve

### 6.3 Use Cases

| Scenario | Recommended Model |
|----------|-------------------|
| **Standard triage (hemodynamics known)** | Base Model |
| **Early triage (pre-hemodynamic assessment)** | No-Hypotension Model |
| **Resource-limited settings** | No-Hypotension Model |
| **Research (non-circulatory predictors)** | No-Hypotension Model |

---

## 7. Validation Summary

| Validation Check | Base Model | No-Hypotension |
|------------------|:----------:|:--------------:|
| Out-of-fold predictions | ✅ | ✅ |
| No preprocessing leakage | ✅ | ✅ |
| Stratified sampling | ✅ | ✅ |
| Bootstrap CIs | ✅ | ✅ |
| Same hyperparameters | ✅ | ✅ |
| Same random seed | ✅ | ✅ |
| Hypotension excluded | — | ✅ |

---

## 8. Visualizations

### Base Model
| Plot | Path |
|------|------|
| ROC Curve | `output/plots/roc_curve.png` |
| PR Curve | `output/plots/pr_curve.png` |
| Feature Importance | `output/plots/feature_importance_gain.png` |
| SHAP Summary | `output/plots/shap_summary.png` |
| Calibration | `output/plots/calibration_curve.png` |

### No-Hypotension Model
| Plot | Path |
|------|------|
| ROC Curve | `output/no_hypotension/plots/roc_curve.png` |
| PR Curve | `output/no_hypotension/plots/pr_curve.png` |
| Feature Importance | `output/no_hypotension/plots/feature_importance_gain.png` |
| SHAP Summary | `output/no_hypotension/plots/shap_summary.png` |
| Calibration | `output/no_hypotension/plots/calibration_curve.png` |

---

## 9. Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| Small sample size (n=149) | Wide confidence intervals | Conservative regularization |
| Single-center data | Limited generalizability | External validation needed |
| Retrospective design | Potential selection bias | Prospective validation needed |
| Pre-encoded features | Limited inspection of distributions | Original values documented |
| Class imbalance (54/46) | Minimal concern | Stratified CV used |

---

## 10. Conclusions

### 10.1 Base Model
- Achieves excellent discrimination (AUROC 0.925)
- Hypotension_Level is the dominant predictor
- Suitable for standard clinical triage when hemodynamic data available

### 10.2 No-Hypotension Model
- Retains good discrimination (AUROC 0.821)
- qSOFA becomes the dominant predictor
- Useful for early triage before complete hemodynamic assessment
- Demonstrates residual predictive signal in non-circulatory features

### 10.3 Key Finding

> **Removing hypotension reduced AUROC from 0.925 (base model) to 0.821, indicating remaining predictive signal beyond deterministic triage triggers.**

This supports the clinical utility of the model even in early-stage triage when hemodynamic status is not yet fully characterized.

---

## 11. Recommendations

1. **Primary model for clinical use**: Base Model (with hypotension)
2. **Early warning/pre-triage**: Consider No-Hypotension Model for initial assessment
3. **Next steps**:
   - External validation at independent sites
   - Prospective validation study
   - Integration with EHR for real-time scoring

---

## Appendix: File Locations

### Scripts
| Script | Purpose | Path |
|--------|---------|------|
| train.py | Base model training | `src/train.py` |
| train_no_hypotension.py | No-hypotension training | `src/train_no_hypotension.py` |
| evaluate.py | Base model evaluation | `src/evaluate.py` |
| evaluate_no_hypotension.py | No-hypotension evaluation | `src/evaluate_no_hypotension.py` |

### Output Artifacts
| Artifact | Base Model | No-Hypotension |
|----------|------------|----------------|
| Metrics JSON | `output/metrics.json` | `output/no_hypotension/metrics.json` |
| OOF Predictions | `output/cv_predictions.csv` | `output/no_hypotension/cv_predictions.csv` |
| Feature Importance | `output/feature_importance.csv` | `output/no_hypotension/feature_importance.csv` |
| Saved Model | `output/xgboost_model.pkl` | `output/no_hypotension/xgboost_model.pkl` |
| Report | `output/REPORT.md` | `output/no_hypotension/REPORT.md` |

---

*Report generated: 2026-01-05 23:29*  
*Analysis: XGBoost with 10×5-fold stratified cross-validation*  
*Random seed: 42*
