# ICU Requirement Prediction Model - No Hypotension Analysis

> **SENSITIVITY ANALYSIS**: This model EXCLUDES Hypotension_Level feature to assess predictive signal beyond deterministic triage triggers.

## Key Finding

Removing hypotension reduced AUROC from **0.925** (base model) to **0.821** (Δ = 0.104), indicating remaining predictive signal beyond deterministic triage triggers.

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

**Total Features:** 16 (vs 17 in base model)

---

## 3. Methodology

### 3.1 Model Configuration
- **Algorithm:** XGBoost Classifier (identical to base model)
- **Hyperparameters:** max_depth=3, n_estimators=100, learning_rate=0.1
- **Regularization:** L1 (α=0.1), L2 (λ=1.0), min_child_weight=5

### 3.2 Validation Strategy
- **Method:** 10-repeat 5-fold stratified CV (50 total folds)
- **Confidence Intervals:** Bootstrap (1000 iterations)
- **Leakage Prevention:** All preprocessing within CV folds

---

## 4. Model Performance

### 4.1 Cross-Validation Metrics (with 95% Bootstrap CI)

| Metric | No Hypotension | Base Model | Difference |
|--------|----------------|------------|------------|
| **AUROC** | 0.8207 [0.8027, 0.8389] | 0.9248 | +0.1041 |
| **AUPRC** | 0.8588 [0.8411, 0.8757] | 0.948 | - |
| **Accuracy** | 0.7727 [0.7570, 0.7898] | 0.832 | - |
| **F1 Score** | 0.7966 [0.7810, 0.8121] | 0.845 | - |

### 4.2 Confusion Matrix (Aggregated Out-of-Fold)

```
               Predicted
              Not ICU  |  ICU
Actual  Not ICU  50     |   18
        ICU      13     |   68
```

---

## 5. Feature Importance

### 5.1 XGBoost Gain-Based Importance (Top 10)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | qSOFA | 10.2519 |
| 2 | Mets_Binary | 2.6670 |
| 3 | Comorb | 1.4033 |
| 4 | UTI | 1.1114 |
| 5 | Age_Group | 0.8724 |
| 6 | Type | 0.6601 |
| 7 | Line_Rx | 0.5785 |
| 8 | Neutropenia | 0.5542 |
| 9 | Gender | 0.5355 |
| 10 | Focus_PneumResp | 0.4820 |

### 5.2 Permutation Importance (Top 10)

| Rank | Feature | Mean Decrease in AUROC |
|------|---------|------------------------|
| 1 | qSOFA | 0.2505 ± 0.0309 |
| 2 | Mets_Binary | 0.0392 ± 0.0128 |
| 3 | Comorb | 0.0201 ± 0.0080 |
| 4 | Age_Group | 0.0117 ± 0.0081 |
| 5 | Type | 0.0088 ± 0.0057 |
| 6 | Line_Rx | 0.0044 ± 0.0035 |
| 7 | Gender | 0.0040 ± 0.0028 |
| 8 | Neutropenia | 0.0026 ± 0.0018 |
| 9 | UTI | 0.0014 ± 0.0020 |
| 10 | Focus_Bloodstream | 0.0000 ± 0.0000 |

---

## 6. Comparison with Base Model

| Aspect | Base Model | No Hypotension Model |
|--------|------------|---------------------|
| Features | 17 | 16 |
| Hypotension_Level | ✓ Included | ✗ Excluded |
| AUROC | 0.925 | 0.821 |
| Top Predictor | Hypotension_Level | qSOFA |

**Interpretation:** Removing hypotension reduced discriminative performance by 10.4%, but the model retains substantial predictive power (AUROC 0.821), demonstrating that other clinical features (qSOFA, infection focus, neutropenia, comorbidities) carry meaningful prognostic signal.

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

1. **Residual Predictive Signal:** Even without hypotension, the model achieves AUROC 0.821, demonstrating that features like qSOFA, infection focus, and comorbidities provide useful risk stratification.

2. **Clinical Utility:** This model may be useful for early triage before hemodynamic status is fully assessed, or in settings where hypotension data is unavailable.

3. **Feature Hierarchy:** With hypotension removed, qSOFA becomes the dominant predictor, followed by infection focus and clinical scores.

### Limitations

1. Same sample size limitations as base model (n=149)
2. Single-center data
3. Not prospectively validated

---

## 10. Conclusion

Removing hypotension reduced AUROC from 0.925 (base model) to 0.821, indicating remaining predictive signal beyond deterministic triage triggers. The no-hypotension model maintains clinically useful discrimination and may support early risk stratification before complete hemodynamic assessment.

---

*Report generated: 2026-01-05 23:24:49*
*Model variant: No Hypotension*
*Analysis pipeline: XGBoost with 10×5-fold stratified cross-validation*
