# ICU Requirement Prediction for Oncology Patients

## Project Overview

This project develops a machine learning model to predict whether oncology patients presenting with febrile illness require ICU admission, using clinical features available at initial presentation.

**Target Variable**: ICU Requirement (Binary)
- 1 = ICU Required
- 0 = Not ICU Required

## Results Summary

| Metric | Value | 95% CI |
|--------|-------|--------|
| **AUROC** | 0.9248 | [0.9133, 0.9358] |
| **AUPRC** | 0.9483 | [0.9414, 0.9551] |
| **Accuracy** | 0.8317 | [0.8172, 0.8452] |
| **F1 Score** | 0.8450 | [0.8313, 0.8574] |

## Top Predictive Features

1. **Hypotension_Level** - Most important predictor
2. **qSOFA** - Sepsis severity indicator
3. **Mets_Binary** - Metastatic disease status
4. **Comorb** - Comorbidity burden
5. **Focus_PneumResp** - Respiratory infection focus

## Repository Structure

```
icu2/
├── src/
│   ├── data_prep.py      # Data ingestion and validation
│   ├── train.py          # XGBoost training with CV
│   └── evaluate.py       # Evaluation and visualization
├── pipelines/
│   └── pipeline.py       # SageMaker Pipeline definition
├── data/                 # Local data storage
├── output/               # Model outputs
│   ├── metrics.json      # CV metrics with CIs
│   ├── cv_predictions.csv
│   ├── feature_importance.csv
│   ├── xgboost_model.pkl
│   ├── REPORT.md         # Full evaluation report
│   └── plots/            # Visualization outputs
├── requirements.txt
└── README.md
```

## Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Run Pipeline Locally

```bash
# Step 1: Data Preparation
python src/data_prep.py

# Step 2: Model Training
python src/train.py

# Step 3: Evaluation
python src/evaluate.py
```

Or run all steps via the pipeline script:

```bash
python pipelines/pipeline.py --local
```

### Run with SageMaker

```bash
# Create/update pipeline
python pipelines/pipeline.py --create --role <your-sagemaker-role-arn>

# Execute pipeline
python pipelines/pipeline.py --execute --role <your-sagemaker-role-arn>
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| S3 Bucket | `icu-required` | S3 bucket for data |
| S3 Key | `cleaned_data.csv` | Path to input CSV |
| Region | `us-east-1` | AWS region |
| K-Folds | 5 | Cross-validation folds |
| N-Repeats | 10 | CV repetitions |
| Random Seed | 42 | For reproducibility |

## Model Details

- **Algorithm**: XGBoost Classifier
- **Validation**: 10-repeat 5-fold stratified cross-validation
- **Hyperparameters**:
  - max_depth: 3 (conservative for small dataset)
  - learning_rate: 0.1
  - n_estimators: 100
  - subsample: 0.8
  - min_child_weight: 5

## Outputs

| File | Description |
|------|-------------|
| `metrics.json` | CV metrics with bootstrap CIs |
| `cv_predictions.csv` | Out-of-fold predictions |
| `feature_importance.csv` | Gain-based importance |
| `permutation_importance.csv` | Permutation importance |
| `xgboost_model.pkl` | Trained model |
| `REPORT.md` | Full evaluation report |
| `plots/roc_curve.png` | ROC curve |
| `plots/pr_curve.png` | Precision-Recall curve |
| `plots/calibration_curve.png` | Calibration plot |
| `plots/confusion_matrix.png` | Confusion matrix |
| `plots/feature_importance_gain.png` | Feature importance (gain) |
| `plots/permutation_importance.png` | Permutation importance |
| `plots/shap_summary.png` | SHAP summary plot |

## Limitations

1. Small sample size (~149 patients)
2. Single-center data (generalizability uncertain)
3. No prospective temporal validation
4. **NOT validated for clinical use**

## License

For research purposes only. Requires prospective validation before clinical implementation.
