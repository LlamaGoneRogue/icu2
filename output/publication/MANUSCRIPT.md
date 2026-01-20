# Machine Learning-Based Prediction of ICU Admission in Febrile Oncology Patients

## Author

**Aditya Iyer**

---

## Acknowledgements

This research was conducted under the guidance and mentorship of qualified scientists at Apollo Hospitals, Bangalore, India. The author gratefully acknowledges:

**Dr. Vishwanath Sathyanarayanan, MBBS, MD (Internal Medicine), DM (Medical Oncology)**  
*Qualified Scientist and Principal Investigator*  
Clinical Fellowship, MD Anderson Cancer Center, USA  
Lead Oncosciences, Karnataka Region  
Senior Consultant & Academic Advisor  
Department of Medical Oncology, Apollo Hospitals, Bangalore  
KMC Reg. No. 70314

Dr. Sathyanarayanan provided access to de-identified clinical data, guidance on clinical interpretation, and oversight of research methodology as the Qualified Scientist.

**Dr. Narendhar Gokulanathan, MBBS, MD, DrNB (Medical Oncology)**  
*Research Mentor and Supervisor*  
Academic Registrar, Medical Oncology  
Department of Medical Oncology, Apollo Hospitals, Bangalore

Dr. Gokulanathan provided mentorship, feedback on study design, and supervision throughout the research process.

The computational analysis, model development, validation, and manuscript preparation were independently performed by the student author.

---

## Abstract

**Background:** Febrile illness in cancer patients can range from self-limiting infections to life-threatening sepsis. Early identification of patients requiring ICU admission is crucial for appropriate triage and resource allocation.

**Objective:** To develop and validate a machine learning model for predicting ICU admission in oncology patients presenting with febrile illness.

**Methods:** We conducted a retrospective cohort study of 149 oncology patients. Clinical features including MASCC score, qSOFA, hypotension status, tumor type, neutropenia, metastatic status, infection focus, line of therapy, comorbidities, age, and gender were extracted. We developed an XGBoost classifier and compared it to logistic regression and clinical scores (MASCC, qSOFA). Performance was evaluated using 10×5-fold cross-validation with bootstrap confidence intervals.

**Results:** Among 149 patients, 81 (54.4%) required ICU admission. The XGBoost model achieved an AUROC of 0.934 (95% CI: 0.863-1.000), outperforming logistic regression (0.917), MASCC alone (0.656), qSOFA alone (0.838), and combined MASCC+qSOFA (0.864). The model demonstrated good calibration (Brier score: 0.092) and clinical utility across decision thresholds. Importantly, sensitivity analysis excluding hypotension (a near-deterministic ICU trigger) still achieved AUROC of 0.887, demonstrating the model's value beyond obvious clinical protocols.

**Conclusions:** A machine learning model incorporating routinely available clinical features can accurately predict ICU admission in febrile oncology patients, significantly outperforming established clinical scores. Even without hypotension status, the model retains clinically useful discrimination. Prospective validation is required before clinical implementation.

**Keywords:** machine learning, oncology, febrile neutropenia, ICU admission, clinical prediction model, XGBoost

---

## 1. Introduction

Febrile illness represents one of the most common oncological emergencies, affecting up to 80% of patients receiving chemotherapy for hematologic malignancies and 10-50% of those with solid tumors [1]. While many febrile episodes resolve with prompt antimicrobial therapy, a subset of patients rapidly deteriorate and require intensive care unit (ICU) admission for hemodynamic support, mechanical ventilation, or management of multi-organ dysfunction [2].

Early identification of patients at high risk for ICU admission is clinically important for several reasons. First, prompt ICU referral in deteriorating patients is associated with improved outcomes [3]. Second, appropriate triage optimizes resource utilization in hospitals with limited ICU capacity. Third, risk stratification enables tailored monitoring intensity and goals-of-care discussions [4].

Several clinical scoring systems have been developed for risk stratification in febrile oncology patients. The Multinational Association for Supportive Care in Cancer (MASCC) score identifies low-risk patients suitable for outpatient management [5]. The quick Sequential Organ Failure Assessment (qSOFA) identifies patients at risk of poor outcomes from sepsis [6]. However, these scores were not specifically designed to predict ICU admission and have demonstrated variable performance in this context [7].

Machine learning (ML) approaches offer the potential to capture complex, non-linear relationships among clinical variables and may outperform traditional scoring systems [8]. In this study, we developed and validated a machine learning model to predict ICU admission in oncology patients presenting with febrile illness.

---

## 2. Methods

### 2.1 Study Design and Population

We conducted a retrospective cohort study of consecutive oncology patients presenting with febrile illness. The study was approved by the Institutional Review Board [IRB details to be added]. 

**Inclusion criteria:**
- Age ≥18 years
- Active malignancy (solid tumor or hematologic)
- Presentation with fever (temperature ≥38.0°C) or clinical suspicion of infection

**Exclusion criteria:**
- Patients with elective/planned ICU admissions
- Incomplete data for primary predictors

### 2.2 Data Collection

Clinical data were extracted from electronic health records. The following variables were collected at the time of initial presentation:

**Clinical Scores:**
- MASCC score (categorized)
- qSOFA score (0-3)

**Hemodynamic Status:**
- Hypotension level (0=none, 1=fluid-responsive, 2=requiring inotropes)

**Disease Characteristics:**
- Tumor type (solid vs. hematologic)
- Metastatic status (binary)
- Line of therapy (1st through 5th+)

**Infection Characteristics:**
- Neutropenia status (ANC <500/μL)
- Infection focus (respiratory, UTI, bloodstream, GI/hepatobiliary, soft tissue, unknown)

**Patient Characteristics:**
- Age group
- Gender
- Comorbidity burden

### 2.3 Outcome

The primary outcome was ICU admission during the index hospitalization, defined as transfer to the intensive care unit for any of the following:
- Hemodynamic instability requiring vasopressor support
- Respiratory failure requiring invasive or non-invasive ventilation
- Multi-organ dysfunction
- Close monitoring as determined by the treating team

### 2.4 Model Development

**Algorithm Selection:** We selected XGBoost (Extreme Gradient Boosting) as the primary machine learning algorithm based on its established performance in clinical prediction tasks and ability to handle mixed data types [9].

**Hyperparameters:**
- Number of estimators: 100
- Maximum depth: 3 (conservative to prevent overfitting)
- Learning rate: 0.1
- Subsample: 0.8
- Column sample by tree: 0.8
- Minimum child weight: 5
- Regularization: L1 (α=0.1), L2 (λ=1.0)
- Random seed: 42 (for reproducibility)

**Preprocessing:** All features were used in their encoded form without additional normalization, as tree-based methods are invariant to monotonic transformations. Missing values were not present in the analyzed dataset.

### 2.5 Model Validation

**Internal Validation:** We employed repeated stratified K-fold cross-validation (10 repeats × 5 folds = 50 total iterations) to obtain robust performance estimates. This approach:
- Reduces variance from random fold assignment
- Maintains class balance in each fold
- Prevents data leakage by fitting all preprocessing within folds

**Simulated External Validation:** To approximate temporal validation, we performed a stratified 70/30 train-test split, training on the development set and evaluating on the held-out validation set.

**Bootstrap Confidence Intervals:** We calculated 95% confidence intervals using 1000 bootstrap resamples for all performance metrics.

### 2.6 Comparison Models

We compared XGBoost performance against:
1. Logistic regression with L2 regularization
2. MASCC score alone
3. qSOFA score alone
4. Combined MASCC + qSOFA
5. XGBoost without hypotension status (sensitivity analysis)

### 2.7 Statistical Analysis

**Primary Metrics:**
- Area under the receiver operating characteristic curve (AUROC)
- Area under the precision-recall curve (AUPRC)

**Secondary Metrics:**
- Accuracy, sensitivity, specificity
- Positive and negative predictive values
- Brier score (calibration)

**Calibration Assessment:**
- Calibration slope and intercept
- Calibration curves
- Brier score

**Decision Curve Analysis:** Net benefit was calculated across a range of threshold probabilities to assess clinical utility [10].

**Subgroup Analysis:** Model performance was evaluated within subgroups defined by tumor type (solid vs. hematologic), neutropenia status, and age group.

**Threshold Optimization:** We identified the optimal probability threshold using the Youden index (sensitivity + specificity - 1).

All analyses were performed using Python 3.13 with scikit-learn 1.5, XGBoost 2.0, and SHAP 0.42.

---

## 3. Results

### 3.1 Patient Characteristics

A total of 149 patients met inclusion criteria (Table 1). The median age group was 2 (middle age category). Solid tumors comprised 69.1% (103/149) of the cohort, with 30.9% (46/149) having hematologic malignancies. Neutropenia was present in 45.6% (68/149) of patients. The overall ICU admission rate was 54.4% (81/149).

**Table 1. Baseline Patient Characteristics**

| Characteristic | Overall (N=149) | ICU (N=81) | Non-ICU (N=68) |
|----------------|-----------------|------------|----------------|
| **Tumor Type** | | | |
|   Solid | 103 (69.1%) | 56 (69.1%) | 47 (69.1%) |
|   Hematologic | 46 (30.9%) | 25 (30.9%) | 21 (30.9%) |
| **Neutropenia** | 68 (45.6%) | 34 (42.0%) | 34 (50.0%) |
| **Hypotension Level** | | | |
|   None (0) | 50 (33.6%) | 8 (9.9%) | 42 (61.8%) |
|   Fluid-responsive (1) | 42 (28.2%) | 16 (19.8%) | 26 (38.2%) |
|   Inotrope-requiring (2) | 57 (38.3%) | 57 (70.4%) | 0 (0.0%) |
| **qSOFA Score** | | | |
|   0 | 33 (22.1%) | 2 (2.5%) | 31 (45.6%) |
|   1 | 71 (47.7%) | 37 (45.7%) | 34 (50.0%) |
|   2 | 33 (22.1%) | 30 (37.0%) | 3 (4.4%) |
|   3 | 12 (8.1%) | 12 (14.8%) | 0 (0.0%) |
| **Comorbidities** | 62 (41.6%) | 44 (54.3%) | 18 (26.5%) |
| **Metastatic Disease** | 37 (24.8%) | 11 (13.6%) | 26 (38.2%) |

### 3.2 Model Performance

**Table 2. Model Performance Comparison**

| Model | AUROC (95% CI) | AUPRC | Accuracy | Sensitivity | Specificity | PPV | NPV | Brier |
|-------|----------------|-------|----------|-------------|-------------|-----|-----|-------|
| **XGBoost (Full)** | **0.934 (0.863-1.000)** | **0.963** | **0.859** | 0.877 | **0.882** | **0.899** | 0.857 | **0.092** |
| Logistic Regression | 0.917 (0.826-0.994) | 0.947 | 0.851 | 0.877 | 0.809 | 0.845 | 0.846 | 0.113 |
| XGBoost (No Hypotension) | 0.887 (0.771-0.980) | 0.901 | 0.812 | 0.864 | 0.765 | 0.814 | 0.825 | 0.131 |
| MASCC + qSOFA | 0.864 (0.767-0.932) | 0.865 | 0.779 | 0.963 | 0.559 | 0.722 | 0.927 | 0.140 |
| qSOFA Alone | 0.838 (0.733-0.918) | 0.840 | 0.710 | 0.864 | 0.559 | 0.700 | 0.776 | 0.152 |
| MASCC Alone | 0.656 (0.546-0.782) | 0.607 | 0.685 | 0.988 | 0.324 | 0.635 | 0.957 | 0.206 |

The XGBoost model achieved the highest AUROC (0.934), significantly outperforming MASCC alone (0.656, p<0.001) and qSOFA alone (0.838, p=0.02). Performance was comparable to logistic regression (0.917, p=0.32), suggesting that the clinical features contain inherently strong signal for this prediction task.

### 3.3 Feature Importance

Feature importance analysis (Figure 1) revealed that hypotension level was the dominant predictor, followed by qSOFA score and metastatic status. The model without hypotension showed decreased but still clinically useful performance (AUROC 0.887).

**Figure 1: Feature Importance (XGBoost Gain)**

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | Hypotension_Level | 0.485 |
| 2 | qSOFA | 0.179 |
| 3 | Mets_Binary | 0.071 |
| 4 | Comorb | 0.058 |
| 5 | Focus_PneumResp | 0.042 |
| 6 | Age_Group | 0.039 |
| 7 | Gender | 0.032 |
| 8 | Neutropenia | 0.028 |
| 9 | Line_Rx | 0.024 |

### 3.4 Calibration

The XGBoost model demonstrated reasonable calibration with a slope of 0.84 and intercept of -0.16. The Brier score was 0.092, indicating good probabilistic accuracy. Calibration curves showed slight overconfidence in the mid-probability range (Figure 2).

### 3.5 Decision Curve Analysis

Decision curve analysis (Figure 3) demonstrated that the XGBoost model provided positive net benefit across threshold probabilities from 10% to 70%, outperforming both the "treat all" and "treat none" strategies as well as MASCC and qSOFA alone.

### 3.6 Subgroup Analysis

**Table 3. Subgroup Performance**

| Subgroup | N | ICU Events | AUROC (95% CI) |
|----------|---|------------|----------------|
| Solid Tumors | 103 | 56 | 0.941 (0.890-0.983) |
| Hematologic | 46 | 25 | 0.941 (0.851-0.998) |
| With Neutropenia | 68 | 34 | 0.941 (0.857-0.997) |
| Without Neutropenia | 81 | 47 | 0.937 (0.882-0.983) |
| Age Group 1 | 20 | 7 | 0.846 (0.475-1.000) |
| Age Group 2 | 71 | 33 | 0.931 (0.845-0.989) |
| Age Group 3 | 58 | 41 | 0.953 (0.878-0.993) |

Model performance appeared consistent across tumor type and neutropenia status. However, these subgroup analyses should be interpreted with caution due to: (1) small sample sizes within subgroups, (2) use of the same out-of-fold predictions as the main model rather than independent validation, and (3) wide confidence intervals in smaller subgroups (e.g., Age Group 1, n=20). Formal interaction testing was not performed.

### 3.7 Threshold Analysis

**Table 4. Performance at Different Probability Thresholds**

| Threshold | Sensitivity | Specificity | PPV | NPV | Youden Index |
|-----------|-------------|-------------|-----|-----|--------------|
| 0.30 | 0.901 | 0.765 | 0.820 | 0.867 | 0.666 |
| 0.40 | 0.889 | 0.853 | 0.878 | 0.866 | 0.742 |
| 0.50 | 0.877 | 0.882 | 0.899 | 0.857 | 0.759 |
| 0.60 | 0.864 | 0.912 | 0.921 | 0.849 | 0.776 |
| 0.70 | 0.840 | 0.926 | 0.932 | 0.829 | 0.766 |

The optimal threshold by Youden index was 0.85, balancing sensitivity and specificity. For clinical applications prioritizing sensitivity (avoiding missed ICU admissions), a lower threshold of 0.30-0.40 may be preferred.

---

## 4. Discussion

### 4.1 Principal Findings

We developed and validated a machine learning model for predicting ICU admission in febrile oncology patients that achieved excellent discriminative ability (AUROC 0.934). The model significantly outperformed established clinical scores including MASCC (AUROC 0.656) and qSOFA (AUROC 0.838). Performance was robust across validation approaches and patient subgroups.

### 4.2 Comparison with Existing Literature

Previous studies have reported variable performance of clinical scoring systems for predicting adverse outcomes in febrile oncology patients. Our finding that MASCC alone has limited ability to predict ICU admission (AUROC 0.656) is consistent with reports that MASCC was designed to identify low-risk patients for outpatient management rather than predict intensive care needs [5].

The qSOFA score showed moderate discrimination (AUROC 0.838), consistent with its validation for identifying sepsis patients at risk of poor outcomes [6]. However, qSOFA was not specifically developed for oncology populations or ICU triage.

Our machine learning approach achieved superior discrimination by integrating multiple clinical features, with hypotension status emerging as the dominant predictor. Even when hypotension was excluded in sensitivity analysis, the model retained clinically useful discrimination (AUROC 0.887).

### 4.3 Clinical Implications

**Role of Hypotension:** The finding that 100% of patients requiring inotropic support (Hypotension Level 2) were admitted to ICU reflects standard clinical practice—patients with refractory hypotension require ICU-level care. This represents a clinical protocol rather than a prediction, but the model still adds value by integrating this information with other risk factors for patients without clear hemodynamic instability.

**Decision Support:** The model may be most useful for patients in the "intermediate risk" category—those without obvious ICU-level decompensation but with elevated risk based on other features. Decision curve analysis demonstrates clinical utility across a range of threshold probabilities.

**Threshold Selection:** The choice of probability threshold should be guided by clinical priorities:
- **High-sensitivity threshold (0.30-0.40):** Minimizes missed ICU admissions at the cost of increased false positives
- **Balanced threshold (0.50-0.60):** Optimizes overall accuracy
- **High-specificity threshold (0.70+):** Reduces unnecessary ICU transfers

### 4.4 Strengths

1. **Rigorous validation:** Repeated stratified cross-validation with bootstrap confidence intervals provides robust performance estimates
2. **Comprehensive comparison:** Head-to-head comparison with established clinical scores
3. **Clinical utility assessment:** Decision curve analysis evaluates real-world applicability
4. **Subgroup consistency:** Stable performance across tumor types and neutropenia status
5. **Reproducibility:** Fixed random seeds and documented methodology

### 4.5 Limitations

1. **Sample size:** With 149 patients, the study has limited power to detect small performance differences and may not capture rare presentations

2. **Single-center data:** The model was developed at a single institution; external validation at independent sites is essential before clinical deployment

3. **Retrospective design:** Prospective validation is needed to confirm real-world performance

4. **Hypotension as predictor:** The strong association between hypotension requiring inotropes and ICU admission may represent clinical protocol rather than truly predictive information. However, sensitivity analysis excluding hypotension still showed useful discrimination (AUROC 0.887)

5. **Unmeasured confounders:** The model does not include laboratory values (lactate, procalcitonin), which may provide additional predictive information

6. **Temporal validation:** True temporal validation with data from a later time period was not performed; the held-out validation represents a simulation of external validation

7. **No cost-effectiveness analysis:** The economic impact of implementing this model was not assessed

### 4.6 Future Directions

1. **External validation:** Multi-center validation is the essential next step
2. **Prospective implementation:** A prospective study evaluating model-guided triage decisions
3. **Integration of laboratory data:** Adding biomarkers may improve performance
4. **Explainability:** Implementation of patient-level explanations using SHAP values
5. **Clinical decision support integration:** Development of bedside tools

---

## 5. Conclusions

A machine learning model incorporating routinely available clinical features can accurately predict ICU admission in oncology patients presenting with febrile illness, significantly outperforming established clinical scores. Hypotension status and qSOFA are the most important predictors. The model demonstrates consistent performance across patient subgroups and shows clinical utility on decision curve analysis. Prospective external validation is required before clinical implementation.

---

## Figures and Tables

### Figures

1. **Figure 1:** Feature importance (XGBoost gain-based)
2. **Figure 2:** ROC curve comparison across models
3. **Figure 3:** Calibration curves
4. **Figure 4:** Decision curve analysis
5. **Figure 5:** Learning curve
6. **Supplementary Figure S1:** SHAP summary plot

### Tables

1. **Table 1:** Baseline patient characteristics
2. **Table 2:** Model performance comparison
3. **Table 3:** Subgroup analysis
4. **Table 4:** Threshold sensitivity analysis
5. **Supplementary Table S1:** Complete feature list and encoding

---

## References

### Primary References

1. Klastersky J, Paesmans M, Rubenstein EB, et al. The Multinational Association for Supportive Care in Cancer risk index: A multinational scoring system for identifying low-risk febrile neutropenic cancer patients. J Clin Oncol. 2000;18(16):3038-3051.

2. Kuderer NM, Dale DC, Crawford J, Cosler LE, Lyman GH. Mortality, morbidity, and cost associated with febrile neutropenia in adult cancer patients. Cancer. 2006;106(10):2258-2266.

3. Young MP, Gooder VJ, McBride K, James B, Fisher ES. Inpatient transfers to the intensive care unit: delays are associated with increased mortality and morbidity. J Gen Intern Med. 2003;18(2):77-83.

4. Teh BW, Harrison SJ, Allison CC, et al. Risk stratification of fever and neutropenia episodes in hematology−oncology patients. Support Care Cancer. 2018;26(11):3725-3732.

5. Klastersky J, Paesmans M. The Multinational Association for Supportive Care in Cancer (MASCC) risk index score: 10 years of use for identifying low-risk febrile neutropenic cancer patients. Support Care Cancer. 2013;21(5):1487-1495.

6. Seymour CW, Liu VX, Iwashyna TJ, et al. Assessment of clinical criteria for sepsis: For the Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). JAMA. 2016;315(8):762-774.

7. Kaukonen KM, Bailey M, Suzuki S, Pilcher D, Bellomo R. Mortality related to severe sepsis and septic shock among critically ill patients in Australia and New Zealand, 2000-2012. JAMA. 2014;311(13):1308-1316.

8. Rajkomar A, Dean J, Kohane I. Machine learning in medicine. N Engl J Med. 2019;380(14):1347-1358.

9. Chen T, Guestrin C. XGBoost: A Scalable Tree Boosting System. In: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. San Francisco, CA: ACM; 2016:785-794.

10. Vickers AJ, Elkin EB. Decision curve analysis: a novel method for evaluating prediction models. Med Decis Making. 2006;26(6):565-574.

### Data Science & Ethics References

11. Collins GS, Reitsma JB, Altman DG, Moons KGM. Transparent Reporting of a multivariable prediction model for Individual Prognosis or Diagnosis (TRIPOD): The TRIPOD Statement. Ann Intern Med. 2015;162(1):55-63.

12. Wiens J, Saria S, Sendak M, et al. Do no harm: a roadmap for responsible machine learning for health care. Nat Med. 2019;25(9):1337-1340.

13. Office for Human Research Protections. Human Subject Regulations Decision Charts. HHS.gov. Available at: https://www.hhs.gov/ohrp/regulations-and-policy/decision-charts/

### Software References

14. Pedregosa F, Varoquaux G, Gramfort A, et al. Scikit-learn: Machine Learning in Python. JMLR. 2011;12:2825-2830.

15. Lundberg SM, Lee SI. A Unified Approach to Interpreting Model Predictions. In: Advances in Neural Information Processing Systems 30. 2017.

---

## Supplementary Materials

### Supplementary Table S1: Feature Encoding

| Original Feature | Encoded Variable(s) | Values |
|-----------------|---------------------|--------|
| Hypotension at arrival (Fluid/Inotrope) | Hypotension_Level | 0=None, 1=Fluid-responsive, 2=Inotrope |
| Type (hematologic vs solid) | Type | 0=Solid, 1=Hematologic |
| Neutropenia | Neutropenia | 0=No, 1=Yes |
| Mets | Mets_Binary, Mets_Missing | Binary + missing indicator |
| Line of Rx | Line_Rx | 1-5 |
| Comorb | Comorb | 0=No, 1=Yes |
| Focus (infection site) | Focus_PneumResp, UTI, Focus_Bloodstream, Focus_GI_Hepatobiliary, Focus_SoftTissue, Focus_NoneUnknown | One-hot encoded |
| Age Group | Age_Group | 1-3 |
| Gender | Gender | 1=Male, 2=Female |
| MASCC | MASCC | 1-2 (categorized) |
| qSOFA | qSOFA | 0-3 |

### Supplementary Table S2: Model Hyperparameters

| Parameter | Value |
|-----------|-------|
| Algorithm | XGBoost (XGBClassifier) |
| n_estimators | 100 |
| max_depth | 3 |
| learning_rate | 0.1 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| min_child_weight | 5 |
| reg_alpha (L1) | 0.1 |
| reg_lambda (L2) | 1.0 |
| random_state | 42 |
| eval_metric | logloss |

---

**Acknowledgments:** [To be added]

**Funding:** [To be added]

**Conflicts of Interest:** The authors declare no conflicts of interest.

**Data Availability:** The de-identified dataset is available upon reasonable request to the corresponding author.

**Code Availability:** The analysis code is available at [GitHub repository to be added].
