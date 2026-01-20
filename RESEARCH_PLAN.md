# Research Plan

## Machine Learning-Based Prediction of ICU Admission in Febrile Oncology Patients

**Primary Author:** Aditya Iyer  
**Principal Investigator:** Dr. Vishwanath Sathyanarayanan  
**Research Mentor:** Dr. Narendhar Gokulanathan  
**Date Prepared:** January 6, 2026  
**IRB Protocol Number:** [To be assigned]

---

## A. RATIONALE

### Background

Febrile illness is one of the most common oncological emergencies, affecting up to 80% of patients receiving chemotherapy for hematologic malignancies and 10-50% of those with solid tumors. While many febrile episodes resolve with prompt antimicrobial therapy, a subset of patients rapidly deteriorate and require intensive care unit (ICU) admission for hemodynamic support, mechanical ventilation, or management of multi-organ dysfunction.

Early identification of patients at high risk for ICU admission is clinically important for several reasons:
1. **Improved outcomes**: Prompt ICU referral in deteriorating patients is associated with improved survival
2. **Resource optimization**: Appropriate triage optimizes utilization in hospitals with limited ICU capacity
3. **Care planning**: Risk stratification enables tailored monitoring intensity and goals-of-care discussions

### Gap in Knowledge

Current clinical scoring systems—including the Multinational Association for Supportive Care in Cancer (MASCC) score and quick Sequential Organ Failure Assessment (qSOFA)—were not specifically designed to predict ICU admission and demonstrate variable performance in this clinical context. Machine learning approaches may capture complex, non-linear relationships among clinical variables and potentially outperform traditional scoring systems.

### Significance and Societal Impact

This research addresses a critical clinical need in oncology practice. Accurate prediction of ICU needs would:
- Enable proactive resource allocation in resource-limited settings
- Support informed shared decision-making between clinicians and patients/families
- Potentially reduce mortality through earlier intervention
- Decrease healthcare costs by optimizing ICU utilization

---

## B. RESEARCH QUESTION(S), HYPOTHESIS(ES), AND EXPECTED OUTCOMES

### Primary Research Question

Can a machine learning model accurately predict ICU admission in oncology patients presenting with febrile illness using clinical variables available at initial presentation?

### Secondary Research Questions

1. Does the machine learning model outperform established clinical scores (MASCC, qSOFA)?
2. Which clinical features are most predictive of ICU admission?
3. Does model performance vary across patient subgroups (tumor type, neutropenia status, age)?

### Hypotheses

**H1 (Primary):** An XGBoost machine learning model incorporating multiple clinical features will achieve an AUROC ≥ 0.80 for predicting ICU admission.

**H2:** The machine learning model will achieve superior discriminative performance (AUROC) compared to MASCC score alone and qSOFA score alone.

**H3:** Model performance will be consistent across patient subgroups defined by tumor type and neutropenia status.

### Expected Outcomes

1. Development of a validated prediction model with good discrimination (AUROC > 0.80)
2. Identification of key clinical features driving ICU admission risk
3. Clinical utility assessment demonstrating positive net benefit across relevant decision thresholds
4. Publication-ready manuscript with model performance metrics and clinical interpretation

---

## C. DETAILED METHODOLOGY

### C.1 List of Materials

#### Data Sources
- **Primary Dataset:** De-identified electronic health record (EHR) data from oncology patients presenting with febrile illness
- **Sample Size:** 149 patients
- **Data File:** `cleaned_data.csv` (stored in secure S3 bucket: `s3://icu-required/cleaned_data.csv`)

#### Software and Computational Tools
| Software | Version | Purpose |
|----------|---------|---------|
| Python | 3.13 | Programming language |
| scikit-learn | 1.5 | Machine learning framework |
| XGBoost | 2.0 | Gradient boosting classifier |
| SHAP | 0.42 | Model interpretability |
| pandas | 2.0+ | Data manipulation |
| matplotlib | 3.8+ | Data visualization |

#### Computing Resources
- Local development machine or AWS SageMaker Processing Jobs
- S3 bucket for data storage

---

### C.2 Procedures

#### Study Design
**Type:** Retrospective cohort study using de-identified data

#### Inclusion Criteria
- Age ≥18 years
- Active malignancy (solid tumor or hematologic)
- Presentation with fever (temperature ≥38.0°C) or clinical suspicion of infection

#### Exclusion Criteria
- Patients with elective/planned ICU admissions
- Incomplete data for primary predictors

#### Variables Collected

**Predictor Variables:**

| Category | Variable | Type | Values |
|----------|----------|------|--------|
| Clinical Scores | MASCC | Categorical | 1-2 |
| Clinical Scores | qSOFA | Ordinal | 0-3 |
| Hemodynamics | Hypotension_Level | Ordinal | 0=None, 1=Fluid-responsive, 2=Inotrope |
| Disease | Type | Binary | 0=Solid, 1=Hematologic |
| Disease | Metastatic status | Binary | 0=No, 1=Yes |
| Disease | Line of therapy | Ordinal | 1-5 |
| Infection | Neutropenia | Binary | 0=No, 1=Yes |
| Infection | Focus | Categorical | 6 categories (one-hot encoded) |
| Demographics | Age_Group | Ordinal | 1-3 |
| Demographics | Gender | Binary | 1=Male, 2=Female |
| Comorbidity | Comorb | Binary | 0=No, 1=Yes |

**Outcome Variable:**
- ICU Requirement (Binary): 1 = ICU admission required, 0 = No ICU admission

#### Data Preprocessing

1. **Data Validation:** Automated checks for missing values, data type consistency, and value range validation
2. **Feature Encoding:** Categorical variables one-hot encoded; ordinal variables preserved as numeric
3. **Missing Data:** No imputation required (complete cases only)
4. **No Normalization:** Tree-based methods are invariant to monotonic transformations

#### Model Development

**Primary Model: XGBoost Classifier**

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| n_estimators | 100 | Balance between performance and overfitting |
| max_depth | 3 | Conservative for small dataset |
| learning_rate | 0.1 | Standard starting point |
| subsample | 0.8 | Reduce overfitting |
| colsample_bytree | 0.8 | Reduce overfitting |
| min_child_weight | 5 | Prevent fitting to noise |
| reg_alpha (L1) | 0.1 | Feature selection |
| reg_lambda (L2) | 1.0 | Weight regularization |
| random_state | 42 | Reproducibility |

**Comparison Models:**
1. Logistic Regression with L2 regularization
2. MASCC score alone
3. qSOFA score alone
4. Combined MASCC + qSOFA
5. XGBoost without hypotension (sensitivity analysis)

#### Validation Strategy

**Internal Validation:**
- 10-repeat × 5-fold stratified cross-validation (50 total iterations)
- Maintains class balance in each fold
- All preprocessing within folds to prevent data leakage

**Simulated External Validation:**
- Stratified 70/30 train-test split
- Training on development set; evaluation on held-out set

**Bootstrap Confidence Intervals:**
- 1000 bootstrap resamples for all performance metrics

#### Responsibilities

| Task | Responsible Party |
|------|-------------------|
| Study design | Principal Investigator |
| Data extraction | [To be specified] |
| De-identification | [To be specified] |
| Pipeline development | Research team |
| Model training | Research team |
| Statistical analysis | Research team |
| Manuscript preparation | Principal Investigator |

---

### C.3 Risk and Safety

#### Data Security Risks and Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| Re-identification of patients | High | Use of fully de-identified data; no direct identifiers; limited dataset elements |
| Unauthorized data access | Medium | Data stored in encrypted S3 bucket with IAM access controls |
| Data loss | Low | Version control; automated backups |

#### Research Integrity Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Overfitting | Medium | Rigorous cross-validation; regularization; conservative hyperparameters |
| P-hacking | Low | Pre-specified analysis plan; primary/secondary outcomes defined a priori |
| Irreproducibility | Low | Fixed random seeds; version-controlled code; documented methodology |

#### Clinical Implementation Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Misapplication of model | High | Clear disclaimer that model requires prospective validation before clinical use |
| Algorithm bias | Medium | Subgroup analysis to assess fairness across populations |

#### No Physical Risks
This study uses only de-identified retrospective data with no direct patient contact, interventions, or biological specimen collection.

---

### C.4 Data Analysis

#### Primary Performance Metrics

| Metric | Description |
|--------|-------------|
| AUROC | Area under ROC curve (discrimination) |
| AUPRC | Area under precision-recall curve |
| Accuracy | Overall classification accuracy |
| Sensitivity | True positive rate |
| Specificity | True negative rate |
| PPV/NPV | Positive/negative predictive values |
| Brier Score | Calibration metric |

#### Secondary Analyses

1. **Feature Importance Analysis:**
   - XGBoost gain-based importance
   - Permutation importance
   - SHAP (SHapley Additive exPlanations) values

2. **Calibration Assessment:**
   - Calibration slope and intercept
   - Visual calibration curves

3. **Decision Curve Analysis:**
   - Net benefit across threshold probabilities
   - Comparison with "treat all" and "treat none" strategies

4. **Subgroup Analysis:**
   - Performance by tumor type (solid vs. hematologic)
   - Performance by neutropenia status
   - Performance by age group

5. **Threshold Optimization:**
   - Youden index optimization
   - Sensitivity analysis at multiple thresholds

6. **Sensitivity Analysis:**
   - Model performance excluding hypotension (to assess non-deterministic predictors)

#### Statistical Software
All analyses performed using Python 3.13 with scikit-learn, XGBoost, and SHAP libraries.

---

## D. HUMAN PARTICIPANTS RESEARCH (SECONDARY DATA ANALYSIS)

> **Note:** This study uses a **de-identified dataset** for secondary data analysis. Per institutional guidelines and federal regulations (45 CFR 46.104(d)(4)), research involving only de-identified data typically does not meet the definition of human subjects research. However, we provide the following information for completeness and IRB determination.

### D.1 Description of Dataset and Source

**Dataset Characteristics:**
| Characteristic | Value |
|----------------|-------|
| Total Records | 149 patients |
| Age Range | Adults ≥18 years (exact ages not included; only age groups) |
| Gender Distribution | Included as encoded variable |
| Racial/Ethnic Composition | Not collected |
| Vulnerable Populations | None identified in dataset |

**Data Source:**
- Electronic health records from oncology patients presenting with febrile illness
- Data extracted by authorized personnel at originating institution
- De-identification performed prior to analysis

### D.2 De-identification Method

The dataset was de-identified in accordance with HIPAA Safe Harbor method (45 CFR 164.514(b)(2)) or Expert Determination method. The following identifiers were removed or modified:

| Identifier | Status |
|------------|--------|
| Names | Removed |
| Geographic data | Removed |
| Dates (except year) | Removed/generalized |
| Phone/fax numbers | Removed |
| Email addresses | Removed |
| SSN | Removed |
| Medical record numbers | Removed |
| Health plan numbers | Removed |
| Account numbers | Removed |
| Certificate numbers | Removed |
| Device identifiers | Removed |
| URLs/IP addresses | Removed |
| Biometric identifiers | Not applicable |
| Full-face photographs | Not applicable |
| Ages over 89 | Generalized to category |

### D.3 Recruitment

**Not applicable.** This study uses only existing de-identified data. No participant recruitment is required.

### D.4 Methods

**Not applicable.** No direct participant interaction. Analysis is performed on pre-existing, de-identified clinical data.

### D.5 Risk Assessment for De-identified Data

| Risk Type | Assessment |
|-----------|------------|
| Physical risks | None (no intervention) |
| Psychological risks | None (no contact) |
| Social risks | None (data de-identified) |
| Legal risks | None (data de-identified) |
| Economic risks | None |
| Re-identification risk | Minimal (HIPAA-compliant de-identification) |

**Benefits:**
- No direct benefit to original patients
- Potential societal benefit through improved clinical decision support tools

### D.6 Protection of Privacy

**Data Handling:**
| Aspect | Procedure |
|--------|-----------|
| Identifiable information collected | None |
| Data status | Fully de-identified prior to analysis |
| Storage location | Encrypted S3 bucket with access controls |
| Access controls | Limited to research team members |
| Retention | Data will be archived after study completion |
| Destruction | Per institutional data retention policy |

### D.7 Informed Consent Process

**Not applicable.** Per 45 CFR 46.116(c)(4), consent is not required for research involving only de-identified data or data/specimens that are publicly available.

**Justification:**
1. The research involves no more than minimal risk
2. The research could not practicably be carried out without the waiver
3. The waiver will not adversely affect the rights and welfare of subjects
4. The research involves only information collection and analysis

---

## E. BIBLIOGRAPHY

### Primary References

1. Klastersky J, Paesmans M, Rubenstein EB, et al. The Multinational Association for Supportive Care in Cancer risk index: A multinational scoring system for identifying low-risk febrile neutropenic cancer patients. *J Clin Oncol*. 2000;18(16):3038-3051.

2. Kuderer NM, Dale DC, Crawford J, Cosler LE, Lyman GH. Mortality, morbidity, and cost associated with febrile neutropenia in adult cancer patients. *Cancer*. 2006;106(10):2258-2266.

3. Young MP, Gooder VJ, McBride K, James B, Fisher ES. Inpatient transfers to the intensive care unit: delays are associated with increased mortality and morbidity. *J Gen Intern Med*. 2003;18(2):77-83.

4. Teh BW, Harrison SJ, Allison CC, et al. Risk stratification of fever and neutropenia episodes in hematology−oncology patients. *Support Care Cancer*. 2018;26(11):3725-3732.

5. Klastersky J, Paesmans M. The Multinational Association for Supportive Care in Cancer (MASCC) risk index score: 10 years of use for identifying low-risk febrile neutropenic cancer patients. *Support Care Cancer*. 2013;21(5):1487-1495.

6. Seymour CW, Liu VX, Iwashyna TJ, et al. Assessment of clinical criteria for sepsis: For the Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). *JAMA*. 2016;315(8):762-774.

7. Kaukonen KM, Bailey M, Suzuki S, Pilcher D, Bellomo R. Mortality related to severe sepsis and septic shock among critically ill patients in Australia and New Zealand, 2000-2012. *JAMA*. 2014;311(13):1308-1316.

8. Rajkomar A, Dean J, Kohane I. Machine learning in medicine. *N Engl J Med*. 2019;380(14):1347-1358.

9. Chen T, Guestrin C. XGBoost: A Scalable Tree Boosting System. In: *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*. San Francisco, CA: ACM; 2016:785-794.

10. Vickers AJ, Elkin EB. Decision curve analysis: a novel method for evaluating prediction models. *Med Decis Making*. 2006;26(6):565-574.

### Data Science and Ethics References

11. Collins GS, Reitsma JB, Altman DG, Moons KGM. Transparent Reporting of a multivariable prediction model for Individual Prognosis or Diagnosis (TRIPOD): The TRIPOD Statement. *Ann Intern Med*. 2015;162(1):55-63.

12. Wiens J, Saria S, Sendak M, et al. Do no harm: a roadmap for responsible machine learning for health care. *Nat Med*. 2019;25(9):1337-1340.

13. Office for Human Research Protections. Human Subject Regulations Decision Charts. *HHS.gov*. Available at: https://www.hhs.gov/ohrp/regulations-and-policy/decision-charts/index.html

### Software References

14. Pedregosa F, Varoquaux G, Gramfort A, et al. Scikit-learn: Machine Learning in Python. *JMLR*. 2011;12:2825-2830.

15. Lundberg SM, Lee SI. A Unified Approach to Interpreting Model Predictions. In: *Advances in Neural Information Processing Systems 30*. 2017.

---

## F. ADDENDUM / PROJECT SUMMARY

*This section is reserved for documenting any changes made during the research process.*

### Changes from Original Plan

| Date | Description of Change | Impact | Approval Required? |
|------|----------------------|--------|-------------------|
| — | No changes to date | — | — |

---

## G. SIGNATURES

**Principal Investigator:**

Signature: _________________________ Date: _____________

Print Name: _________________________

**Research Mentor (if applicable):**

Signature: _________________________ Date: _____________

Print Name: _________________________

---

## Document Version Control

| Version | Date | Author | Description |
|---------|------|--------|-------------|
| 1.0 | 2026-01-06 | [Author] | Initial research plan |

---

*This research plan was created in accordance with ISEF/SRC Research Plan requirements and institutional IRB guidelines for secondary data analysis using de-identified data.*
