# Cancer Risk Triage | Baseline Model

A machine learning baseline for a rural cervical cancer risk triage tool, built as a methods dry-run to validate the pipeline before sourcing a more suitable dataset.

## Intended goal

The goal of this project is a triage mechanism that could flag patients for follow-up screening based on risk factors, aimed at a rural India context where access to diagnostic testing (colposcopy, biopsy) is limited. This session focuses on building and validating the modeling pipeline on a real, biopsy-confirmed dataset before adapting it to India-specific data.

## Dataset

- **Source:** UCI Cervical Cancer Risk Factors dataset 
- **Size:** 858 patients, with real biopsy-confirmed outcomes
- **Note on fit:** This is not Indian data. It is used here as a real-data proxy to validate the triage pipeline mechanism, not as a production dataset. Two India-specific alternatives were considered and ruled out:
  - NFHS-5 / ICMR data — no patient-level outcome variable available
  - A synthetic India-labeled Kaggle dataset — data provenance unclear
- **Conclusion:** After completing this baseline, the dataset was found to be too limited (see Limitations) to support the intended triage use case. Future work will source a dataset with a larger number of positive cases.

## Data Cleaning

- Missing values were encoded as `'?'` in the raw CSV and loaded using `na_values='?'`.
- Dropped for excessive sparsity (>90% missing):
  - `STDs: Time since first diagnosis`
  - `STDs: Time since last diagnosis`
- Dropped for data leakage (these are diagnostic test results, not pre-screening risk factors):
  - `Hinselmann`, `Schiller`, `Citology`, `Dx:Cancer`, `Dx:CIN`, `Dx:HPV`, `Dx`
- Imputation:
  - Numeric columns (age, years, counts) — median
  - Binary 0/1 columns (STD indicators, contraceptive/IUD flags) mode
- **Target variable:** `Biopsy` (ground-truth diagnostic result)

## Class Imbalance

The dataset has a severe class imbalance: 55 positive cases out of 858 total (6.4% positive rate). Accuracy is a misleading metric here — a model that always predicts "negative" achieves ~94% accuracy while being clinically useless.

To address this:
- Used a **stratified train/test split** (80/20, `random_state=42`) to preserve the 6.4% ratio in both sets, confirmed at 161/11 in the test set.
- Used `class_weight='balanced'` in both models tried.
- Evaluated models on **recall and precision at multiple thresholds**, not accuracy.

## Models Tried

### Logistic Regression (`class_weight='balanced'`, `max_iter=1000`)

| Threshold | Recall | Precision |
|-----------|--------|-----------|
| 0.5       | 0.36   | 0.09      |
| 0.4       | 0.55   | 0.06      |
| 0.3       | 1.00   | (drops further — flags nearly everyone) |

### Random Forest (`class_weight='balanced'`)

| Threshold | Recall | Precision | Accuracy |
|-----------|--------|-----------|----------|
| 0.5 (default) | 0.09 | 0.33 | 93% |
| 0.2       | 0.27   | 0.20      | —        |

The Random Forest's default-threshold result is a textbook accuracy trap: 93% accuracy while catching almost none of the real positive cases.

### Model Selection

**Logistic Regression at threshold 0.4** was selected as the working baseline model. This prioritizes recall (catching real cases) over precision, which is the appropriate tradeoff for a triage tool: a missed case is costly, while a false positive just results in an additional screening visit. This is documented as an explicit, justified choice for the triage use case — not the model with the best-looking aggregate metrics.

## Limitation: Coefficient Instability

The baseline logistic regression produced a **negative coefficient for `STDs:HPV`**, which contradicts well-established medical knowledge — HPV is a primary cause of cervical cancer.

**Root cause:** With only ~55 positive cases across 26 features (~2 events per feature, well below the general rule-of-thumb of ~10 events per feature), coefficient estimates are inherently unstable. Several sub-features are also correlated with one another (e.g., the condylomatosis subtypes), which further destabilizes individual coefficient estimates.

### Follow-up: L1-Regularized Logistic Regression

To test whether this instability was an artifact of insufficient regularization (rather than a genuine data limitation), L1-regularized logistic regression (`penalty='l1'`, `solver='liblinear'`) was run at three regularization strengths:

- **C = 1.0** (weak penalty): Produced implausibly large coefficients for rare binary features (`STDs:syphilis`: -2.29, `STDs:HIV`: +1.64) — a signature of overfitting to a handful of data points rather than recovering a real signal.
- **C = 0.1**: Most STD-related features collapsed to zero; `STDs (number)` and a few contraceptive/IUD features remained.
- **C = 0.01** (strong penalty): Nearly all features collapsed to zero, leaving only 3–4 non-clinical features.

Critically, **`STDs:HPV` was zeroed out at every regularization strength tested**, including the weakest (C = 1.0). The model could not find enough signal from this feature to assign it any weight at all, at any level of regularization.

**Conclusion:** This is not a hyperparameter problem. Regularization did not recover a clinically plausible signal for HPV, and instead confirmed the sample-size limitation: with ~55 positive cases spread across 26 features, several features (including clinically important ones) cannot be stably estimated by this model on this dataset. This is documented as a genuine limitation of the dataset for this modeling approach, not a real clinical finding about HPV's risk contribution.

## Saved Artifact

- `logistic_regression_v1.pkl` (saved via `joblib.dump`)

## Status and Next Steps

This baseline pipeline (cleaning → imputation → stratified split → threshold-tuned logistic regression) is considered validated and complete for this dataset. Given the coefficient instability findings above, this dataset will not be carried forward into the triage tool itself.

**Next steps:**
1. Source a larger dataset with more positive cases per feature — ideally India-specific or with sufficient patient-level detail to support stable coefficient estimation.
2. Re-run this same pipeline (cleaning, imputation, stratified split, threshold-tuned logistic regression, L1 check) on the new dataset.
3. Once a stable model is achieved, build a Streamlit wrapper: patient data entry → risk score (Low / Medium / High) + top contributing factors.
