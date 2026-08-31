# Loan-Default-Risk-Prediction-AI-Explainability-Dashboard
## Overview
Not only does the loan provider need a risk score but they must provide an explanation as to why a certain loan is considered risky, either for their own records or because of increasing demands for accountability when using explainable AI for lending. The model predicts the likelihood of loan default and explains the reasoning behind it by providing an explainability pipeline (SHAP + LLM), which converts complex features into English-language explanations of a nature that a lending platform would expect to provide to a loan officer.

The project is based on the open-source loan data from Lending Club (peer-to-peer lending).
## Business Problem
Based on the information provided by a loan application, including income, credit score, debt level, and loan profile, we:

1. Determine whether the loan is likely to default versus being repaid in full.

2. Provide a Explanation for why a particular application has been classified as high-risk.
## Dataset
- **Source**: Public dataset from the Lending Club.

- **Size**: 396,000 observations after cleaning.

- **Target variable**: loan_status – encoded as 0 for “Fully Paid” and 1 for “Charged Off”.

- **Classes**: 80% “Fully Paid”, 20% “Charged Off” (unbalanced classes).
## Feature Engineering
- **Dropped high-noise/leakage-prone columns**: `emp_title`, `title`, `address` (high cardinality free text), `int_rate`, `grade`, `sub_grade` (they already contain Lending Club's own risk assessment – their presence in the features would allow for cheating, since we want to teach the model to predict the raw risk based on borrower's factors rather than Lending Club's own pre-calculated one).   

- **Rare categories merging**: Low-frequency categories in `purpose`, `application_type`, `home_ownership` (< 200–400 rows each) were merged into `other` category bucket to avoid the estimation instability in the model due to low sample size of these categories.

- **emp_length**: Converted from text ("10+ years", "< 1 year") to a numeric scale; missing values filled with the median.

- **mortage_account**: (≈10% missing) Instead of the plain median imputation, `mort_acc` was filled using the average `mort_acc` among loans with the same `total_acc` (the number of mortgage accounts has correlation with the total number of accounts), with a fallback median imputation for the cases where `total_acc` group had no valid values to average.

- **revolving_utilization (missing)**: Compared with revolving_balance prior to imputation, because most null values coincided with zero value of revolving balance ( filled with 0), while a small number of high-balances were filled with the median, since the zero utilization assumption could not be true in this case.

- **public_record_bankruptcies**: Filled with 0, as this is a count feature and null values mean "no bankruptcies recorded", confirmed by comparison with public_record feature.

- **earliest_credit_line**: Converted (along with `issue_date`) to `credit_history_years` derived feature, then raw dates were dropped.

- **encoding**: Applied to the rest of the categorical features (`home_ownership`, `purpose`, `application_type`, `verification_status`) with drop_first=True to avoid dummy variable trap.
## Modeling
### Baseline: Logistic Regression
Scaled features, class_weight='balanced' to address class imbalance.

| Metric | Class 0 (Fully Paid) | Class 1 (Charged Off) |
|---|---|---|
| Precision | 0.87 | 0.31 |
| Recall | 0.67 | **0.60** |

**AUC: 0.69**
### Improved model: XGBoost
At the default 0.5 threshold, XGBoost achieved higher accuracy (0.81) but recall on defaulters collapsed to 0.09 — the model was effectively defaulting to "predict everyone as safe," which looks good on accuracy but fails at the actual goal.

**Threshold tuning** (adjusting the decision cutoff instead of using the default 0.5) recovered a usable trade-off:

| Metric | Logistic Regression | XGBoost (0.5 threshold) | XGBoost (tuned threshold) |
|---|---|---|---|
| Recall (Charged Off) | 0.60 | 0.09 | 0.38 |
| Precision (Charged Off) | 0.31 | 0.55 | 0.39 |
| AUC | 0.69 | 0.71 | 0.71 |

 **Note**: Logistic regression catches more actual defaulters (higher recall) at the cost of more false alarms, tuned XGBoost model offers a more balanced precision/recall trade-off 
## Explainability: SHAP
Using `TreeExplainer` on the tuned XGBoost model:

- **Top risk drivers:** `installment`, `term`, and `loan_amnt` — loan *structuring* features — ranked as highly influential as, or more than, borrower financial profile features like `annual_inc`, `revol_util`, and `dti`(debt to income ratio).
- **pattern:** `loan_amnt` on its own tended to *reduce* predicted risk, while `installment` and `term` increased it — suggesting the model learned that, for a given loan size, a longer term (and its associated smaller monthly payment) still carries more risk, likely because of extended repayment exposure over time.
- Per-applicant SHAP breakdowns were generated to show exactly which factors pushed an individual application's risk up or down, and by how much.
## Explainability: LLM-Generated Plain-English Explanations

For each high-risk applicant, the top 3 SHAP-driving factors were extracted and passed to an LLM to generate a short, plain-English explanation of the risk assessment — intended to read like something a loan officer could act on directly.

**Example output** (applicant with dti = 44.22%, missing employment length, installment = $536.81):
> "The borrower's debt-to-income ratio of 44.22% flags elevated risk, showing that a sizable share of their earnings is already tied up in debt payments. Moreover, the missing employment-length indicator combined with a monthly installment of $536.81 further heightens the risk assessment."

Of 10,278 applicants flagged as high-risk by the tuned model, a representative sample of 100 — spanning both high-confidence and borderline risk cases — was processed through the explanation pipeline and cached for the dashboard.
## Dashboard (Power BI)

The model outputs, SHAP values, and LLM explanations are exported from Python to CSV and visualized in a single-page Power BI dashboard, including:

- Headline KPIs (total loans, default rate, avg loan amount, avg DTI)
- Loan outcome distribution and default rate by DTI bracket, loan term, and income verification status
- A model comparison table (Logistic Regression vs. XGBoost vs. tuned XGBoost)
- Global SHAP feature importance
- An interactive **AI Risk Explanation** section — pick any high-risk applicant from a dropdown and see their plain-English risk explanation alongside a chart of the underlying SHAP factors.
## Tech Stack

- **Data & modeling:** Python, pandas, scikit-learn, XGBoost
- **Explainability:** SHAP
- **LLM explanation layer:** Groq API (`openai/gpt-oss-120b`)
- **Dashboard:** Power BI
  
