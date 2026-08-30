# Loan-Default-Risk-Prediction-AI-Explainability-Dashboard
## Overview
Not only does the loan provider need a risk score but they must provide an explanation as to why a certain loan is considered risky, either for their own records or because of increasing demands for accountability when using explainable AI for lending. The model predicts the likelihood of loan default and explains the reasoning behind it by providing an explainability pipeline (SHAP + LLM), which converts complex features into English-language explanations of a nature that a lending platform would expect to provide to a loan officer.

The project is based on the open-source loan data from Lending Club (peer-to-peer lending).
## Business Problem
Based on the information provided by a loan application, including income, credit score, debt level, and loan profile, we:

1. Determine whether the loan is likely to default versus being repaid in full.

2. Provide a Explanation for why a particular application has been classified as high-risk.
## Dataset
**Source**: Public dataset from the Lending Club.

**Size**:396,000 observations after cleaning.

**Target variable**: loan_status – encoded as 0 for “Fully Paid” and 1 for “Charged Off”.

**Classes**: 80% “Fully Paid”, 20% “Charged Off” (unbalanced classes).

   
