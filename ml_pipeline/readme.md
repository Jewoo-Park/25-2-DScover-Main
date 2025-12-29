# 🧪 `ML_pipeline.py` — Layer-wise Hallucination Detection Pipeline

This script implements the **full machine-learning evaluation pipeline** used to test whether
LLM hidden-state representations contain predictive signals for hallucination detection.

It loads the **layer-wise hidden-representation CSV files** (see `/data`), runs multiple
lightweight ML classifiers with cross-validation, optionally applies PCA, and reports
**Accuracy / AUROC / Feature Importance** per layer and model.

---

## ✅ Main Objectives

- Compare **First / Middle / Last** layer representations  
- Evaluate **Logistic Regression, Linear SVM, Random Forest, XGBoost (if enabled)**  
- Measure performance using **Stratified-KFold CV**
- Optionally apply:
  - PCA → fixed 10-D
  - PCA → retain 95% variance
- Export experiment results to CSV
