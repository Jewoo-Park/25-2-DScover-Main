# Hallucination Detection via Internal Layer Representations  
### An Empirical Study on LLM Interpretability using Machine Learning Probing

This repository explores the interpretability of Large Language Models (LLMs) by leveraging their **internal hidden representations** as features for **machine learning (ML) classifiers** to predict and analyze hallucinations **before text generation is complete**.

---

## 📖 Overview

While LLMs have demonstrated remarkable performance across diverse NLP tasks, **hallucination** remains a critical challenge. Existing detection approaches—such as evaluator LLMs or heuristic filtering—often suffer from:

- High inference cost  
- Strong dependency on specific models  
- Limited interpretability  

This work investigates whether **LLM internal hidden states already encode signals related to hallucination risk**, and evaluates how effectively lightweight ML classifiers can detect such signals.

---

## 🛠 Methodology

### 1. Pipeline

**Feature Extraction**  
Hidden representations are extracted from selected transformer layers of pre-trained LLMs.  
Typical feature dimension:  
- *e.g.,* `d = 4096` for 7B-class models (LLaMA, Mistral, etc.)

**Layer Selection**  
To analyze the information flow, the following layers are compared:

- First Layer  
- Middle Layer  
- Last Layer  

**ML Classification**  
Extracted representations are fed into lightweight ML models for **binary hallucination detection**.

---

### 2. Experimental Setup

**Backbone LLMs**
- Mistral-7B  
- Qwen2-7B  
- LLaMA  
- Falcon  

**Dataset**
- **HaluEval (EMNLP 2023)**  
- ~35,000 samples across QA & Summarization tasks  

**Machine Learning Models**
- Logistic Regression (L1 / L2)
- Linear SVM
- Random Forest
- XGBoost

**Dimensionality Reduction**
- PCA variants:
  - None
  - Fixed 10-D
  - 95% Variance

---

## 📊 Key Results

### 🏆 The Middle-Layer Advantage

Across all models, **middle-layer representations** provide the strongest predictive signals for hallucination detection.

| LLM Backbone | Best Layer | Best ML Model | Best AUROC |
|-------------|-----------|---------------|-----------:|
| LLaMA       | Middle    | Logistic (L1) | 0.830 |
| Qwen2       | Middle    | Logistic (L1) | 0.806 |
| Mistral     | Middle    | Logistic (L1) | 0.807 |
| Falcon      | Middle    | Logistic (L1) | 0.785 |

---

## 📈 Major Findings

- **Layer-wise Consistency**  
  Middle layers achieve **~0.80 AUROC**, outperforming first & last layers (**0.72–0.76** range).

- **Linear Models Perform Best**  
  Logistic Regression & Linear SVMs show strong stability in high-dimensional feature spaces.

- **PCA Reduces Performance**  
  Compressing to 10-D significantly degrades AUROC, suggesting hallucination signals are  
  **not confined to a simple low-dimensional linear space**.

---

## 💡 Key Insights

- **Internal Awareness**  
  LLMs encode hallucination risk signals *before* generating output.

- **Efficient Detection**  
  Detection is possible using **lightweight ML classifiers** without running another LLM.

- **Interpretability & Cost Efficiency**  
  Re-using hidden states avoids extra inference cost and improves transparency.

---

## ⚠️ Limitations & Future Work

- **Dataset Dependency**  
  Results are based on HaluEval (factual consistency).  
  Other hallucination types (reasoning, numeric errors) require further study.

- **Static Evaluation**  
  Current approach extracts post-hoc features rather than real-time detection.

- **Model Optimization**  
  Future directions include:
  - Non-linear classifiers  
  - Multi-layer fusion  
  - Real-time hallucination control  

---

## 📬 Contact & Contributions

Contributions, discussions, and issues are welcome.  
Please feel free to submit PRs or open GitHub Issues.
