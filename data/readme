# 📂 Data: Layer-wise Hidden Representations

This folder contains **layer-wise hidden representations** extracted from multiple Large Language Models (LLMs). These vectors are used as feature inputs for downstream hallucination detection experiments.

All CSV files can be downloaded here:

👉 https://drive.google.com/drive/folders/1LFBDDrGE-kNu4oixwUbvalRjAyag6ydy?usp=sharing

---

## 📑 Description

Each CSV file corresponds to hidden representations extracted from a specific layer location (*First / Middle / Last*) of a given LLM.

Token-level hidden states were **mean-pooled** to obtain a single fixed-length feature vector per sample. These vectors are then used as inputs to lightweight ML classifiers (e.g., Logistic Regression, SVM).

---

## 🧠 Included Language Models

- LLaMA (7B class)
- Qwen2 (7B class)
- Mistral (7B class)
- Falcon

---

## 🏗 CSV Schema

Each CSV file contains the following columns:

1. **model**  
   - Name of the language model  
   - e.g., `mistral-7b`, `qwen2-7b`

2. **layer_position**  
   - Layer where the representation was extracted  
   - Possible values:
     - `first`
     - `middle`
     - `last`

3. **dim_1 … dim_N**  
   - Hidden representation feature vector  
   - `N` = hidden dimension size (e.g., 4096)  
   - Example columns:
     - `dim_1`
     - `dim_2`
     - …
     - `dim_4096`

---

## 📌 Example

| model       | layer_position | dim_1   | dim_2   | ... | dim_4096 |
|------------|----------------|--------:|--------:|-----|---------:|
| mistral-7b | middle         | 0.0241  | -0.0182 | ... | 0.1129   |

---

## 🎯 Purpose

This dataset enables:

- Analysis of whether internal LLM states encode **hallucination risk**
- Comparison of information content across different layers
- Training lightweight ML-based hallucination detectors

---

## 📥 Usage Example (Python)

```python
import pandas as pd

df = pd.read_csv("mistral_middle.csv")

X = df[[c for c in df.columns if c.startswith("dim_")]]
y = ...  # hallucination labels

