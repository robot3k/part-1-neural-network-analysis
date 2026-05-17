# Part 1: Neural Network Fundamentals and Training Behavior Analysis

## Overview

This project builds and analyzes a feed-forward neural network to predict customer churn using a structured telecom dataset. The focus is not just on accuracy but on deeply understanding **how neural networks learn** — forward pass, backpropagation, loss minimization, and parameter updates.

## Dataset

**File:** `customer_churn_nn.csv`  
**Source:** Part 1 dataset (Masai Module 5 Assignment)  
**Records:** 2,000 customer records | **Features:** 15 (after dropping ID) | **Target:** `churn` (binary)

**Key observations:**
- Severe class imbalance: only **31 churners (1.55%)** out of 2,000 customers
- No missing values
- Mix of categorical (region, plan_type, contract_type, payment_method) and numerical features

## Approach

### Preprocessing
- Dropped `customer_id` (non-predictive identifier)
- Label-encoded all categorical columns
- Applied `StandardScaler` to numerical features
- Stratified 80/20 train-test split to preserve churn proportion
- Used **class weights** (`{0: 0.51, 1: 32.0}`) to handle imbalance — the model penalizes missing a churner 32× more than misclassifying a retained customer

### Model Architecture (Baseline)
```
Input (15 features)
    → Dense(64, ReLU)
    → Dropout(0.3)
    → Dense(32, ReLU)
    → Dropout(0.3)
    → Dense(1, Sigmoid)
```
- **Loss:** Binary Cross-Entropy
- **Optimizer:** Adam (lr=0.001)
- **Epochs:** 50 | **Batch size:** 32

## Results

| Configuration | Test Accuracy | Precision (Churn) | Recall (Churn) | F1 (Churn) | AUC-ROC |
|---------------|--------------|-------------------|----------------|------------|---------|
| Baseline (64-32, lr=0.001, bs=32) | 0.9375 | 0.1875 | 0.5000 | 0.2727 | 0.8638 |
| Deeper (128-64-32, lr=0.001, bs=32) | 0.9125 | 0.1250 | 0.3333 | 0.1818 | 0.8629 |
| High LR (64-32, lr=0.01, bs=64) | 0.9300 | 0.2222 | 0.3333 | 0.2667 | 0.8316 |

> **Note:** Raw accuracy is misleading here due to class imbalance. **AUC-ROC (0.86)** is the correct primary metric — it measures how well the model ranks churners above non-churners regardless of threshold.

## Key Learnings

1. **Weights and biases** are the learnable parameters — weights scale feature importance, biases shift activation thresholds
2. **ReLU activation** is essential for non-linear decision boundaries — without it, stacking layers is mathematically equivalent to a single linear transformation
3. **Learning rate 0.001 vs 0.01:** The higher learning rate (0.01) caused unstable loss curves and lower AUC, confirming that overshooting the gradient minimum hurts generalization
4. **Overfitting signal:** By epoch 50, training accuracy exceeded validation accuracy by ~4%, suggesting the model began memorizing training patterns — Dropout at 0.3 partially mitigated this

## Repository Structure

```
part-1-neural-network-analysis/
├── README.md
├── notebook.ipynb
├── requirements.txt
└── results/
    ├── model_comparison_table.csv
    └── evaluation_outputs.png
```

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

Place `customer_churn_nn.csv` in the same directory as the notebook before running.
