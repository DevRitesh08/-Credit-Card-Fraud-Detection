# Credit Card Fraud Detection System

A production-grade classification system designed to detect fraudulent credit card transactions in highly imbalanced datasets. The system is deployed as a live interactive dashboard integrated directly into a React/Vite portfolio.

**[Live Interactive Dashboard](https://riteshswami.tech/projects/creditcardfraud)**

## Technical Overview

Credit card fraud datasets present a notorious challenge in machine learning: severe class imbalance. In this dataset, consisting of 284,807 transactions, only 0.17% are fraudulent. A naive algorithm will optimize for overall accuracy by predicting the majority class exclusively, leading to model collapse.

This project tackles the imbalance using advanced ensemble techniques, cost-based threshold optimization, and deep learning architectures to force algorithms to rigorously learn the latent features of fraudulent activity rather than relying on statistical priors.

### 1. Ensemble Benchmarking & High-Performance Boosting

The research phase (`showcase/Complete-Fraud-Detection.ipynb`) rigorously benchmarks four distinct machine learning paradigms:
- **Supervised Boosting (XGBoost):** The top-performing model. By leveraging sequential tree building and a mathematically calculated `scale_pos_weight`, the model is forced to severely penalize missed fraud cases, yielding an exceptional F1 score on sparse data.
- **Supervised Bagging (Random Forest + SMOTE):** Utilizes Synthetic Minority Over-sampling Technique to synthetically balance the dataset prior to training ensemble trees.
- **Unsupervised Anomaly Detection (Isolation Forest):** Operates without labels to isolate systemic outliers in the feature space.
- **Deep Learning (PyTorch DNN):** A deep neural network optimized with `BCEWithLogitsLoss` and dynamic positive class weighting (`pos_weight = 578.0`).

### 2. Cost-Based Threshold Optimization
In enterprise fraud detection, a False Negative (missed fraud) costs a financial institution significantly more than a False Positive (false alarm). 

The system implements a custom cost function that dynamically searches for the optimal probability threshold to minimize the total financial cost. For our top-performing XGBoost model, the threshold was mathematically optimized downward to `0.10` to drastically reduce expensive False Negatives.

### 3. Decision Bands & Analyst Review Queue
Unlike standard binary classifiers, this system models real-world enterprise logic by implementing customizable **Decision Bands**. Instead of a simple Approve/Block dichotomy, the system yields three routing states:

1. **Approve:** `P(fraud) < 0.25` - Transaction clears automatically.
2. **Manual Review:** `0.25 <= P(fraud) < 0.70` - Transaction flagged and routed to a human analyst queue.
3. **Block:** `P(fraud) >= 0.70` - Transaction blocked instantly.

The backend generates a structured Review Queue payload to optimize the workload of human fraud analysts, automatically classifying historical outcomes.

### 4. Data Privacy
Feature vectors (V1-V28) are inherently anonymized via Principal Component Analysis (PCA) prior to modeling, ensuring no Personally Identifiable Information (PII) is exposed while retaining statistical variance.

## Project Structure

- `showcase/Complete-Fraud-Detection.ipynb`: Comprehensive research notebook benchmarking XGBoost, Random Forest, Isolation Forest, and PyTorch.
- `main.py`: Central PyTorch training orchestrator for the live DNN model.
- `app.py`: Streamlit frontend dashboard providing a live inference UI.
- `src/model.py`: Defines the PyTorch Neural Network and custom training algorithms.
- `src/evaluate.py`: Computes performance metrics, precision-recall constraints, and confusion matrices.
- `src/review_queue.py`: Implements the enterprise Decision Band logic and analyst queue generation.

## Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/DevRitesh08/-Credit-Card-Fraud-Detection.git
   cd -Credit-Card-Fraud-Detection
   ```

2. **Train the PyTorch Model** 
   *(Note: The trained `fraud_model.pth` and `scaler.pkl` are already included in the repository for immediate inference).*
   ```bash
   python main.py
   ```

3. **Launch the Dashboard**
   ```bash
   python -m streamlit run app.py
   ```
