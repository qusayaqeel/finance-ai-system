# Progress Report: Financial Market Forecasting & Anomaly Detection

**To:** Course Instructor / Professor
**Subject:** Summary of Implementation, Methodology Updates, and Future Plans
**Project:** Financial Market Forecasting & Anomaly Detection

---

## 1. Introduction
This progress report outlines the completed milestones for Phase 1 and Phase 2 of the Financial Market Forecasting and Anomaly Detection project. The primary objective is to predict stock price trends (UP/DOWN) and identify fraudulent or anomalous trading activities. The report highlights the methodologies applied, results achieved, challenges faced during the implementation, and the proposed plan for Phase 3.

---

## 2. Phase 1: Data Processing & Classical Machine Learning

### 2.1 Methodology
- **Data Engineering:** We successfully built an automated pipeline to fetch historical stock data.
- **Feature Engineering:** A key focus was designing robust technical indicators to represent market momentum and volatility. We calculated features such as Relative Strength Index (RSI), Moving Average Convergence Divergence (MACD), Simple Moving Averages (SMA), and Bollinger Bands.
- **Modeling:** We established strong baselines using Classical Machine Learning algorithms, including `K-Nearest Neighbors (KNN)`, `Gaussian Naive Bayes`, `Random Forest`, and `XGBoost`. `PCA` was also utilized for dimensionality reduction.

### 2.2 Challenges Faced
- **Feature Engineering Complexity:** Designing and selecting the right features (like MACD and RSI) required careful consideration of time windows to prevent data leakage while accurately capturing market trends.
- **Data Volatility:** Financial data is inherently noisy and highly volatile. The rapid fluctuations in stock prices made it difficult for simple models to distinguish between actual trends and market noise.

### 2.3 Results
- Engineered features significantly boosted model performance. Ensemble regressors (`Random Forest` and `XGBoost`), combined with hyperparameter tuning via `GridSearchCV`, provided the most reliable predictions and outperformed the simpler baseline models.

---

## 3. Phase 2: Deep Learning (CNN / AutoEncoder / Transfer Learning)

### 3.1 Methodology
- **1D-CNN for Time-Series:** We implemented a 1D Convolutional Neural Network (1D-CNN) tailored for sequential data. It successfully extracted temporal patterns from the sliding windows of historical prices to classify future price directions.
- **AutoEncoder for Anomaly Detection:** We built an unsupervised AutoEncoder to learn the "normal" representation of the stock data. 
- **Transfer Learning:** We implemented a pipeline to train a CNN on a source stock, freeze its feature-extraction layers, and fine-tune it on a target stock.

### 3.2 Challenges Faced & Architectural Decisions
- **Why an AutoEncoder?** One of the main challenges was that anomalies and fraudulent activities in financial data are extremely rare and lack explicit labels (Highly Imbalanced Data). To solve this, we opted for an AutoEncoder. By training it exclusively on standard market data, the model yields a high Mean Squared Error (MSE) / Reconstruction Error when encountering abnormal data. We established the 95th percentile of the validation MSE as a threshold to flag anomalies dynamically.
- **Model Overfitting:** Deep learning models naturally tend to overfit small financial datasets. We mitigated this by applying `Dropout` layers, `GlobalAveragePooling`, and dynamic learning rate reduction (`ReduceLROnPlateau`).

### 3.3 Results
- The 1D-CNN model delivered strong Accuracy and ROC-AUC scores for trend classification.
- The AutoEncoder successfully identified the top 5% of abnormal trading days without needing prior labels.
- Transfer Learning demonstrated a clear accuracy improvement and faster convergence when adapting the model to unseen stocks.

---

## 4. Phase 3: Advanced Architectures (Upcoming Plan)

For the final phase of the project, we aim to integrate advanced architectures and Explainable AI (XAI) to improve robustness and transparency:

- **Transformers & Sequential Modeling:** Introduce LSTMs/Transformers for processing complex sequences (Note: adapting sequence processing for spatial/scene regions using Attention mechanisms).
- **Synthetic Data Generation:** Generate synthetic scenarios using Deep Convolutional Generative Adversarial Networks (DCGAN) to balance the dataset and simulate extreme market/environmental conditions.
- **Explainable AI (XAI):** Apply SHAP and GradCAM techniques to explain the model's predictions, ensuring the decisions are interpretable.
- **Fairness & Bias Documentation:** Document the model's fairness and reliability across various distinct conditions (e.g., extreme volatility or varying external environmental factors).

---
*End of Report*
