# Financial Market Forecasting & Anomaly Detection System

## 📌 Overview
This project focuses on building a machine learning pipeline to predict stock price directions (UP/DOWN) and detect anomalous or fraudulent trading activities. The project leverages both Classical Machine Learning and Deep Learning models applied to time-series stock market data.

## ⚠️ Problems Solved
1. **Market Trend Prediction:** Predicting whether a stock's closing price will go up or down using historical data and technical indicators.
2. **Anomaly Detection:** Identifying irregular trading days that deviate from normal market behavior, which could indicate fraudulent activity or extreme market events.

## 🚀 Project Phases & Details

### Phase 1: Data & Classical ML (Implemented in `notebooks/phase-1.ipynb`)
- **Data Fetching:** Python scripts to download historical stock data using the `yfinance` library (`src/data/fetcher.py`).
- **Feature Engineering:** Calculating key technical indicators such as Moving Averages, RSI, MACD, and Bollinger Bands (`src/data/features.py`).
- **Exploratory Data Analysis (EDA):** Analyzing feature correlations and rolling statistics.
- **Classical Models:** 
  - `KNeighborsRegressor` for basic price modeling.
  - `GaussianNB` (Naive Bayes) for buy/sell signal classification.
  - `PCA` for dimensionality reduction.
  - Ensemble methods (`RandomForest` and `XGBoost`) utilizing `GridSearchCV` for hyperparameter tuning.

### Phase 2: Deep Learning (Implemented in `notebooks/phase-2.ipynb`)
- **1D-CNN (Convolutional Neural Networks):** A specialized architecture for time-series pattern recognition, predicting UP/DOWN price movements. Built with Dropout layers and GlobalAveragePooling to prevent overfitting.
- **AutoEncoder (Anomaly Detection):** An unsupervised learning model trained on normal stock behaviors. It flags days with a reconstruction error (MSE) higher than the 95th percentile as anomalies.
- **Transfer Learning:** Demonstrates how to take a pre-trained model on one stock, freeze early layers, and fine-tune it on a new stock to save computational resources and improve accuracy.

## 📊 Results
- **Phase 1:** Engineered features significantly boosted the baseline performance of classical models, with ensemble methods (`Random Forest`, `XGBoost`) providing the most reliable predictions.
- **Phase 2 (1D-CNN):** The convolutional network successfully learned complex temporal patterns, resulting in solid Accuracy and ROC-AUC scores for trend prediction.
- **Phase 2 (AutoEncoder):** The reconstruction-based anomaly detection effectively flagged the top 5% of trading days exhibiting highly unusual price action.
- **Phase 2 (Transfer Learning):** Showed noticeable accuracy improvements when adapting a pre-trained model to new, unseen stocks compared to training from scratch.

## 📂 Project Architecture

```plaintext
finance-ai-system/
├── data/               # Local data storage
│   ├── raw/            # Raw historical data (e.g., historical_stock_data.csv)
│   └── processed/      # Data with engineered features (e.g., processed_stock_data.csv)
├── notebooks/          # Jupyter notebooks for experimentation
│   ├── phase-1.ipynb   # EDA, Feature Engineering, and Classical ML models
│   └── phase-2.ipynb   # Deep Learning (CNN, AutoEncoder, Transfer Learning)
├── src/                # Core Python scripts
│   └── data/           
│       ├── fetcher.py  # Script to download data from Yahoo Finance
│       └── features.py # Script to compute RSI, MACD, Bollinger Bands, etc.
├── saved_models/       # Directory where trained models (.keras) are saved
│   └── deep_learning/  
├── configs/            
│   └── config.yaml     # Configuration file for tickers, dates, and parameters
├── .env.example        # Example environment variables file
├── .gitignore          # Git ignore rules
├── requirements.txt    # Project Python dependencies
└── README.md           # Project documentation
```

## ⚙️ Setup & Installation

### 1. Install Dependencies
Make sure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

### 2. Fetch Data
Download historical data for the configured stocks:
```bash
python src/data/fetcher.py
```

### 3. Process Data & Engineer Features
Generate technical indicators required for the models:
```bash
python src/data/features.py
```

### 4. Run Notebooks
You can now open and run `notebooks/phase-1.ipynb` and `notebooks/phase-2.ipynb` to train the models and see the results!
