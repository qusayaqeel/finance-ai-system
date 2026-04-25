# Financial Market Forecasting & Anomaly Detection System

## 📌 Overview
This project focuses on building a machine learning pipeline to predict stock price directions (UP/DOWN) and detect anomalous or fraudulent trading activities. The project leverages both Classical Machine Learning and Deep Learning models applied to time-series stock market data.

## ⚠️ Problems Solved
1. **Market Trend Prediction:** Predicting whether a stock's closing price will go up or down using historical data and technical indicators.
2. **Anomaly Detection:** Identifying irregular trading days that deviate from normal market behavior, which could indicate fraudulent activity or extreme market events.

## 📈 Engineered Technical Features (Data Dictionary)
In order to help the Machine Learning models understand market momentum and volatility, several technical indicators were engineered in `src/data/features.py`:

1. **SMA_20 & SMA_50 (Simple Moving Averages):** 
   - *What it is:* The average closing price over the last 20 and 50 days.
   - *Problem it solves:* Stock prices are noisy. SMAs smooth out daily price fluctuations, helping the model identify the underlying long-term and short-term trends rather than getting confused by daily spikes.

2. **RSI_14 (Relative Strength Index):** 
   - *What it is:* A momentum oscillator measured on a scale from 0 to 100.
   - *Problem it solves:* It tells the model if a stock is "Overbought" (too high, might drop soon) or "Oversold" (too low, might bounce back). This provides crucial reversal signals to the AI.

3. **MACD & MACD_Signal (Moving Average Convergence Divergence):**
   - *What it is:* A trend-following momentum indicator that shows the relationship between two moving averages of a stock's price.
   - *Problem it solves:* It helps the model detect changes in the strength, direction, momentum, and duration of a trend. Crosses between the MACD and Signal line are strong predictors for UP/DOWN movements.

4. **Bollinger_Upper & Bollinger_Lower (Bollinger Bands):**
   - *What it is:* Two volatility bands placed above and below a moving average.
   - *Problem it solves:* Markets constantly change between quiet and highly volatile periods. These bands give the model a dynamic range; if the price hits the upper band, it might be overvalued, and if it hits the lower band, it might be undervalued.

5. **Target_Direction:**
   - *What it is:* A binary label (`1` for UP, `0` for DOWN) created by comparing tomorrow's closing price to today's closing price.
   - *Problem it solves:* This is the fundamental "Ground Truth" the model learns to predict. It shifts future data into a target label without causing data leakage.

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
