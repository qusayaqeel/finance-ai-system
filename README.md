# Financial Market Forecasting & AI Chatbot System

A comprehensive system to predict stock price direction (UP/DOWN) and explain the predictions in plain language via an AI chatbot using SHAP values and an LLM API.

## 🚀 Features
- **Automated Data Pipeline:** Fetches raw stock data automatically from Yahoo Finance.
- **Advanced Feature Engineering:** Calculates RSI, MACD, Moving Averages, and Bollinger Bands.
- **Multi-stage Modeling Pipeline:**
  - *Phase 1:* Classical ML (KNN, Naive Bayes, Random Forest, XGBoost)
  - *Phase 2:* Deep Learning (1D-CNN, Autoencoder, LSTM/GRU)
  - *Phase 3:* Advanced (Transformer, GAN for synthetic data)
- **XAI (Explainable AI):** Uses SHAP values to interpret model decisions.
- **AI Chatbot Interface:** Translates complex SHAP metrics into human-readable explanations.
- **Automated Scheduler:** Runs daily inference automatically after market close.

## 🛠️ Tech Stack
- **Machine Learning:** PyTorch, scikit-learn, XGBoost, SHAP
- **Backend:** FastAPI, apscheduler
- **Frontend:** React, Recharts
- **Data:** yfinance, pandas, numpy

## ⚙️ Setup & Installation

### 1. Clone & Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
Create a `.env` file based on `.env.example`:
```env
LLM_API_KEY=your_api_key_here
```

### 3. Fetch Data
To fetch historical data for training:
```bash
python src/data/fetcher.py
```

### 4. Start the Application
**Backend:**
```bash
uvicorn backend.main:app --reload
```

**Frontend:**
```bash
cd frontend
npm install
npm start
```
