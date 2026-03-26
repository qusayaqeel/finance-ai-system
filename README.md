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

## 📂 Project Architecture

```plaintext
finance-ai-system/
├── data/           # يحتفظ بالبيانات (Raw الخام، Processed للميزات، Synthetic المولّدة بواسطة GAN).
├── notebooks/      # بيئة التجارب (Jupyter). لعمل الـ EDA وتصميم النماذج قبل برمجتها النهائية.
├── src/            # المنطق الأساسي للمشروع (Core Logic). وتتفرع إلى:
│   ├── data/           # سحب البيانات وإعداد هندسة الميزات (RSI, MACD, Volume).
│   ├── models/         # خوارزميات الذكاء الاصطناعي (Classical، Deep Learning، GANs).
│   ├── explainability/ # تطبيق تقنية SHAP لتفسير قرارات الصندوق الأسود (XAI).
│   ├── evaluation/     # تقييم النماذج (Accuracy, Sharpe ratio, وغيرها).
│   └── chatbot/        # ممر لترجمة أرقام SHAP المعقدة إلى نصوص مفهومة عبر LLM.
├── backend/        # واجهة برمجة التطبيقات (FastAPI) والمجدول الزمني للتدريب والسحب اليومي.
├── frontend/       # واجهة المستخدم (React) وعروض الرسوم البيانية التفاعلية.
├── saved_models/   # تخزين أوزان النماذج للمنصة لضمان سرعة التنبؤ (Inference).
├── tests/          # اختبارات الوحدة (Unit Tests).
└── configs/        # إعدادات النظام المركزية للتحكم بالأسهم والموديلات.
```

## 📊 Data Dictionary

The processed dataset in (`data/processed/processed_stock_data.csv`) contains the following columns engineered for AI model training:

- **Date:** The trading date.
- **Open, High, Low, Close:** The daily opening, highest, lowest, and closing prices of the stock.
- **Volume:** The number of shares traded during the day.
- **Ticker:** The stock symbol (e.g., AAPL, MSFT).
- **SMA_20 / SMA_50:** Simple Moving Averages for 20 and 50 days. (Helps the AI identify the general trend and filter daily noise).
- **RSI_14:** Relative Strength Index. (Measures if a stock is overbought or oversold to anticipate price corrections).
- **MACD / MACD_Signal:** Moving Average Convergence Divergence. (Identifies changes in momentum and potential upcoming trend reversals).
- **Bollinger_Upper / Bollinger_Lower:** Bollinger Bands. (Measures market volatility; upper band indicates a relative high, lower indicates a low).
- **Target_Direction:** The target variable the model will train on. `1` if tomorrow's closing price is *higher* than today's, and `0` if it's lower or equal.

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
