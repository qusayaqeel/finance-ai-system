import os
import json

base_dir = r"d:\projects\Finance-Ai-System\finance-ai-system"

dirs = [
    "data/raw", "data/processed", "data/synthetic",
    "notebooks",
    "src/data", "src/models/classical", "src/models/deep_learning", "src/models/gan",
    "src/explainability", "src/evaluation", "src/chatbot",
    "backend/routers", "backend/schemas", "backend/services",
    "frontend/src/components", "frontend/src/services",
    "saved_models/classical", "saved_models/deep_learning",
    "tests",
    "configs"
]

for d in dirs:
    os.makedirs(os.path.join(base_dir, d), exist_ok=True)

files = {
    "notebooks/01_eda.ipynb": '{\n "cells": [],\n "metadata": {},\n "nbformat": 4,\n "nbformat_minor": 5\n}',
    "notebooks/02_classical_ml.ipynb": '{\n "cells": [],\n "metadata": {},\n "nbformat": 4,\n "nbformat_minor": 5\n}',
    "notebooks/03_deep_learning.ipynb": '{\n "cells": [],\n "metadata": {},\n "nbformat": 4,\n "nbformat_minor": 5\n}',
    "notebooks/04_advanced_models.ipynb": '{\n "cells": [],\n "metadata": {},\n "nbformat": 4,\n "nbformat_minor": 5\n}',
    
    "src/data/__init__.py": '"""Data module for fetching, feature engineering, and preprocessing."""\n',
    "src/data/fetcher.py": '"""\nResponsible for fetching stock data from yfinance.\nSupports historical mode (N years of data) and live mode (latest 30 days for daily inference).\n"""\n\ndef fetch_data(mode="historical", **kwargs):\n    """\n    Fetch data for training (historical) or daily inference (live).\n    """\n    pass\n',
    "src/data/features.py": '"""Responsible for engineering technical indicators: RSI, MACD, SMAs, Bollinger Bands."""\n',
    "src/data/preprocessor.py": '"""Responsible for scaling, normalizing, and splitting data into train/val/test sets."""\n',

    "src/models/__init__.py": '"""Models module for classical, deep learning, and advanced architectures."""\n',
    
    "src/models/classical/__init__.py": '"""Classical Machine Learning models."""\n',
    "src/models/classical/knn.py": '"""K-Nearest Neighbors model for stock direction prediction."""\n',
    "src/models/classical/naive_bayes.py": '"""Naive Bayes model for stock direction prediction."""\n',
    "src/models/classical/random_forest.py": '"""Random Forest model for stock direction prediction."""\n',
    "src/models/classical/xgboost_model.py": '"""XGBoost model for stock direction prediction."""\n',

    "src/models/deep_learning/__init__.py": '"""Deep Learning models."""\n',
    "src/models/deep_learning/cnn1d.py": '"""1D Convolutional Neural Network for time series features."""\n',
    "src/models/deep_learning/autoencoder.py": '"""Autoencoder for anomaly detection in stock market data."""\n',
    "src/models/deep_learning/lstm.py": '"""LSTM/GRU models for capturing temporal dependencies."""\n',
    "src/models/deep_learning/transformer.py": '"""Transformer model with attention mechanisms for forecasting."""\n',

    "src/models/gan/__init__.py": '"""Generative Adversarial Networks for synthetic financial data generation."""\n',
    "src/models/gan/generator.py": '"""GAN Generator for creating synthetic market data."""\n',
    "src/models/gan/discriminator.py": '"""GAN Discriminator for distinguishing real vs synthetic market data."""\n',

    "src/explainability/__init__.py": '"""Model explainability utilizing SHAP values."""\n',
    "src/explainability/shap_explainer.py": '"""Calculates and extracts SHAP values to explain model predictions."""\n',
    
    "src/evaluation/__init__.py": '"""Model evaluation metrics and validation logic."""\n',
    "src/evaluation/metrics.py": '"""Custom metrics for evaluating forecasting models (Accuracy, F1, Sharpe ratio pseudo-metrics)."""\n',

    "src/chatbot/__init__.py": '"""Chatbot integration for plain language explanations."""\n',
    "src/chatbot/llm_client.py": '"""Client for interacting with LLM API to translate SHAP values into text."""\n',
    "src/chatbot/prompt_builder.py": '"""Constructs prompts containing prediction results and SHAP explanations for the LLM."""\n',

    "backend/main.py": '"""\nMain FastAPI entry point.\n"""\nfrom fastapi import FastAPI\n\napp = FastAPI(title="Financial Market Forecasting API")\n\n@app.get("/health")\ndef health_check():\n    """Health check endpoint."""\n    return {"status": "ok"}\n',
    
    "backend/routers/__init__.py": '"""FastAPI routers for various endpoints."""\n',
    "backend/routers/predict.py": '"""Router for prediction endpoints."""\n',
    "backend/routers/explain.py": '"""Router for SHAP explainability endpoints."""\n',
    "backend/routers/chat.py": '"""Router for chatbot interactions."""\n',

    "backend/schemas/__init__.py": '"""Pydantic schemas for request and response validation."""\n',
    "backend/schemas/models.py": '"""Pydantic models for API interactions validation."""\n',

    "backend/services/__init__.py": '"""Business logic and background services."""\n',
    "backend/services/prediction_service.py": '"""Service for handling inference logic using loaded models."""\n',
    "backend/services/chat_service.py": '"""Service coordinating prediction, SHAP explanations, and LLM translation."""\n',
    "backend/services/scheduler.py": '"""\nAPScheduler jobs for daily market data fetch and inference.\n"""\nfrom apscheduler.schedulers.background import BackgroundScheduler\nimport pytz\n\ndef daily_inference_job():\n    """\n    Job that runs daily after market close.\n    Calls fetcher in live mode (latest 30 days), runs inference, \n    and saves results to processed/.\n    """\n    pass\n\ndef start_scheduler():\n    """Start the APScheduler for daily tasks."""\n    scheduler = BackgroundScheduler(timezone=pytz.timezone("US/Eastern"))\n    # Run at 16:30 EST (after market close)\n    scheduler.add_job(daily_inference_job, "cron", day_of_week="mon-fri", hour=16, minute=30)\n    scheduler.start()\n',

    "frontend/src/components/Dashboard.jsx": '// Main Dashboard component layout.\n',
    "frontend/src/components/StockChart.jsx": '// Component for live stock charts using Recharts.\n',
    "frontend/src/components/PredictionCard.jsx": '// Component displaying UP/DOWN prediction and probability.\n',
    "frontend/src/components/Chatbot.jsx": '// Chat interface component for interacting with the LLM explanation API.\n',
    "frontend/src/services/api.js": '// API service functions to communicate with the FastAPI backend.\n',
    "frontend/src/App.jsx": '// Main React Application entry point.\n',
    "frontend/package.json": '{\n  "name": "finance-frontend",\n  "version": "1.0.0",\n  "private": true\n}',

    "tests/test_features.py": '"""Unit tests for feature engineering."""\n',
    "tests/test_models.py": '"""Unit tests for model training and inference."""\n',
    "tests/test_api.py": '"""Unit tests for FastAPI endpoints."""\n',

    "configs/config.yaml": '# Configuration variables\n\ntickers:\n  - AAPL\n  - MSFT\n  - GOOGL\n  - AMZN\n  - TSLA\n\ndata:\n  window_size: 30\n  train_start_date: "2010-01-01"\n  train_end_date: "2022-12-31"\n  val_start_date: "2023-01-01"\n  val_end_date: "2023-12-31"\n  test_start_date: "2024-01-01"\n  test_end_date: "2024-12-31"\n\nscheduler:\n  timezone: "US/Eastern"\n  market_close_run_time: "16:30"\n\nmodels:\n  random_forest:\n    n_estimators: 100\n    max_depth: 10\n  xgboost:\n    learning_rate: 0.1\n    n_estimators: 200\n  lstm:\n    hidden_size: 64\n    num_layers: 2\n    dropout: 0.2\n    learning_rate: 0.001\n    epochs: 50\n    batch_size: 32\n',

    "requirements.txt": 'fastapi==0.104.1\nuvicorn==0.24.0\nyfinance==0.2.31\npandas==2.1.3\nnumpy==1.26.2\nscikit-learn==1.3.2\ntorch==2.1.1\nxgboost==2.0.2\nshap==0.43.0\napscheduler==3.10.4\npytz==2023.3\npyyaml==6.0.1\n',

    ".env.example": '# Environment variables example\nLLM_API_KEY=\nDB_CONNECTION_STRING=\n',

    ".gitignore": '# Environment variables\n.env\n.env.*\n\n# Data\ndata/raw/\nsaved_models/\n\n# Python caching\n__pycache__/\n*.py[cod]\n*$py.class\n\n# Jupyter\n.ipynb_checkpoints\n\n# Frontend\nnode_modules/\ndist/\n',

    "README.md": '# Financial Market Forecasting & AI Chatbot System\nA comprehensive system to predict stock price direction and explain the insights via an AI chatbot.\n'
}

for fp, content in files.items():
    full_path = os.path.join(base_dir, fp)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content)
