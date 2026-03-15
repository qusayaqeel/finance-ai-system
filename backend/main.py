"""
Main FastAPI entry point.
"""
from fastapi import FastAPI

app = FastAPI(title="Financial Market Forecasting API")

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok"}
