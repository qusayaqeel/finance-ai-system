import pandas as pd
import numpy as np
import os

def calculate_rsi(data, column='Close', window=14):
    delta = data[column].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, column='Close', fast=12, slow=26, signal=9):
    exp1 = data[column].ewm(span=fast, adjust=False).mean()
    exp2 = data[column].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_bollinger_bands(data, column='Close', window=20, num_std=2):
    sma = data[column].rolling(window=window).mean()
    std = data[column].rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return upper_band, lower_band

def engineer_features(input_path="data/raw/historical_stock_data.csv", output_path="data/processed/processed_stock_data.csv"):
    if not os.path.exists("data/processed"):
        os.makedirs("data/processed")
        
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found. Please run fetcher.py first.")
        return None
        
    print(f"Loading raw data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # We need to process features per ticker
    # Ensure it's sorted by Date
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['Ticker', 'Date']).reset_index(drop=True)
    
    print("Calculating technical indicators (RSI, MACD, Bollinger Bands, Moving Averages)...")
    
    # Apply calculations per ticker
    def add_features(group):
        group = group.copy()
        
        # Moving Averages
        group['SMA_20'] = group['Close'].rolling(window=20).mean()
        group['SMA_50'] = group['Close'].rolling(window=50).mean()
        
        # RSI
        group['RSI_14'] = calculate_rsi(group, window=14)
        
        # MACD
        macd, signal = calculate_macd(group)
        group['MACD'] = macd
        group['MACD_Signal'] = signal
        
        # Bollinger Bands
        upper, lower = calculate_bollinger_bands(group, window=20)
        group['Bollinger_Upper'] = upper
        group['Bollinger_Lower'] = lower
        
        # Target variable: Price Direction (1 if Tomorrow's Close > Today's Close, 0 otherwise)
        # Shift(-1) brings tomorrow's close price to today's row
        group['Target_Direction'] = (group['Close'].shift(-1) > group['Close']).astype(int)
        
        # We drop the last row of each group since we don't know tomorrow's price yet
        return group.iloc[:-1]

    # Calculate features for each ticker
    processed_df = df.groupby('Ticker', group_keys=False).apply(add_features).reset_index(drop=True)
    
    # Drop initial rows with NaN due to rolling windows (e.g., SMA_50 needs 50 days)
    processed_df.dropna(inplace=True)
    processed_df.reset_index(drop=True, inplace=True)
    
    # Save the processed data
    processed_df.to_csv(output_path, index=False)
    print(f"Feature engineering complete! Processed dataset shape: {processed_df.shape}")
    print(f"Data saved to {output_path}")
    
    return processed_df

if __name__ == "__main__":
    engineer_features()
