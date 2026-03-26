import yfinance as yf
import pandas as pd
import os

def fetch_data(tickers, period="5y", save_path="data/raw/historical_stock_data.csv"):
    if not os.path.exists("data/raw"):
        os.makedirs("data/raw")
        
    print(f"Fetching data for: {tickers}")
    df_list = []
    
    for ticker in tickers:
        print(f"Downloading {ticker}...")
        stock_obj = yf.Ticker(ticker)
        stock_df = stock_obj.history(period=period)
        
        if stock_df.empty:
            continue
            
        stock_df['Ticker'] = ticker
        stock_df.reset_index(inplace=True)
        # Ensure standard column names
        stock_df = stock_df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Ticker']]
        # Date column might have timezone, let's normalize it to YYYY-MM-DD
        stock_df['Date'] = pd.to_datetime(stock_df['Date']).dt.date
        
        df_list.append(stock_df)
        
    if not df_list:
        print("No data fetched.")
        return None
        
    final_df = pd.concat(df_list, ignore_index=True)
    final_df.to_csv(save_path, index=False)
    print(f"Data successfully fetched and saved to {save_path}")
    
    return final_df

if __name__ == "__main__":
    # Default stocks if run directly
    default_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META']
    fetch_data(default_tickers)
