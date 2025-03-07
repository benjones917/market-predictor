import yfinance as yf
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler

seq_length = 50  # Match training sequence length

def preprocess_stock_data(stock_symbol):
    df = yf.download(stock_symbol, period="6mo")
    
    if df.empty:
        raise ValueError("Invalid stock symbol or no data available.")

    # Use only the 'Close' price
    prices = df[['Close']].values

    # Normalize data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    prices_scaled = scaler.fit_transform(prices)

    # Create sequences
    X = []
    for i in range(len(prices_scaled) - seq_length):
        X.append(prices_scaled[i:i + seq_length])

    # Convert to PyTorch tensor (use last sequence)
    X_test = torch.Tensor([X[-1]])  # Only latest data for prediction
    return X_test