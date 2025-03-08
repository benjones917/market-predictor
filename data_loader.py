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
    # Ensure X is a list of NumPy arrays
    X = np.array(X, dtype=np.float32)  # Convert entire list to a NumPy array

    print(f"X converted to NumPy, shape: {X.shape}")  # Debugging print
    print(f"X length: {len(X)}") 
    # Convert last sequence into PyTorch tensor
    X_test = torch.tensor(X[-1], dtype=torch.float32)  # Use last sequence
    X_test = X_test.unsqueeze(0)  # Add batch dimension: (1, seq_length, features)
    print(f"X_test shape for prediction: {X_test.shape}")
    return X_test