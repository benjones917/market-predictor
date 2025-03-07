import torch
import torch.nn as nn
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from model import LSTMClassifier

# List of stocks to train on
stock_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]

all_X, all_y = [], []

seq_length = 50  # Sequence length

def create_sequences(data, labels, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length]
        y = labels[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

for stock in stock_symbols:
    print(f"Downloading data for {stock}...")
    # Download the maximum available historical data
    df = yf.download(stock, period="max")
    if df.empty:
        print(f"No data found for {stock}. Skipping.")
        continue

    # Calculate daily returns
    df['Returns'] = df['Close'].pct_change()

    # Define signals: 1 (Buy), 0 (Hold), -1 (Sell)
    threshold = 0.005  # Example threshold of 0.5%
    df['Signal'] = 0  # Default to Hold
    df.loc[df['Returns'] > threshold, 'Signal'] = 1  # Buy signal
    df.loc[df['Returns'] < -threshold, 'Signal'] = -1  # Sell signal

    # Shift labels to be in the range [0, 1, 2]
    df['Signal'] = df['Signal'] + 1

    # Remove any rows with NaN values
    df.dropna(inplace=True)

    # Use only the 'Close' price for input
    prices = df[['Close']].values

    # Normalize the price data between -1 and 1
    scaler = MinMaxScaler(feature_range=(-1, 1))
    prices_scaled = scaler.fit_transform(prices)

    # Create sequences for the LSTM model
    X, y = create_sequences(prices_scaled, df['Signal'].values, seq_length)
    all_X.append(X)
    all_y.append(y)

if len(all_X) == 0:
    raise ValueError("No training data was collected for any of the stocks.")

# Combine the sequences from all stocks
X_train = np.vstack(all_X)
y_train = np.hstack(all_y)

# Convert the numpy arrays to PyTorch tensors
X_train = torch.Tensor(X_train)
y_train = torch.LongTensor(y_train)

# Initialize the LSTM model
model = LSTMClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Save the trained model's state dictionary
torch.save(model.state_dict(), "model.pth")
print("Model trained on multiple stocks and saved as model.pth")