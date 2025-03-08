import torch
import torch.nn as nn
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from model import LSTMClassifier

# List of stocks to train on
stock_symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "IBM"]

all_X, all_y = [], []

seq_length = 50  # Sequence length

def create_sequences(data, labels, seq_length):
    xs, ys = [], []
    
    if len(data) < seq_length:  # Ensure data is long enough
        print(f"Not enough data for sequence length {seq_length}. Skipping.")
        return np.array([]), np.array([])

    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length]
        y = labels[i + seq_length]
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)

for stock in stock_symbols:
    print(f"Fetching data for {stock}...")

    df = yf.download(stock, period="max")
    
    if df.empty:
        print(f"Warning: No data found for {stock}. Skipping.")
        continue

    df['Returns'] = df['Close'].pct_change()

    # Define buy/sell/hold signals
    threshold = 0.005
    df['Signal'] = 0
    df.loc[df['Returns'] > threshold, 'Signal'] = 1  # Buy
    df.loc[df['Returns'] < -threshold, 'Signal'] = -1  # Sell

    df.dropna(inplace=True)

    # Normalize price data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    prices_scaled = scaler.fit_transform(df[['Close']].values)

    df['Signal'] = df['Signal'] + 1

    # Create sequences
    X, y = create_sequences(prices_scaled, df['Signal'].values, seq_length=50)

    if len(X) == 0 or len(y) == 0:  # Check if sequences are empty
        print(f"Warning: No valid sequences for {stock}. Skipping.")
        continue

    all_X.append(X)
    all_y.append(y)

# Ensure we have data before converting
if not all_X:
    raise ValueError("No valid stock data found. Check stock symbols or data availability.")

X_train = torch.Tensor(np.vstack(all_X))  # Stack sequences properly
y_train = torch.LongTensor(np.hstack(all_y))  # Flatten labels correctly

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