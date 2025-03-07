import torch
import torch.nn as nn
import numpy as np

class LSTMClassifier(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=3):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Take last time step
        return out

def predict_stock_signal(model, X_test):
    with torch.no_grad():
        output = model(X_test)
        predicted_class = torch.argmax(output, axis=1).item()
    return ["Sell", "Hold", "Buy"][predicted_class + 1]  # Convert to readable label