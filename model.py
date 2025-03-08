import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

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
        print(f"Model output: {output}")
        print(f"Output shape: {output.shape}")
        probabilities = F.softmax(output, dim=1)
        print(f"Probabilities: {probabilities}")
        predicted_class = torch.argmax(probabilities, dim=1).item()  # Convert tensor to a Python number
        print(f"Predicted class: {predicted_class}")
        class_map = {0: "Buy", 1: "Neutral", 2: "Sell"}
        predicted_label = class_map.get(predicted_class, "Unknown")
        print(f"Predicted action: {predicted_label}")
    return predicted_label