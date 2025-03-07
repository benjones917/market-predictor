from flask import Flask, render_template, request, jsonify
import torch
import numpy as np
import yfinance as yf
from model import LSTMClassifier, predict_stock_signal
from data_loader import preprocess_stock_data

app = Flask(__name__)

# Load trained model
model = LSTMClassifier()
model.load_state_dict(torch.load("model.pth"))
model.eval()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        stock_symbol = request.form["stock"]
        try:
            X_test = preprocess_stock_data(stock_symbol)
            prediction = predict_stock_signal(model, X_test)
            return render_template("index.html", stock=stock_symbol, prediction=prediction)
        except Exception as e:
            return render_template("index.html", error=f"Error: {str(e)}")
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)