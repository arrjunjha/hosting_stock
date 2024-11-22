# Stock Predictor with LSTM Neural Networks

## Overview
This project uses Long Short-Term Memory (LSTM) Neural Networks to predict stock prices based on historical stock data. It also provides an interactive Streamlit interface for data visualization and predictions.

## Technologies Used
- Python
- TensorFlow/Keras
- yFinance API
- Streamlit
- Matplotlib

## Features
- Predict next day's stock price.
- Visualize historical data with moving averages (MA50, MA200).
- Interactive web-based interface.

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git

Navigate to the directory:
cd <repo-name>

Install dependencies:
pip install -r requirements.txt

Run the Streamlit app:
streamlit run app.py

Future Improvements:
Incorporate external factors like news sentiment analysis.
Add support for real-time stock predictions.