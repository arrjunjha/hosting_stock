import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from datetime import date

# Load pre-trained model
model = load_model('stock.keras')

# Streamlit interface
st.header('Stock Predictor')

# Stock input
st.subheader('Please write .NS after the stock symbol, only for Indian stocks')

stock = st.text_input('Enter Stock Symbol', 'RELIANCE.NS')

start = '2012-01-01'
# end = '2024-11-17'
end = date.today().strftime('%Y-%m-%d')


# Download data
data = yf.download(stock, start, end)

# Display stock data
st.subheader('Stock Data')
st.write(data)

# Train and test split
data_train = pd.DataFrame(data['Close'][0: int(len(data) * 0.80)])
data_test = pd.DataFrame(data['Close'][int(len(data) * 0.80): len(data)])

# Scaling
scaler = MinMaxScaler(feature_range=(0, 1))

# Combine last 100 days of train data with test data for scaling
past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days, data_test])
data_test_scaled = scaler.fit_transform(data_test)

# MA50 (Moving Average 50 Days), MA100 (Moving Average 100 Days), MA200 (Moving Average 200 Days)
st.subheader('Moving Averages (MA50, MA200)')

ma_50_days = data['Close'].rolling(50).mean()

ma_200_days = data['Close'].rolling(200).mean()

fig1 = plt.figure(figsize=(10, 6))
plt.plot(ma_50_days, label="MA50 (50 Days)", color='r')

plt.plot(ma_200_days, label="MA200 (200 Days)", color='g')
plt.plot(data['Close'], label="Closing Price", color='b')
plt.legend()
plt.xlabel("Date")
plt.ylabel("Price")
plt.title("Moving Averages (MA50, MA200) and Closing Prices")
st.pyplot(fig1)


# Prepare data for prediction
x = []
y = []
for i in range(100, data_test_scaled.shape[0]):
    x.append(data_test_scaled[i - 100:i])
    y.append(data_test_scaled[i, 0])

x, y = np.array(x), np.array(y)

# Reshaping x to be 3D as expected by LSTM
x = np.reshape(x, (x.shape[0], x.shape[1], 1))

# Predictions
predictions = model.predict(x)

# Rescale predictions and true values back to original range
predictions = scaler.inverse_transform(predictions)
y = scaler.inverse_transform(y.reshape(-1, 1))

# Plot predictions vs actual values
st.subheader('Predicted vs Actual Prices')
fig2 = plt.figure(figsize=(10, 6))
plt.plot(y, label="Actual Prices", color='b')
plt.plot(predictions, label="Predicted Prices", color='orange')
plt.legend()
plt.xlabel("Time Steps")
plt.ylabel("Price")
plt.title("Predicted vs Actual Prices")
st.pyplot(fig2)

# Option to predict tomorrow's price
st.subheader("Predict Tomorrow's Price")

# Get the last 100 days' data for prediction
last_100_days = data['Close'].tail(100).values.reshape(-1, 1)
last_100_days_scaled = scaler.transform(last_100_days)

# Reshape for LSTM (3D input)
last_100_days_scaled = np.reshape(last_100_days_scaled, (1, last_100_days_scaled.shape[0], 1))

# Predict tomorrow's price
predicted_tomorrow = model.predict(last_100_days_scaled)

# Rescale the prediction to original range
predicted_tomorrow_price = scaler.inverse_transform(predicted_tomorrow)

# Display predicted tomorrow's price
st.write(f"Predicted price for tomorrow: {predicted_tomorrow_price[0][0]:.2f}")
