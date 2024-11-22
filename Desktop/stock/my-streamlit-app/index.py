import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# Load stock data
start = '2012-01-01'
end = '2024-11-20'
stock = 'RELIANCE.NS'

data = yf.download(stock, start, end)
data.reset_index(inplace=True)

# Calculate moving averages
ma_200_days = data.Close.rolling(200).mean()
ma_100_days = data.Close.rolling(100).mean()

# Split data into training and test sets
data_train = pd.DataFrame(data.Close[0: int(len(data) * 0.80)])
data_test = pd.DataFrame(data.Close[int(len(data) * 0.80): len(data)])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Scale training data
data_train_scale = scaler.fit_transform(data_train)

# Prepare training sequences
x = []
y = []
for i in range(100, data_train_scale.shape[0]):
    x.append(data_train_scale[i - 100:i])
    y.append(data_train_scale[i, 0])

x, y = np.array(x), np.array(y)

# Reshape x for LSTM
x = x.reshape((x.shape[0], x.shape[1], 1))

import tensorflow as tf
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential

# Build the LSTM model
model = Sequential()

# First LSTM layer
model.add(LSTM(units=50, activation='relu', return_sequences=True,
               input_shape=(x.shape[1], 1)))
model.add(Dropout(0.2))

# Second LSTM layer
model.add(LSTM(units=60, activation='relu', return_sequences=True))
model.add(Dropout(0.3))

# Third LSTM layer
model.add(LSTM(units=80, activation='relu', return_sequences=True))
model.add(Dropout(0.4))

# Fourth LSTM layer
model.add(LSTM(units=120, activation='relu'))
model.add(Dropout(0.5))

# Dense output layer
model.add(Dense(units=1))

# Compile the model
model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['accuracy']
)

# Train the model
model.fit(x, y, epochs=50, batch_size=32, verbose=1)
pas_100_days = data_train.tail(100)

data_test = pd.concat([pas_100_days, data_test], ignore_index=True )
data_test_scale = scaler.fit_transform(data_test)

x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i - 100:i])
    y.append(data_test_scale[i, 0])

x , y = np.array(x), np.array(y)

y_predict = model.predict(x)

scale = 1/scaler.scale_

y_predict = y_predict*scale
y = y*scale

plt.figure(figsize=(10, 8))
plt.plot(y_predict, 'r', label = 'pridicted ')
plt.plot(y, 'g', label = 'original')
plt.xlabel('time')
plt.ylabel('price')
plt.legend()
plt.show()

model.save('stock.keras')