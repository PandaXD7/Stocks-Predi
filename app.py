import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as df
from keras.models import load_model # type: ignore
import streamlit as st
import yfinance as yf


start="2010-01-01"
end="2019-12-31"

st.title("Stock Trend Prediction")

user_input = st.text_input("Enter Stock Ticker","AAPL")
df = yf.download(user_input,start=start, end=end)


st.subheader('Data from 2010 - 2019')
st.write(df.describe())

st.subheader('Closings Price vs Time chart')
fig = plt.figure(figsize=(12, 6))
plt.plot(df['Close'])
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA')
ma100 = df['Close'].rolling(100).mean()  # Add parentheses to call the mean() function
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100, label='100MA')  # Optional: add label for clarity
plt.plot(df['Close'], label='Closing Price')
plt.legend() 
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA & 200MA')
ma100 = df['Close'].rolling(100).mean()  # Line 38: Make sure parentheses are used with .mean()
ma200 = df['Close'].rolling(200).mean()  # Line 39


fig = plt.figure(figsize=(12, 6))
plt.plot(ma100, 'r', label='100MA')  # Label added for clarity
plt.plot(ma200, 'g', label='200MA')  # Label added for clarity
plt.plot(df['Close'], 'b', label='Closing Price')  # Label added for clarity
plt.legend()  
st.pyplot(fig)


data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])  
data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70): int(len(df))])  

print(data_training.shape)
print(data_testing.shape)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data_training_array = scaler.fit_transform(data_training)

x_train = []
y_train = []

for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100: i])
    y_train.append(data_training_array[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# Load my model
model = load_model('my_model.keras')

past_100_days = data_training.tail(100)
final_data = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_data)
x_test = []
y_test= []

for i in range (100, data_training_array.shape[0]):
    x_test.append(data_training_array[i-100: i])
    y_test.append(data_training_array[i,0])


x_test ,y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
scaler = scaler.scale_
scale_factor = 1/ 0.0220294
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

st.subheader('Predictions VS Original')
fig = plt.figure(figsize = (12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted,'r' , label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)

