# Stocks-Trend-Prediction

1. Abstract
This project focuses on predicting stock trends using historical stock market data and machine learning techniques. A user can input a stock ticker to retrieve data from 2010 to 2019 in a tabular format, accompanied by visualizations such as closing price vs. time charts, moving averages (100 days), and predictions vs. actual values. The application is built using Streamlit, providing an intuitive interface for data visualization and trend forecasting. The project aims to offer insights into stock behavior, enabling better investment decisions.

2. Introduction
Stock market prediction has always been a challenging and intriguing domain due to its complex and dynamic nature. Leveraging AI and machine learning, this project explores how historical stock data can be used to forecast future trends. This Streamlit-based application allows users to visualize historical data, calculate moving averages, and predict stock prices with minimal effort. The project demonstrates how data-driven methods can enhance decision-making in financial markets while emphasizing the importance of accessible, user-friendly tools.

3. Related Work
Several studies and projects have attempted to predict stock market trends using AI and machine learning. Traditional approaches, such as time-series analysis and technical indicators, have evolved with the inclusion of models like Linear Regression, ARIMA, LSTM, and GRU. For instance:

Research on using Recurrent Neural Networks (RNN) for time-series predictions has shown promising results.
Tools like Yahoo Finance and libraries such as yfinance and pandas have enabled seamless data retrieval and preprocessing.
Similar projects have been developed using libraries like TensorFlow and PyTorch, providing baseline methodologies for comparison.

4. Methodology
The project workflow involves the following steps:

Data Collection:

Historical stock data (2010–2019) is retrieved using APIs like yfinance.
Data includes Open, High, Low, Close, Volume, and Adjusted Close values.
Preprocessing:

Data cleaning and handling of missing values.
Calculation of technical indicators like moving averages (100 days).
Visualization:

Closing Price vs. Time Chart.
Closing Price with 100-day Moving Average Chart.
Prediction Model:

Use of machine learning techniques (e.g., Linear Regression or LSTM) for forecasting future stock prices.
Train-test split to evaluate model performance.
Streamlit Integration:

Building an interactive interface to input stock tickers, view data tables, visualizations, and predictions.
