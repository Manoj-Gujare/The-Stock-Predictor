# Stock Trend Prediction

## Description
This project implements a stock trend prediction system using historical stock price data and deep learning techniques. The system predicts the future trend of a selected stock based on its historical closing prices. It utilizes Long Short-Term Memory (LSTM) networks for time-series forecasting and is deployed using Streamlit for interactive visualization.

## Tools & Libraries
- Python (Version 3.x)
- NumPy
- Pandas
- Matplotlib
- Pandas Datareader
- TensorFlow
- Keras
- yfinance
- Streamlit

## Dataset
The project fetches historical stock price data from Yahoo Finance using the yfinance library. The data includes daily closing prices of the selected stock from January 1, 2010, to December 31, 2019.

## Data Preprocessing
- The fetched data is preprocessed to remove unnecessary columns and visualize the closing prices over time.
- Moving averages (MA) of 100 and 200 days are calculated to visualize trends in stock prices.

## Model Loading
- The pre-trained LSTM model (`stock_predictor.h5`) is loaded into the Streamlit application for predicting stock trends.

## Visualization
- The application allows users to enter a stock ticker symbol to visualize the closing price trends of the selected stock.
- Various charts are displayed, including the closing price vs. time chart, closing price vs. time chart with 100-day MA, and closing price vs. time chart with both 100-day and 200-day MAs.

## Prediction
- The system predicts future stock prices based on the pre-trained LSTM model.
- The predicted and original stock prices are plotted for comparison.

## Usage
To use the stock trend prediction system:
1. Run the Streamlit application script containing the code.
2. Enter the desired stock ticker symbol in the text input field.
3. The application fetches historical stock price data and displays various visualizations.
4. Predictions for future stock prices are generated and displayed.
