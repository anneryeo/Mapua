import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from arch import arch_model
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error

def LinearReg_eval(X_train, y_train, X_test, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return mse

def AuReg_eval(data, lags):
    model = AutoReg(data, lags=lags)
    model_fit = model.fit()
    predictions = model_fit.predict(start=len(data), end=len(data) + 10)
    return predictions

def MovAve_eval(data, window):
    moving_avg = data.rolling(window=window).mean()
    predictions = moving_avg[-10:]
    return predictions

def ARIMA_eval(data, order):
    model = ARIMA(data, order=order)
    model_fit = model.fit()
    predictions = model_fit.forecast(steps=10)
    return predictions

def SARIMA_eval(data, order, seasonal_order):
    model = SARIMAX(data, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit()
    predictions = model_fit.forecast(steps=10)
    return predictions

def GARCH_eval(data):
    model = arch_model(data, vol='Garch', p=1, q=1)
    model_fit = model.fit()
    predictions = model_fit.forecast(horizon=10)
    return predictions.mean['h.1'].iloc[-1]

def LSTM_eval(X_train, y_train, X_test):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=200, verbose=0)
    predictions = model.predict(X_test)
    return predictions

