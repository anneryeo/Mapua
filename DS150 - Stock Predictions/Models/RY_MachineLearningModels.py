import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from arch import arch_model
import tensorflow as tf
from tensorflow import keras
from functools import lru_cache

#-------------------------------------------
''' 
LinearReg_model: A simple wrapper around the LinearRegression class for modeling and prediction.

Attributes:
    model : LinearRegression
        An instance of the LinearRegression model.

Methods:
    fit(X, y):
        Fits the LinearRegression model to the training data (X, y).
    predict(X):
        Predicts outputs based on input features X using the fitted model.
        Uses an LRU cache to optimize repeated predictions for the same input.

Example:
    Import necessary libraries
        from sklearn.linear_model import LinearRegression
        from functools import lru_cache

    Initialize the model
        lr_model = LinearReg_model()

    Prepare training data
        X_train = [[1], [2], [3]]
        y_train = [2, 4, 6]

    Fit the model
        lr_model.fit(X_train, y_train)

    Predict using the fitted model
        X_test = [[4]]
        y_pred = lr_model.predict(X_test)
        print(f"Predicted value: {y_pred[0]}")
'''

class LinearReg_model:
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

#-------------------------------------------
''' 
AuReg_model: A wrapper class for the AutoReg model to handle univariate autoregressive modeling and forecasting.

Attributes:
    model : AutoReg or None
        The AutoReg model instance after fitting. Defaults to None before fitting.

Methods:
    fit(y, lags):
        Fits an AutoReg model to the time series data `y` with the specified number of lags.
    predict(steps):
        Predicts future values for the specified number of steps ahead using the fitted AutoReg model.
        Utilizes an LRU cache to optimize repeated predictions for the same input.

Example:
    Import necessary libraries
        from statsmodels.tsa.ar_model import AutoReg
        from functools import lru_cache

    Initialize the model
        ar_model = AuReg_model()

    Prepare data
        y_train = [1, 2, 3, 4, 5]

    Fit the model with a specified number of lags
        ar_model.fit(y_train, lags=2)

    Predict the next 3 steps
        y_pred = ar_model.predict(steps=3)
        print(f"Predicted values: {y_pred}")
'''

class AuReg_model:
    def __init__(self):
        self.model = None

    def create_lagged_features(self, y, lags):
        # lag features
        lagged_data = pd.DataFrame(y)
        for lag in range(1, lags + 1):
            lagged_data[f'lag_{lag}'] = lagged_data.shift(lag)
        lagged_data.dropna(inplace=True)
        return lagged_data

    def fit(self, y, lags):
        lagged_data = self.create_lagged_features(y, lags)
        self.model = AutoReg(lagged_data[y.name], lags=lags).fit()

    def predict(self, steps):
        return self.model.predict(start=len(self.model.data.endog), end=len(self.model.data.endog) + steps - 1)

#-------------------------------------------
''' 
MovAve_model: A class for modeling and predicting using a simple moving average approach.

Attributes:
    window_size : int or None
        The size of the rolling window for the moving average. Defaults to None before fitting.
    y : pandas.Series or None
        The time series data to compute the moving average on. Defaults to None before fitting.

Methods:
    fit(y, window_size):
        Initializes the rolling window size and stores the time series data `y` for the moving average computation.
    predict():
        Calculates and returns the most recent value of the moving average using the fitted window size.
        Utilizes an LRU cache to optimize repeated predictions for the same parameters.

Example:
    Import necessary libraries
        import pandas as pd
        from functools import lru_cache

    Prepare data
        y_train = pd.Series([1, 2, 3, 4, 5])

    Initialize the model
        ma_model = MovAve_model()

    Fit the model with the series and window size
        ma_model.fit(y_train, window_size=3)

    Predict the most recent moving average
        y_pred = ma_model.predict()
        print(f"Predicted moving average: {y_pred}")
'''

class MovAve_model:
    def __init__(self):
        self.window_size = None
        self.y = None

    def fit(self, y, window_size):
        self.window_size = window_size
        self.y = y

    #@lru_cache(maxsize=128)
    def predict(self):
        return self.y.rolling(window=self.window_size).mean().iloc[-1]

#-------------------------------------------
''' 
ARIMA_model: A class for modeling and forecasting time series data using the ARIMA (AutoRegressive Integrated Moving Average) approach.

Attributes:
    model : ARIMA or None
        The ARIMA model instance after fitting. Defaults to None before fitting.

Methods:
    fit(y, order):
        Fits an ARIMA model to the time series data `y` with the specified order (p, d, q).
    predict(steps):
        Forecasts future values for the specified number of steps ahead using the fitted ARIMA model.
        Utilizes an LRU cache to optimize repeated predictions for the same input.

Example:
    Import necessary libraries
        from statsmodels.tsa.arima.model import ARIMA
        from functools import lru_cache

    Prepare data
        y_train = [1, 2, 3, 4, 5]

    Initialize the model
        arima_model = ARIMA_model()

    Fit the model with the time series data and ARIMA order (p, d, q)
        arima_model.fit(y_train, order=(1, 1, 0))

    Predict the next 3 steps
        y_pred = arima_model.predict(steps=3)
        print(f"Predicted values: {y_pred}")
'''

class ARIMA_model:
    def __init__(self):
        self.model = None

    def fit(self, y, order):
        self.model = ARIMA(y, order=order).fit()

    #@lru_cache(maxsize=128)
    def predict(self, steps):
        return self.model.forecast(steps)

#-------------------------------------------
''' 
SARIMA_model: A class for modeling and forecasting time series data using the SARIMA (Seasonal AutoRegressive Integrated Moving Average) approach.

Attributes:
    model : SARIMAX or None
        The SARIMAX model instance after fitting. Defaults to None before fitting.

Methods:
    fit(y, order, seasonal_order):
        Fits a SARIMAX model to the time series data `y` with the specified order (p, d, q) and seasonal order (P, D, Q, m).
    predict(steps):
        Forecasts future values for the specified number of steps ahead using the fitted SARIMAX model.
        Utilizes an LRU cache to optimize repeated predictions for the same input.

Example:
    Import necessary libraries
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        from functools import lru_cache

    Prepare data
        y_train = [1, 2, 3, 4, 5]

    Initialize the model
        sarima_model = SARIMA_model()

    Fit the model with the time series data, order (p, d, q), and seasonal order (P, D, Q, m)
        sarima_model.fit(y_train, order=(1, 1, 0), seasonal_order=(1, 1, 0, 12))

    Predict the next 3 steps
        y_pred = sarima_model.predict(steps=3)
        print(f"Predicted values: {y_pred}")
'''

class SARIMA_model:
    def __init__(self):
        self.model = None

    def fit(self, y, order, seasonal_order):
        self.model = SARIMAX(y, order=order, seasonal_order=seasonal_order).fit()

    #@lru_cache(maxsize=128)
    def predict(self, steps):
        return self.model.forecast(steps)

#-------------------------------------------
''' 
GARCH_model: A class for modeling and forecasting time series volatility using the GARCH (Generalized AutoRegressive Conditional Heteroskedasticity) approach.

Attributes:
    model : arch.univariate.base.ARCHModelResult or None
        The GARCH model instance after fitting. Defaults to None before fitting.

Methods:
    fit(y):
        Fits a GARCH(1, 1) model to the provided time series data `y` using conditional heteroskedasticity.
    predict(steps):
        Forecasts volatility for the specified number of steps ahead using the fitted GARCH model.
        Utilizes an LRU cache to optimize repeated predictions for the same input.

Example:
    Import necessary libraries
        from arch import arch_model
        from functools import lru_cache

    Prepare data
        y_train = [0.1, 0.2, -0.1, 0.3, 0.4]

    Initialize the model
        garch_model = GARCH_model()

    Fit the model with the time series data
        garch_model.fit(y_train)

    Predict the next 3 steps of volatility
        volatility_pred = garch_model.predict(steps=3)
        print(f"Predicted volatility: {volatility_pred.variance}")
'''

class GARCH_model:
    def __init__(self):
        self.model = None

    def fit(self, y):
        self.model = arch_model(y, vol='Garch', p=1, q=1).fit()

    #@lru_cache(maxsize=128)
    def predict(self, steps):
        return self.model.forecast(horizon=steps)

#-------------------------------------------
''' 
LSTM_model: A class for building and training an LSTM-based neural network for time series prediction or sequence modeling tasks.

Attributes:
    model : keras.Sequential
        The LSTM model instance created during initialization.

Methods:
    build_model(input_shape):
        Constructs the LSTM model with a specified input shape, one LSTM layer with 50 units, and a Dense output layer.
        Uses the Adam optimizer and Mean Squared Error (MSE) loss function.
    fit(X, y, epochs=100, batch_size=32):
        Trains the LSTM model on the input data `X` and target data `y` for the given number of epochs and batch size.
    predict(X):
        Predicts the output for the input data `X` using the trained LSTM model.
        Utilizes an LRU cache to optimize repeated predictions for the same input.

Example:
    Import necessary libraries
        from tensorflow import keras
        from functools import lru_cache
        import numpy as np

    Define input shape (e.g., timesteps and features)
        input_shape = (10, 1)

    Initialize the model
        lstm_model = LSTM_model(input_shape=input_shape)

    Generate sample data
        X_train = np.random.rand(100, 10, 1) # 100 samples, 10 timesteps, 1 feature
        y_train = np.random.rand(100, 1)     # 100 target values

    Train the model
        lstm_model.fit(X_train, y_train, epochs=50, batch_size=16)

    Predict on new data
        X_test = np.random.rand(1, 10, 1)    # Single test sample
        y_pred = lstm_model.predict(X_test)
        print(f"Predicted value: {y_pred}")
'''

class LSTM_model:
    def __init__(self, input_shape):
        self.model = Sequential()
        self.model.add(LSTM(50, activation='relu', input_shape=input_shape))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mse')

    def fit(self, X_train, y_train, epochs=200):
        self.model.fit(X_train, y_train, epochs=epochs, verbose=0)

    def predict(self, X_test):
        return self.model.predict(X_test)
#-------------------------------------------