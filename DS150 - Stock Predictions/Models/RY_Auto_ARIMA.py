import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def ARIMA_model(data):
    """
    Fits an ARIMA model to the provided data and generates predictions.

    Parameters:
    data (DataFrame): A pandas DataFrame containing the target variable 'target'.

    Returns:
    predictions (ndarray): An array containing the predicted values for the next 5 time steps.
    """
    model = ARIMA(data['target'], order=(1, 1, 1))  # Adjust order as needed
    model_fit = model.fit()
    predictions = model_fit.forecast(steps=5)  # Predict next 5 days
    return predictions