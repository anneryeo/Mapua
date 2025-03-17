import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

def SARIMA_model(data):
    """
    Fits a SARIMA model to the provided data and generates predictions.

    Parameters:
    data (DataFrame): A pandas DataFrame containing the target variable 'target'.

    Returns:
    predictions (ndarray): An array containing the predicted values for the next 5 time steps.
    """
    model = SARIMAX(data['target'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))  # adjust order/s as needed
    model_fit = model.fit()
    predictions = model_fit.forecast(steps=5)  # 5 = time steps to predict
    return predictions
