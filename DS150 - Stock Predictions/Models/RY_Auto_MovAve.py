import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def MovAve_model(data):
    """
    Fits a moving average (MA) model to the provided data and generates predictions.

    Parameters:
    data (DataFrame): A pandas DataFrame containing the target variable 'targ'.

    Returns:
    predictions (ndarray): An array containing the predicted values for the next 5 time steps.
    """
    model = ARIMA(data['targ'], order=(0, 0, 1))  # adjust order if needed
    model_fit = model.fit()
    predictions = model_fit.forecast(steps=5)  # 5 = time steps to predict
    return predictions
