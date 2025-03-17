import pandas as pd
from statsmodels.tsa.ar_model import AutoReg

def AuReg_model(data):
    """
    Fits an autoregressive (AR) model to the provided data and generates predictions.

    Parameters:
    data (DataFrame): A pandas DataFrame containing the time series data. 
                      Ensure that the DataFrame has a column for the target variable.

    Returns:
    predictions (Series): A pandas Series containing the predicted values for the next 5 time steps.
    """

    model = AutoReg(data['target'], lags=1)
    model_fit = model.fit()
    predictions = model_fit.predict(start=len(data), end=len(data) + 5)  # 5 = number of time steps to predict
    return predictions
