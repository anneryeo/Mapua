import pandas as pd
from arch import arch_model

def GARCH_model(data):
    """
    Fits a GARCH model to the provided data and generates predictions.

    Parameters:
    data (DataFrame): A pandas DataFrame containing the target variable 'target'.

    Returns:
    predictions (float): The forecasted mean value for the next 5 time steps.
    """
    model = arch_model(data['target'], vol='Garch', p=1, q=1)  # adjust p and q as needed
    model_fit = model.fit()
    predictions = model_fit.forecast(horizon=5)  # 5 = time steps
    return predictions.mean['h.1'].iloc[-1]  # return the forecasted mean