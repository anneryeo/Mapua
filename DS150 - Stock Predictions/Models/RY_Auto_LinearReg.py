import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def LinearReg_model(data):
    """
    Fits a linear regression model to the provided data and generates predictions.

    Parameters:
    data (DataFrame): A pandas DataFrame containing the features 'feat1', 'feat2', and the target variable 'targ'.

    Returns:
    predictions (ndarray): An array containing the predicted values for the test set.
    """
    X = data[['feature1', 'feature2']]
    y = data['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    return predictions
