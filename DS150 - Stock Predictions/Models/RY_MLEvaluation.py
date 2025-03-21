import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_models(y_true, predictions):
    results = {}
    
    for model_name, preds in predictions.items():
        mae = mean_absolute_error(y_true, preds)
        mse = mean_squared_error(y_true, preds)
        rmse = np.sqrt(mse)
        
        results[model_name] = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse
        }
    
    return results

def best_model(results):
    # Find the model with the lowest RMSE
    best = min(results, key=lambda x: results[x]['RMSE'])
    return best, results[best]