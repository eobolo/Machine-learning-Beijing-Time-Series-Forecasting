import numpy as np

def calculate_rmse(y_true, y_pred):
    """
    Calculate Root Mean Squared Error (RMSE) for evaluation or submission.
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))