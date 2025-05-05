import numpy as np

def mae(y_true, y_pred):
    """
    Mean Absolute Error (MAE)

    Args:
        y_true (np.ndarray): The ground truth values.
        y_pred (np.ndarray): The predicted values.

    Returns:
        float: The average absolute difference between y_true and y_pred.
    """
    return np.mean(np.abs(y_true - y_pred))

def mse(y_true, y_pred):
    """
    Mean Squared Error (MSE)

    Args:
        y_true (np.ndarray): The ground truth values.
        y_pred (np.ndarray): The predicted values.

    Returns:
        float: The average of the squared differences between y_true and y_pred.
    """
    return np.mean((y_true - y_pred) ** 2)

def rmse(y_true, y_pred):
    """
    Root Mean Squared Error (RMSE)

    Args:
        y_true (np.ndarray): The ground truth values.
        y_pred (np.ndarray): The predicted values.

    Returns:
        float: The square root of the MSE.
    """
    return np.sqrt(mse(y_true, y_pred))

def mape(y_true, y_pred):
    """
    Mean Absolute Percentage Error (MAPE)

    Args:
        y_true (np.ndarray): The ground truth values.
        y_pred (np.ndarray): The predicted values.

    Returns:
        float: The average absolute percentage error between y_true and y_pred, multiplied by 100.
    """
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def mspe(y_true, y_pred):
    """
    Mean Squared Percentage Error (MSPE)

    Args:
        y_true (np.ndarray): The ground truth values.
        y_pred (np.ndarray): The predicted values.

    Returns:
        float: The average squared percentage error between y_true and y_pred, multiplied by 100.
    """
    return np.mean(((y_true - y_pred) / y_true) ** 2) * 100

def rse(y_true, y_pred):
    """
    Relative Squared Error (RSE)

    Args:
        y_true (np.ndarray): The ground truth values.
        y_pred (np.ndarray): The predicted values.

    Returns:
        float: The ratio of the root of the sum of squared errors 
               to the root of the sum of squared deviations from the mean of y_true.
    """
    numerator = np.sqrt(np.sum((y_true - y_pred) ** 2))
    denominator = np.sqrt(np.sum((y_true - np.mean(y_true)) ** 2))
    return numerator / denominator

def corr(y_true, y_pred):
    """
    Correlation metric

    Computes a scaled correlation between y_true and y_pred.

    Args:
        y_true (np.ndarray): The ground truth values.
        y_pred (np.ndarray): The predicted values.

    Returns:
        float: A scaled version (multiplied by 0.01) of the correlation coefficient
               between y_true and y_pred.
    """
    # Compute the numerator: covariance-like term
    u = np.sum((y_true - np.mean(y_true)) * (y_pred - np.mean(y_pred)))
    
    # Compute the denominator: product of standard deviations
    d = np.sqrt(np.sum((y_true - np.mean(y_true)) ** 2 * (y_pred - (np.mean(y_pred)) ** 2)))
    
    # To avoid division by zero
    d += 1e-12

    # Return the scaled mean correlation
    return 0.01 * (u / d)

def calculate_metrics(y_true, y_pred):
    """
    Calculate multiple metrics (MAE, MSE, RMSE, MAPE, MSPE, RSE, CORR) for given predictions.

    Args:
        y_true (np.ndarray): The ground truth values.
        y_pred (np.ndarray): The predicted values.

    Returns:
        dict: A dictionary mapping metric names to their computed values.
    """
    metrics_dict = {
        'mae': mae(y_true, y_pred),
        'mse': mse(y_true, y_pred),
        'rmse': rmse(y_true, y_pred),
        'mape': mape(y_true, y_pred),
        'mspe': mspe(y_true, y_pred),
        'rse': rse(y_true, y_pred),
        'corr': corr(y_true, y_pred),
    }
    return metrics_dict
