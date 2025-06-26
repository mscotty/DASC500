import numpy as np
import pandas as pd


def calculate_mean(data):
    """
    Calculate the mean of a NumPy array or Pandas Series.

    Args:
        data (np.ndarray or pd.Series): Input data.

    Returns:
        float: The mean of the data.

    Raises:
        TypeError: If the input is not a NumPy array or Pandas Series.
        ValueError: If the input contains fewer than 1 data point.
    """
    if not isinstance(data, (np.ndarray, pd.Series)):
        raise TypeError("Input must be a NumPy array or Pandas Series")

    n = len(data)
    if n == 0:  # Corrected condition: mean needs at least one data point
        raise ValueError("Mean requires at least one data point")

    return float(np.sum(data)) / n  # Explicitly use NumPy's sum and force float division
