import numpy as np
import pandas as pd


def calculate_sample_variance(data):
    """
    Calculate the sample variance of a NumPy array or Pandas Series.

    Args:
        data (np.ndarray or pd.Series): Input data.

    Returns:
        float: The sample variance of the data.

    Raises:
        TypeError: If the input is not a NumPy array or Pandas Series.
        ValueError: If the input contains fewer than two data points.
    """
    if not isinstance(data, (np.ndarray, pd.Series)):
        raise TypeError("Input must be a NumPy array or Pandas Series")

    n = len(data)
    if n < 2:
        raise ValueError("Sample variance requires at least two data points")

    mean = float(np.sum(data)) / n  # Explicitly use NumPy's sum and force float division
    return float(np.sum((x - mean) ** 2 for x in data)) / (n - 1)  # Sample variance formula
