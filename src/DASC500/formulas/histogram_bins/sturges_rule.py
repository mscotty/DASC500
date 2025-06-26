import numpy as np

def sturges_rule(data):
    """
    Compute bin width and bin count using Sturges' Rule.

    Args:
        data (np.ndarray): Numeric data for histogram binning.

    Returns:
        tuple: (bin_width, bin_count)

    Raises:
        TypeError: If input data is not a NumPy array.
        ValueError: If input data is empty.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a NumPy array.")
    if data.size == 0:
        raise ValueError("Input data must not be empty.")

    bin_count = int(np.ceil(np.log2(len(data))) + 1)
    bin_width = (np.max(data) - np.min(data)) / bin_count
    return bin_width, bin_count
