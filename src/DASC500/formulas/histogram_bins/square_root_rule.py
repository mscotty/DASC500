import numpy as np

def square_root_rule(data):
    """
    Compute bin width and bin count using the Square Root Rule.

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

    bin_width = (data.max() - data.min()) / np.sqrt(len(data))
    bin_count = int(np.sqrt(len(data)))
    return bin_width, bin_count
