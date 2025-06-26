import numpy as np

def scotts_rule(data):
    """
    Compute bin width and bin count using Scott's Rule.

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

    sigma = np.std(data)
    bin_width = 3.5 * sigma / (len(data) ** (1/3))
    bin_count = max(1, int((data.max() - data.min()) / bin_width)) # Ensure at least 1 bin
    return bin_width, bin_count
