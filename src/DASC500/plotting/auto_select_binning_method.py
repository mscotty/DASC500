import numpy as np

def auto_select_binning_method(data):
    """
    Automatically select a binning method based on data characteristics.

    Args:
        data (np.ndarray): Numeric data for determining the binning method.

    Returns:
        str: The name of the selected binning method
             ("Freedman-Diaconis", "Square Root", "Sturges", "Scott's", or "Rice").

    Raises:
        TypeError: If input data is not a NumPy array.
        ValueError: If input data is empty.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a NumPy array.")
    if data.size == 0:
        raise ValueError("Input data must not be empty.")


    iqr = np.percentile(data, 75) - np.percentile(data, 25)
    data_range = data.max() - data.min()
    std_dev = np.std(data)

    if iqr > 0.1 * data_range:
        return "Freedman-Diaconis"
    elif std_dev < 0.05 * data_range:  # Adjusted condition using standard deviation
        return "Rice"
    elif len(data) < 50:
        return "Square Root"
    elif len(data) > 500:
        return "Sturges"
    else:
        return "Scott's"
