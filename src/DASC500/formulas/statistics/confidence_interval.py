import numpy as np
import scipy.stats as stats
import pandas as pd


def calculate_confidence_interval(data, confidence=0.95):
    """
    Compute confidence intervals for the mean and variance.

    Args:
        data (pd.Series or np.ndarray): Input data.
        confidence (float, optional): Confidence level (default: 0.95).

    Returns:
        dict: A dictionary containing the calculated statistics and confidence intervals.
              Returns None if the input data has fewer than 2 data points.

        The dictionary includes the following keys:
            "mean": Sample mean.
            "mean_CI": Confidence interval for the mean (tuple).
            "variance": Sample variance.
            "variance_CI": Confidence interval for the variance (tuple).

    Raises:
        TypeError: If the input data is not a Pandas Series or NumPy array.
    """

    if not isinstance(data, (pd.Series, np.ndarray)):
        raise TypeError("Input data must be a Pandas Series or NumPy array.")

    results = {}
    alpha = 1 - confidence
    data = data.dropna().values if isinstance(data, pd.Series) else data  # Remove NaN values if Series
    n = len(data)

    if n < 2:
        return None # Return None if not enough data

    # Sample mean and sample variance
    mean = np.mean(data)
    variance = np.var(data, ddof=1)  # Sample variance

    # Mean CI
    t_critical = stats.t.ppf(1 - alpha / 2, df=n-1)  # t critical value
    mean_margin = t_critical * (np.sqrt(variance) / np.sqrt(n))
    mean_ci = (mean - mean_margin, mean + mean_margin)

    # Variance CI
    chi2_lower = stats.chi2.ppf(alpha / 2, df=n-1)  # Lower chi-square critical value
    chi2_upper = stats.chi2.ppf(1 - alpha / 2, df=n-1)  # Upper chi-square critical value

    var_ci_lower = (n - 1) * variance / chi2_upper
    var_ci_upper = (n - 1) * variance / chi2_lower if chi2_lower > 0 else np.nan  # Prevent division by zero

    variance_ci = (var_ci_lower, var_ci_upper)

    # Store results
    results = {
        "mean": mean,
        "mean_CI": mean_ci,
        "variance": variance,
        "variance_CI": variance_ci
    }
    return results
