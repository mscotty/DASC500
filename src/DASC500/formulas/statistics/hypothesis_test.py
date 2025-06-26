import numpy as np
import scipy.stats as stats
import pandas as pd # Import pandas

def hypothesis_test(sample, mu_0=30.55, alpha=0.05):
    """
    Conduct a two-tailed t-test for the sample mean.

    Args:
        sample (np.ndarray or pd.Series): Sample data.
        mu_0 (float, optional): Hypothesized population mean (default: 30.55).
        alpha (float, optional): Significance level (default: 0.05).

    Returns:
        dict: A dictionary containing the test results.
              Returns a string message if there is insufficient data.

        The dictionary includes the following keys:
            "Test Statistic (W)": Test statistic.
            "Critical Value": Critical value for the t-distribution.
            "p-value": p-value of the test.
            "Conclusion": Conclusion of the hypothesis test.
    """
    n = len(sample)
    if n < 2:
        return "Not enough data for hypothesis test"

    x_bar = np.mean(sample)  # Sample mean
    s = np.std(sample, ddof=1)  # Sample standard deviation

    # Compute test statistic W
    W = (x_bar - mu_0) / (s / np.sqrt(n))

    # Critical value for two-tailed test
    t_critical = stats.t.ppf(1 - alpha / 2, df=n-1)

    # Compute p-value
    p_value = 2 * (1 - stats.t.cdf(abs(W), df=n-1))

    # Conclusion
    if abs(W) > t_critical:
        conclusion = "Reject H0: There is significant evidence that μ ≠ 30.55"
    else:
        conclusion = "Fail to reject H0: There is not enough evidence to say μ ≠ 30.55"

    return {
        "Test Statistic (W)": W,
        "Critical Value": t_critical,
        "p-value": p_value,
        "Conclusion": conclusion
    }
