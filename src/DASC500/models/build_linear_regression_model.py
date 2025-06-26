import pandas as pd
import statsmodels.api as sm


def build_linear_regression_model(df, response_var, predictor_vars):
    """
    Build simple linear regression models for each predictor variable.

    Args:
        df (pd.DataFrame): The dataset.
        response_var (str): The name of the response variable.
        predictor_vars (list): A list of predictor variable names.

    Returns:
        pd.DataFrame: DataFrame containing results for each predictor, including
                      beta0, beta1, t-statistic, confidence interval, and R-squared.

    Raises:
        TypeError: If df is not a Pandas DataFrame.
        ValueError: If response_var or any predictor_vars are not in the DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a Pandas DataFrame.")

    if response_var not in df.columns:
        raise ValueError(f"Response variable '{response_var}' not found in DataFrame.")

    for predictor in predictor_vars:
        if predictor not in df.columns:
            raise ValueError(f"Predictor variable '{predictor}' not found in DataFrame.")

    # Prepare table to store results
    results = []

    # Loop through each predictor and fit a simple linear regression model
    for predictor in predictor_vars:
        X = df[[predictor]]  # Predictor variable
        X = sm.add_constant(X)  # Add intercept term
        y = df[response_var]  # Response variable

        model = sm.OLS(y, X).fit()  # Fit regression model

        # Extract required values
        beta0 = model.params["const"]  # Intercept
        beta1 = model.params[predictor]  # Slope
        t_stat = model.tvalues[predictor]  # t-stat for β1
        lcl, ucl = model.conf_int().loc[predictor]  # Confidence Interval for β1
        r2 = model.rsquared  # R² value

        # Append to results list
        results.append([predictor, beta0, beta1, t_stat, lcl, ucl, r2])

    # Convert to DataFrame
    return pd.DataFrame(results, columns=["Predictor", "β0", "β1", "t-stat", "LCL", "UCL", "R2"])
