import statsmodels.api as sm
import pandas as pd


def build_multiple_linear_regression_model(df, response_var, predictor_vars, display_summary=True):
    """
    Fit a multiple linear regression model and return the model summary.

    Args:
        df (pd.DataFrame): The dataset containing the target and predictor variables.
        response_var (str): The name of the response variable (dependent variable).
        predictor_vars (list): A list of predictor variable names (independent variables).
        display_summary (bool, optional): Whether to print the model summary (default: True).

    Returns:
        sm.OLS: The fitted regression model.

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

    # Ensure all predictor_vars exist in the DataFrame
    X = df[predictor_vars]

    # Add intercept term
    X = sm.add_constant(X)

    # Define response variable
    y = df[response_var]

    # Fit multiple linear regression model
    model = sm.OLS(y, X).fit()

    # Display or return the summary
    if display_summary:
        print(model.summary())

    return model  # Returning the model object for further use
