import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

def build_lasso_regression_model(df, response_var, predictor_vars, alpha=1.0, test_size=0.2, random_state=42):
    """
    Build a Lasso Regression model.

    Args:
        df (pd.DataFrame): The dataset.
        response_var (str): The name of the response variable.
        predictor_vars (list): A list of predictor variable names.
        alpha (float, optional): Regularization strength (default: 1.0).
        test_size (float, optional): Proportion of data for testing (default: 0.2).
        random_state (int, optional): Random seed for reproducibility (default: 42).

    Returns:
        tuple: (model, mse)
            model: The fitted Lasso regression model.
            mse (float): Mean Squared Error on the test set.

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

    # Prepare data
    X = df[predictor_vars]
    y = df[response_var]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build and train the Lasso Regression model
    model = Lasso(alpha=alpha)
    model.fit(X_train_scaled, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test_scaled)

    # Calculate Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)

    print(f"Lasso Regression Mean Squared Error: {mse:.4f}")

    return model, mse
