import statsmodels.api as sm
import pandas as pd
# Assuming compute_vif is in the same directory or accessible in PYTHONPATH
from DASC500.formulas.statistics.compute_vif import compute_vif


def build_stepwise_parsimonious_regression_model(
    df,
    target_var,
    predictors,
    threshold_in=0.05,
    threshold_out=0.10,
    vif_threshold=5.0,
):
    """
    Perform stepwise regression with added options:
    - Multicollinearity check (VIF)
    - AIC/BIC model selection
    - Manual variable removal option

    Args:
        df (pd.DataFrame): The dataset containing the target and predictors.
        target_var (str): The name of the response variable.
        predictors (list): List of potential predictor variables.
        threshold_in (float, optional): Max p-value for adding a predictor (default: 0.05).
        threshold_out (float, optional): Min p-value for removing a predictor (default: 0.10).
        vif_threshold (float, optional): Maximum allowable VIF value for a variable (default: 5.0).

    Returns:
        tuple: (final_model, selected_vars, vif_data)
            final_model (sm.OLS): The final fitted regression model.
            selected_vars (list): List of selected predictor variables.
            vif_data (pd.DataFrame): DataFrame containing VIF values for selected variables.

    Raises:
        TypeError: If df is not a Pandas DataFrame.
        ValueError: If response_var or any predictor_vars are not in the DataFrame.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a Pandas DataFrame.")

    if target_var not in df.columns:
        raise ValueError(f"Target variable '{target_var}' not found in DataFrame.")

    for predictor in predictors:
        if predictor not in df.columns:
            raise ValueError(f"Predictor variable '{predictor}' not found in DataFrame.")

    selected_vars = []
    remaining_vars = set(predictors)

    while remaining_vars:
        # Forward Selection: Try adding a new variable
        new_pval = pd.Series(index=remaining_vars, dtype=float)
        for var in remaining_vars:
            model = sm.OLS(
                df[target_var], sm.add_constant(df[selected_vars + [var]])
            ).fit()
            new_pval[var] = model.pvalues[var]

        # Select the best new variable
        min_pval = new_pval.min()
        if min_pval < threshold_in:
            best_var = new_pval.idxmin()
            selected_vars.append(best_var)
            remaining_vars.remove(best_var)
        else:
            break  # No new variables meet the threshold

        # Backward Elimination: Remove the worst variable
        while selected_vars:
            model = sm.OLS(
                df[target_var], sm.add_constant(df[selected_vars])
            ).fit()
            max_pval = model.pvalues.iloc[1:].max()  # Ignore intercept
            if max_pval > threshold_out:
                worst_var = model.pvalues.iloc[1:].idxmax()
                selected_vars.remove(worst_var)
            else:
                break  # No variables to remove

        # Multicollinearity Check: Remove variables with high VIF
        vif_data = compute_vif(df, selected_vars)
        high_vif_vars = vif_data[vif_data["VIF"] > vif_threshold][
            "Variable"
        ].tolist()
        if high_vif_vars:
            print(f"⚠️ High VIF detected! Removing: {high_vif_vars}")
            for var in high_vif_vars:
                selected_vars.remove(var)

    # Fit the final model
    final_model = sm.OLS(
        df[target_var], sm.add_constant(df[selected_vars])
    ).fit()

    # Compare AIC and BIC with full model
    full_model = sm.OLS(
        df[target_var], sm.add_constant(df[predictors])
    ).fit()
    print("\n🔍 **Model Selection Summary**")
    print(
        f"📊 Full Model AIC: {full_model.aic:.2f}, BIC: {full_model.bic:.2f}"
    )
    print(
        f"📉 Parsimonious Model AIC: {final_model.aic:.2f}, BIC: {final_model.bic:.2f}"
    )

    # Show the final model summary
    print("\n📌 **Final Parsimonious Model Summary**")
    print(final_model.summary())

    return final_model, selected_vars, vif_data
