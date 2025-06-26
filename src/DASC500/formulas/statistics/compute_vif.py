import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


def compute_vif(df, selected_vars):
    """
    Compute Variance Inflation Factor (VIF) to check for multicollinearity.

    Args:
        df (pd.DataFrame): Input DataFrame.
        selected_vars (list): List of column names to compute VIF for.

    Returns:
        pd.DataFrame: DataFrame containing the VIF for each variable.
                     Returns a DataFrame with VIF = 1.0 for each variable if
                     len(selected_vars) < 2.
    """
    if len(selected_vars) < 2:
        return pd.DataFrame({"Variable": selected_vars, "VIF": [1.0] * len(selected_vars)})  # Single variable has VIF=1
    
    X = sm.add_constant(df[selected_vars])  # Add intercept
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data[vif_data["Variable"] != "const"].reset_index(drop=True)  # Exclude intercept and reset index


def test():
    """Test function for compute_vif."""
    # ✅ **Test with a Sample Dataset**
    test_data = pd.DataFrame({
        "mpg": [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
        "cyl": [4, 4, 6, 6, 8, 8, 4, 6, 8, 4],
        "disp": [160, 160, 258, 258, 360, 360, 140, 200, 320, 180],
        "wt": [2.62, 2.88, 3.21, 3.44, 3.57, 3.78, 2.46, 3.00, 3.68, 2.80]
    })

    selected_vars = ["cyl", "disp", "wt"]
    vif_report = compute_vif(test_data, selected_vars)
    print("\n✅ **VIF Report**")
    print(vif_report)


if __name__ == "__main__":
    test()
