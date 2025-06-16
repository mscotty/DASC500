import numpy as np
import pandas as pd
from scipy import stats
from sklearn import preprocessing

def safe_statistical_test(test_func, data, *args, test_name="Statistical test", **kwargs):
    """Helper function to safely run statistical tests with error handling"""
    try:
        result = test_func(data, *args, **kwargs)
        return result, None
    except Exception as e:
        return None, f"{test_name} failed: {str(e)}"

def handle_missing_values(df, column, operation_name="analysis"):
    """Helper function to handle and report on missing values"""
    data_no_na = df[column].dropna()
    if len(data_no_na) == 0:
        return None, f"Column '{column}' contains only missing values."
    elif len(data_no_na) < len(df[column]):
        warning = f"Column '{column}' contains {len(df[column]) - len(data_no_na):,} missing values that were excluded from {operation_name}."
        return data_no_na, warning
    return data_no_na, None
