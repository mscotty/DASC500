import pandas as pd
import numpy as np
from typing import Dict, Union, List, Tuple, Optional


def distinguish_data_types(df: pd.DataFrame) -> Dict[str, str]:
    """
    Analyze a DataFrame to categorize columns into specific data types.
    
    This function examines each column and categorizes it as:
    - 'Numeric': For continuous numerical data
    - 'Categorical': For nominal categorical data with limited unique values
    - 'Ordinal': For ordered categorical data
    - 'DateTime': For date/time data
    - 'Binary': For binary/boolean data (including 0/1 or True/False)
    - 'Text': For free-form text data
    
    Args:
        df: pandas DataFrame to analyze
        
    Returns:
        Dict mapping column names to their identified data type
    """
    column_types = {}
    
    for column in df.columns:
        # Skip columns that are entirely empty
        if df[column].isna().all():
            column_types[column] = 'Text'
            continue
        
        # Get non-null values for analysis
        non_null_values = df[column].dropna()
        
        # Identify if the column is already numeric
        if pd.api.types.is_numeric_dtype(df[column]):
            # Check if binary (0/1)
            unique_values = set(non_null_values.unique())
            if len(unique_values) <= 2 and unique_values.issubset({0, 1, 0.0, 1.0, True, False}):
                column_types[column] = 'Binary'
            # Check if categorical (few unique values compared to total)
            elif len(unique_values) <= min(10, len(non_null_values) * 0.05):
                column_types[column] = 'Categorical'
            else:
                column_types[column] = 'Numeric'
        
        # Check for datetime
        elif pd.api.types.is_datetime64_dtype(df[column]) or _is_convertible_to_datetime(non_null_values):
            column_types[column] = 'DateTime'
        
        # Check for boolean or binary string values
        elif all(isinstance(v, (bool)) or (isinstance(v, str) and v.lower() in ['true', 'false', 'yes', 'no', 't', 'f', 'y', 'n']) for v in non_null_values):
            column_types[column] = 'Binary'
        
        # Check for potential categorical (few unique values)
        elif len(non_null_values.unique()) <= min(20, len(non_null_values) * 0.1):
            column_types[column] = 'Categorical'
        
        # Default to text for everything else
        else:
            column_types[column] = 'Text'
    
    return column_types


def _is_convertible_to_datetime(series: pd.Series) -> bool:
    """Check if a series can be converted to datetime format"""
    try:
        pd.to_datetime(series, errors='raise')
        return True
    except (ValueError, TypeError):
        return False