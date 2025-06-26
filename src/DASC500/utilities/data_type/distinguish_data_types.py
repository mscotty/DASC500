import os
import pandas as pd
from typing import Union, Dict

def distinguish_data_types(input):
    """!
    @brief Check a pandas dataframe for the types of data it stores.
    
    Analyzes each column in a DataFrame to determine if it contains numeric or other data types.
    Primarily focused on distinguishing between numeric (int/float) and non-numeric data.
    
    @param input: Either a pandas DataFrame or a path to a CSV file
    @return: Dictionary mapping column names to their data types ('Numeric' or specific type)
    """
    # Handle DataFrame input
    if isinstance(input, pd.DataFrame):
        df = input
    # Handle file path input
    elif isinstance(input, str):
        if os.path.exists(input):
            df = pd.read_csv(input)
        else:
            raise FileNotFoundError(f'The provided file {input} does not exist.')
    else:
        raise TypeError("Input must be a pandas DataFrame or a valid file path")
    
    # Initialize a dictionary to store the column types
    column_types = {}
    
    for column in df.columns:
        # First check if column is entirely empty
        if df[column].isna().all():
            column_types[column] = 'String'
            continue
            
        # Check if the column contains boolean values
        if pd.api.types.is_bool_dtype(df[column]):
            column_types[column] = 'String'  # Classify booleans as non-numeric
            continue
            
        # Check if all values in the column can be numeric using pandas built-in function
        if pd.api.types.is_numeric_dtype(df[column]):
            # Additional check to exclude boolean columns that might be stored as 0/1
            if not all(value in [0, 1, 0.0, 1.0] for value in df[column].dropna().unique()):
                column_types[column] = 'Numeric'
            else:
                # If all values are 0/1, check if it might be a boolean column
                if len(df[column].dropna().unique()) <= 2:
                    column_types[column] = 'String'  # Treat potential boolean as non-numeric
                else:
                    column_types[column] = 'Numeric'
        else:
            # If that built-in function fails, rely on checking each individual value
            non_na_values = df[column].dropna()
            
            # Check for string representations of booleans
            if all(isinstance(value, str) and value.lower() in ['true', 'false'] for value in non_na_values):
                column_types[column] = 'String'
            elif all(_is_numeric(value) and not _is_boolean(value) for value in non_na_values):
                column_types[column] = 'Numeric'
            else:
                column_types[column] = 'String'
    
    return column_types

def _is_numeric(value):
    """
    Check if a value is numeric or can be converted to a numeric value.
    Excludes boolean values.
    
    @param value: The value to check
    @return: True if the value is numeric or can be converted to numeric, False otherwise
    """
    # Explicitly exclude boolean values
    if isinstance(value, bool):
        return False
    
    if isinstance(value, (int, float)):
        return True
    
    if isinstance(value, str):
        # Check if it's a string representation of a boolean
        if value.lower() in ['true', 'false']:
            return False
            
        # Try to convert to float
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False
    
    return False

def _is_boolean(value):
    """
    Check if a value is a boolean or can be interpreted as a boolean.
    
    @param value: The value to check
    @return: True if the value is a boolean or can be interpreted as boolean, False otherwise
    """
    if isinstance(value, bool):
        return True
        
    if isinstance(value, (int, float)):
        return value in [0, 1, 0.0, 1.0]
        
    if isinstance(value, str):
        return value.lower() in ['true', 'false']
        
    return False
