import streamlit as st
import pandas as pd

def display_dataset_overview(df):
    """Display basic dataset information and summary"""
    with st.expander("Dataset Overview", expanded=True):
        st.write("### Data Preview")
        st.dataframe(df.head())
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("### Dataset Shape")
            st.write(f"Rows: {df.shape[0]:,}, Columns: {df.shape[1]}")
        
        with col2:
            st.write("### Data Types")
            st.write(df.dtypes)
        
        st.write("### Missing Values")
        missing_data = pd.DataFrame({
            'Column': df.columns,
            'Missing Values': df.isnull().sum().values,
            'Percentage': (df.isnull().sum().values / len(df) * 100).round(2)
        })
        st.dataframe(missing_data)
    
    # Move this outside the first expander
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        with st.expander("Numeric Columns Summary Statistics"):
            st.dataframe(df[numeric_cols].describe().T)

