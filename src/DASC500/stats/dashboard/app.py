import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc

# Import components
from DASC500.stats.dashboard.components.data_overview import display_dataset_overview
from DASC500.stats.dashboard.components.numeric_analysis import analyze_numeric_column
from DASC500.stats.dashboard.components.categorical_analysis import analyze_categorical_column
from DASC500.stats.dashboard.components.statistical_tests import explore_statistical_tests
from DASC500.stats.dashboard.components.transformations.basic import explore_basic_transformations
from DASC500.stats.dashboard.components.transformations.advanced import explore_advanced_transformations

# Import utilities
from DASC500.stats.dashboard.utils.data_handling import load_data, generate_sample_data

# Set page configuration
st.set_page_config(page_title="Statistical Analysis Dashboard", layout="wide")

# Add a title and description
st.title("📊 Statistical Analysis Dashboard")
st.markdown("""
This dashboard allows you to upload a CSV file and perform various statistical analyses on your data.
Select columns to analyze, visualize distributions, and run statistical tests.
""")

# Initialize session state variables
if 'show_download' not in st.session_state:
    st.session_state.show_download = False
if 'use_sample_data' not in st.session_state:
    st.session_state.use_sample_data = False
if 'sample_data' not in st.session_state:
    st.session_state.sample_data = None

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=['csv'])

def run_app():
    """Main function to run the app"""
    # Check if we should use sample data
    if st.session_state.use_sample_data and st.session_state.sample_data is not None:
        df = st.session_state.sample_data
        st.sidebar.success("Using sample data!")
        analyze_data(df)
    elif uploaded_file is not None:
        # Load data
        df, error = load_data(uploaded_file)
        if error:
            st.error(error)
            return
        
        st.sidebar.success("File successfully loaded!")
        analyze_data(df)
    else:
        # Display welcome message and sample data option
        st.write("## Welcome to the Statistical Analysis Dashboard")
        st.write("""
        This dashboard allows you to:
        - Upload and analyze your CSV data files
        - Explore descriptive statistics and visualizations
        - Perform statistical tests and transformations
        - Generate insights about your data
        
        To get started, upload a CSV file using the sidebar.
        """)
        
        if st.button("Load Sample Data"):
            with st.spinner("Generating sample data..."):
                sample_df = generate_sample_data()
                
                # Store in session state
                st.session_state.sample_data = sample_df
                st.session_state.use_sample_data = True
                
                # Display success message
                st.success("Sample data loaded successfully!")
                st.rerun()  # Use st.rerun() instead of st.experimental_rerun()

def analyze_data(df):
    """Main function to analyze the loaded data"""
    # Display basic information
    display_dataset_overview(df)
    
    # Select columns for analysis
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not numeric_columns and not categorical_columns:
        st.error("No suitable columns found for analysis. Please check your data.")
        return
    
    st.sidebar.header("Column Selection")
    all_columns = df.columns.tolist()
    selected_column = st.sidebar.selectbox("Select a column for detailed analysis:", all_columns)
    
    # Determine column type
    if selected_column in numeric_columns:
        column_type = "numeric"
    else:
        column_type = "categorical"
    
    # Display summary statistics
    st.header(f"Analysis of: {selected_column}")
    
    if column_type == "numeric":
        analyze_numeric_column(df, selected_column, numeric_columns)
    else:
        analyze_categorical_column(df, selected_column, numeric_columns, categorical_columns)
    
    # Data exploration section
    st.header("Advanced Data Exploration")
    
    exploration_tab1, exploration_tab2, exploration_tab3 = st.tabs([
        "Custom Visualization", "Correlation Matrix", "Statistical Tests"
    ])
    
    with exploration_tab1:
        explore_custom_visualization(df, numeric_columns, categorical_columns)
    
    with exploration_tab2:
        explore_correlation_matrix(df, numeric_columns)
    
    with exploration_tab3:
        explore_statistical_tests(df, numeric_columns, categorical_columns)
    
    # Data transformation section
    st.header("Data Transformation")
    
    transform_tab1, transform_tab2 = st.tabs(["Basic Transformations", "Advanced Transformations"])
    
    with transform_tab1:
        explore_basic_transformations(df, numeric_columns)
    
    with transform_tab2:
        explore_advanced_transformations(df, numeric_columns)
    
    # Download transformed data
    if st.button("Generate Transformed Dataset"):
        st.session_state.show_download = True
    
    if st.session_state.get('show_download', False):
        generate_transformed_dataset(df, numeric_columns)

# Custom visualization function (could be moved to a separate module)
def explore_custom_visualization(df, numeric_columns, categorical_columns):
    """Create custom visualizations"""
    from DASC500.stats.dashboard.components.custom_visualizations import create_custom_visualization
    create_custom_visualization(df, numeric_columns, categorical_columns)

# Correlation matrix function (could be moved to a separate module)
def explore_correlation_matrix(df, numeric_columns):
    """Explore correlation matrix"""
    from DASC500.stats.dashboard.components.correlation_analysis import analyze_correlations
    analyze_correlations(df, numeric_columns)

# Generate transformed dataset function (could be moved to a separate module)
def generate_transformed_dataset(df, numeric_columns):
    """Generate and download transformed dataset"""
    from DASC500.stats.dashboard.components.transformations.download import create_transformed_dataset
    create_transformed_dataset(df, numeric_columns)

# Run the app
if __name__ == "__main__":
    run_app()
