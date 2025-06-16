import streamlit as st
import pandas as pd
import numpy as np

def load_data(uploaded_file):
    """Load data from uploaded file with error handling"""
    try:
        with st.spinner("Loading data..."):
            df = pd.read_csv(uploaded_file)
        return df, None
    except Exception as e:
        return None, f"Error loading the data: {str(e)}"

def generate_sample_data(sample_size=1000, random_seed=42):
    """Generate sample data for demonstration"""
    np.random.seed(random_seed)
    
    # Create synthetic dataset
    data = {
        'Age': np.random.normal(35, 12, sample_size).astype(int),
        'Income': np.random.lognormal(10, 1, sample_size).astype(int),
        'Experience': np.random.normal(10, 7, sample_size).astype(int),
        'Satisfaction': np.random.randint(1, 11, sample_size),
        'Department': np.random.choice(['Sales', 'Marketing', 'HR', 'Engineering', 'Finance'], sample_size),
        'Gender': np.random.choice(['Male', 'Female'], sample_size),
        'Performance': np.random.normal(7, 1.5, sample_size).round(1).clip(1, 10)
    }
    
    # Add some correlations
    data['Salary'] = data['Experience'] * 5000 + data['Performance'] * 2000 + np.random.normal(50000, 10000, sample_size)
    data['Salary'] = data['Salary'].astype(int)
    
    # Create education levels with some correlation to income
    education_mapping = {0: 'High School', 1: 'Bachelor', 2: 'Master', 3: 'PhD'}
    education_prob = np.clip(data['Income'] / data['Income'].max(), 0.1, 0.9)
    data['Education'] = [education_mapping[min(3, np.random.binomial(3, p))] for p in education_prob]
    
    # Create a binary target variable
    data['Promotion'] = (data['Performance'] > 8).astype(int)
    
    # Create DataFrame
    sample_df = pd.DataFrame(data)
    
    return sample_df
