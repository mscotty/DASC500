import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from scipy import stats
from sklearn import preprocessing
import traceback

from ..utils.statistics import safe_statistical_test, handle_missing_values

def apply_box_cox_transformation(data):
    """Apply Box-Cox transformation to data"""
    # Box-Cox requires positive data
    min_val = data.min()
    if min_val <= 0:
        offset = abs(min_val) + 1
        info_msg = f"Adding {offset} to all values before Box-Cox transformation to handle non-positive values."
        data = data + offset
    else:
        info_msg = None
    
    transformed_data, lambda_value = stats.boxcox(data.flatten())
    lambda_msg = f"Optimal λ (lambda) value: {lambda_value:.4f}"
    
    return transformed_data, info_msg, lambda_msg

def apply_yeo_johnson_transformation(data):
    """Apply Yeo-Johnson transformation to data"""
    pt = preprocessing.PowerTransformer(method='yeo-johnson')
    transformed_data = pt.fit_transform(data).flatten()
    lambda_msg = f"Optimal λ (lambda) value: {pt.lambdas_[0]:.4f}"
    
    return transformed_data, None, lambda_msg

def apply_quantile_transformation(data, output_dist="normal"):
    """Apply quantile transformation to data"""
    qt = preprocessing.QuantileTransformer(output_distribution=output_dist, random_state=42)
    transformed_data = qt.fit_transform(data).flatten()
    
    return transformed_data, None, None

def apply_power_transformation(data, power=1.0):
    """Apply power transformation to data"""
    if power == 0:
        transformed_data = np.log(data.flatten())
        info_msg = "Power = 0 corresponds to log transformation"
    else:
        transformed_data = np.power(data.flatten(), power)
        info_msg = None
    
    return transformed_data, info_msg, None

def apply_robust_scaling(data):
    """Apply robust scaling to data"""
    rs = preprocessing.RobustScaler()
    transformed_data = rs.fit_transform(data).flatten()
    info_msg = "Scaled using median and interquartile range (robust to outliers)"
    
    return transformed_data, info_msg, None

def show_original_data(df, column):
    """Display original data distribution"""
    st.write("#### Original Data")
    fig = px.histogram(
        df,
        x=column,
        title=f"Original Distribution of {column}"
    )
    st.plotly_chart(fig, use_container_width=True)

def show_transformed_data(transformed_data, title):
    """Display transformed data distribution"""
    st.write("#### Transformed Data")
    fig = px.histogram(
        transformed_data,
        title=title
    )
    st.plotly_chart(fig, use_container_width=True)

def check_normality(transformed_data):
    """Check normality of transformed data"""
    if len(transformed_data) > 3:  # Minimum sample size for Shapiro-Wilk test
        sample_size = min(5000, len(transformed_data))
        sample = np.random.choice(transformed_data, size=sample_size, replace=False)
        
        shapiro_result, error = safe_statistical_test(
            stats.shapiro, sample, test_name="Shapiro-Wilk test"
        )
        
        if error:
            return error, None
        else:
            shapiro_stat, shapiro_p = shapiro_result
            result = f"Shapiro-Wilk Test: statistic={shapiro_stat:.4f}, p-value={shapiro_p:.4f}"
            
            if shapiro_p < 0.05:
                conclusion = "The transformed data is still not normally distributed (p < 0.05)."
            else:
                conclusion = "The transformed data appears to be normally distributed (p >= 0.05)."
                
            return None, (result, conclusion)
    else:
        return "Not enough data points for normality test (need at least 3).", None

def show_qq_plot(data, title):
    """Display Q-Q plot for data"""
    try:
        fig, ax = plt.subplots(figsize=(8, 4))
        stats.probplot(data, plot=ax)
        ax.set_title(title)
        st.pyplot(fig)
        plt.close(fig)  # Close the figure to free memory
        return None
    except Exception as e:
        return f"Could not generate Q-Q plot: {str(e)}"

def compare_distributions(original_data, transformed_data):
    """Compare original and transformed distributions"""
    compare_df = pd.DataFrame({
        'Original': original_data,
        'Transformed': transformed_data
    })
    
    # Standardize both for better comparison
    compare_df_std = pd.DataFrame({
        'Original (Standardized)': (compare_df['Original'] - compare_df['Original'].mean()) / compare_df['Original'].std(),
        'Transformed (Standardized)': (compare_df['Transformed'] - compare_df['Transformed'].mean()) / compare_df['Transformed'].std()
    })
    
    fig = px.histogram(compare_df_std, 
                       barmode='overlay', 
                       opacity=0.7,
                       title="Comparison of Standardized Distributions")
    st.plotly_chart(fig, use_container_width=True)
    
    # Show statistics comparison
    st.write("#### Statistics Comparison")
    stats_comparison = pd.DataFrame({
        'Statistic': ['Mean', 'Median', 'Std Dev', 'Skewness', 'Kurtosis'],
        'Original': [
            original_data.mean(),
            np.median(original_data),
            original_data.std(),
            stats.skew(original_data),
            stats.kurtosis(original_data)
        ],
        'Transformed': [
            np.mean(transformed_data),
            np.median(transformed_data),
            np.std(transformed_data),
            stats.skew(transformed_data),
            stats.kurtosis(transformed_data)
        ]
    })
    
    st.dataframe(stats_comparison.style.format({
        'Original': '{:.4f}',
        'Transformed': '{:.4f}'
    }))

def explore_advanced_transformations(df, numeric_columns):
    """Explore advanced data transformations"""
    st.subheader("Advanced Transformations")
    
    if not numeric_columns:
        st.warning("No numeric columns available for transformation.")
        return
        
    advanced_transform = st.selectbox(
        "Select advanced transformation:",
        ["Box-Cox Transformation", "Yeo-Johnson Transformation", "Quantile Transformation", 
         "Power Transformation", "Robust Scaling"]
    )
    
    transform_col = st.selectbox("Select column to transform:", numeric_columns, key="adv_transform_col")
    
    if advanced_transform:
        # Check for missing values
        data_no_na, warning = handle_missing_values(df, transform_col, "transformation")
        if data_no_na is None:
            st.error(warning)
            return
        if warning:
            st.warning(warning)
            
        col1, col2 = st.columns(2)
        
        with col1:
            show_original_data(df, transform_col)
        
        with col2:
            st.write("#### Transformed Data")
            
            # Get data and handle NaNs
            data = data_no_na.values.reshape(-1, 1)
            
            try:
                # Apply the selected transformation
                if advanced_transform == "Box-Cox Transformation":
                    transformed_data, info_msg, lambda_msg = apply_box_cox_transformation(data)
                
                elif advanced_transform == "Yeo-Johnson Transformation":
                    transformed_data, info_msg, lambda_msg = apply_yeo_johnson_transformation(data)
                
                elif advanced_transform == "Quantile Transformation":
                    output_dist = st.radio("Output distribution:", ["normal", "uniform"], horizontal=True)
                    transformed_data, info_msg, lambda_msg = apply_quantile_transformation(data, output_dist)
                
                elif advanced_transform == "Power Transformation":
                    power = st.slider("Power value:", -3.0, 3.0, 1.0, 0.1)
                    transformed_data, info_msg, lambda_msg = apply_power_transformation(data, power)
                
                elif advanced_transform == "Robust Scaling":
                    transformed_data, info_msg, lambda_msg = apply_robust_scaling(data)
                
                # Display information messages if any
                if info_msg:
                    st.info(info_msg)
                if lambda_msg:
                    st.info(lambda_msg)
                
                # Plot transformed data
                show_transformed_data(transformed_data, f"{advanced_transform} of {transform_col}")
                
                # Check normality of transformed data
                error, normality_results = check_normality(transformed_data)
                if error:
                    st.warning(error)
                elif normality_results:
                    result, conclusion = normality_results
                    st.write(result)
                    st.write(conclusion)
                
                # Add Q-Q plot for transformed data
                if st.checkbox("Show Q-Q plot of transformed data"):
                    error = show_qq_plot(transformed_data, f"Q-Q Plot for Transformed {transform_col}")
                    if error:
                        st.warning(error)
                
                # Compare original and transformed data
                if st.checkbox("Compare original and transformed distributions"):
                    compare_distributions(data_no_na, transformed_data)
            
            except Exception as e:
                st.error(f"Error during transformation: {str(e)}")
                st.error(f"Details: {type(e).__name__}")
                
                # More detailed error information for debugging
                st.error(f"Traceback: {traceback.format_exc()}")
