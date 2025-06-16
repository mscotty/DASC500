import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats

from utils.statistics import safe_statistical_test, handle_missing_values

def explore_basic_transformations(df, numeric_columns):
    """Explore basic data transformations"""
    st.subheader("Apply Basic Transformations")
    
    if not numeric_columns:
        st.warning("No numeric columns available for transformation.")
        return
        
    transform_col = st.selectbox("Select column to transform:", numeric_columns)
    transform_type = st.selectbox(
        "Select transformation:",
        ["None", "Log", "Square Root", "Square", "Cube", "Z-Score", "Min-Max Scaling"]
    )
    
    if transform_type != "None":
        # Check for missing values
        data_no_na, warning = handle_missing_values(df, transform_col, "transformation")
        if data_no_na is None:
            st.error(warning)
            return
        if warning:
            st.warning(warning)
            
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("#### Original Data")
            fig = px.histogram(
                df,
                x=transform_col,
                title=f"Original Distribution of {transform_col}"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.write("Original Statistics:")
            original_stats = df[transform_col].describe()
            st.dataframe(pd.DataFrame(original_stats).T)
        
        with col2:
            st.write("#### Transformed Data")
            
            try:
                # Apply transformation
                if transform_type == "Log":
                    # Handle negative or zero values
                    min_val = data_no_na.min()
                    offset = 0
                    if min_val <= 0:
                        offset = abs(min_val) + 1
                        st.info(f"Adding {offset} to all values before log transformation to handle non-positive values.")
                    
                    transformed = np.log(data_no_na + offset)
                    transform_name = f"Log({transform_col})"
                
                elif transform_type == "Square Root":
                    # Handle negative values
                    min_val = data_no_na.min()
                    offset = 0
                    if min_val < 0:
                        offset = abs(min_val) + 0.01
                        st.info(f"Adding {offset} to all values before square root transformation to handle negative values.")
                    
                    transformed = np.sqrt(data_no_na + offset)
                    transform_name = f"Sqrt({transform_col})"
                
                elif transform_type == "Square":
                    transformed = data_no_na ** 2
                    transform_name = f"{transform_col}²"
                
                elif transform_type == "Cube":
                    transformed = data_no_na ** 3
                    transform_name = f"{transform_col}³"
                
                elif transform_type == "Z-Score":
                    transformed = (data_no_na - data_no_na.mean()) / data_no_na.std()
                    transform_name = f"Z({transform_col})"
                
                elif transform_type == "Min-Max Scaling":
                    min_val = data_no_na.min()
                    max_val = data_no_na.max()
                    if max_val == min_val:
                        st.error("Cannot perform Min-Max Scaling: all values are identical.")
                        return
                    transformed = (data_no_na - min_val) / (max_val - min_val)
                    transform_name = f"MinMax({transform_col})"
                
                # Plot transformed data
                fig = px.histogram(
                    transformed,
                    title=f"Transformed Distribution: {transform_name}"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.write("Transformed Statistics:")
                transformed_stats = transformed.describe()
                st.dataframe(pd.DataFrame(transformed_stats).T)
                
                # Check normality of transformed data
                if st.checkbox("Test normality of transformed data"):
                    if len(transformed.dropna()) < 3:
                        st.warning("Not enough data points for normality test (need at least 3).")
                    else:
                        sample_size = min(5000, len(transformed.dropna()))
                        sample = transformed.dropna().sample(sample_size, random_state=42)
                        
                        shapiro_result, error = safe_statistical_test(
                            stats.shapiro, sample, test_name="Shapiro-Wilk test"
                        )
                        
                        if error:
                            st.warning(error)
                        else:
                            shapiro_stat, shapiro_p = shapiro_result
                            st.write(f"Shapiro-Wilk Test: statistic={shapiro_stat:.4f}, p-value={shapiro_p:.4f}")
                            
                            if shapiro_p < 0.05:
                                st.write("The transformed data is still not normally distributed (p < 0.05).")
                            else:
                                st.write("The transformed data appears to be normally distributed (p >= 0.05).")
                                
                # Compare with original data
                if st.checkbox("Compare with original data"):
                    import matplotlib.pyplot as plt
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                    
                    # Original data histogram
                    ax1.hist(data_no_na, bins=30, alpha=0.7)
                    ax1.set_title(f"Original: {transform_col}")
                    ax1.set_xlabel("Value")
                    ax1.set_ylabel("Frequency")
                    
                    # Transformed data histogram
                    ax2.hist(transformed, bins=30, alpha=0.7, color='orange')
                    ax2.set_title(f"Transformed: {transform_name}")
                    ax2.set_xlabel("Value")
                    ax2.set_ylabel("Frequency")
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)  # Close the figure to free memory
            
            except Exception as e:
                st.error(f"Error during transformation: {str(e)}")
