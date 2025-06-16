import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from scipy import stats

from utils.statistics import safe_statistical_test, handle_missing_values

def analyze_numeric_column(df, selected_column, numeric_columns):
    """Analyze a numeric column"""
    # Check for missing values
    data_no_na, warning = handle_missing_values(df, selected_column)
    if data_no_na is None:
        st.error(warning)
        return
    if warning:
        st.warning(warning)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Summary Statistics")
        stats_df = pd.DataFrame({
            'Statistic': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Q1 (25%)', 'Q3 (75%)', 'IQR', 'Skewness', 'Kurtosis'],
            'Value': [
                data_no_na.mean(),
                data_no_na.median(),
                data_no_na.std(),
                data_no_na.min(),
                data_no_na.max(),
                data_no_na.quantile(0.25),
                data_no_na.quantile(0.75),
                data_no_na.quantile(0.75) - data_no_na.quantile(0.25),
                data_no_na.skew(),
                data_no_na.kurt()
            ]
        })
        st.dataframe(stats_df)
    
    with col2:
        st.subheader("Distribution Plot")
        show_distribution_plot(df, selected_column)
    
    # Normality tests
    st.subheader("Normality Tests")
    run_normality_tests(data_no_na, selected_column)
    
    # Outlier detection
    st.subheader("Outlier Detection")
    detect_outliers(df, selected_column)
    
    # Correlation analysis
    if len(numeric_columns) > 1:
        st.subheader("Correlation Analysis")
        analyze_correlations_with_column(df, selected_column, numeric_columns)

def show_distribution_plot(df, column, max_points=10000):
    """Show distribution plot for a numeric column"""
    # Sample for large datasets
    if len(df) > max_points:
        plot_df = df.sample(max_points, random_state=42)
        st.info(f"Dataset is large. Visualizing a random sample of {max_points:,} rows.")
    else:
        plot_df = df
    
    fig = px.histogram(
        plot_df, 
        x=column, 
        marginal="box", 
        title=f"Distribution of {column}"
    )
    st.plotly_chart(fig, use_container_width=True)

def run_normality_tests(data, column_name):
    """Run normality tests on a data series"""
    col1, col2 = st.columns(2)
    
    with col1:
        # Shapiro-Wilk test (limited to 5000 samples)
        sample_size = min(5000, len(data))
        if sample_size < 3:
            st.warning("Not enough non-missing values for Shapiro-Wilk test (need at least 3).")
        else:
            sample = data.sample(sample_size, random_state=42) if len(data) > sample_size else data
            
            shapiro_result, error = safe_statistical_test(
                stats.shapiro, sample, test_name="Shapiro-Wilk test"
            )
            
            if error:
                st.warning(error)
            else:
                shapiro_stat, shapiro_p = shapiro_result
                st.write("**Shapiro-Wilk Test**")
                st.write(f"Statistic: {shapiro_stat:.4f}")
                st.write(f"p-value: {shapiro_p:.4f}")
                if shapiro_p < 0.05:
                    st.write("Conclusion: Data is **not normally distributed** (p < 0.05)")
                else:
                    st.write("Conclusion: Data appears to be normally distributed (p >= 0.05)")
    
    with col2:
        # Q-Q Plot
        if len(data) >= 3:  # Need at least 3 points for Q-Q plot
            st.write("**Q-Q Plot**")
            try:
                fig, ax = plt.subplots(figsize=(8, 4))
                stats.probplot(data, plot=ax)
                st.pyplot(fig)
                plt.close(fig)  # Close the figure to free memory
            except Exception as e:
                st.warning(f"Could not generate Q-Q plot: {str(e)}")
        else:
            st.warning("Not enough non-missing values for Q-Q plot (need at least 3).")

def detect_outliers(df, column):
    """Detect outliers in a numeric column"""
    data = df[column].dropna()
    
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"IQR Method (1.5 × IQR)")
        st.write(f"Lower bound: {lower_bound:.4f}")
        st.write(f"Upper bound: {upper_bound:.4f}")
        st.write(f"Number of outliers: {len(outliers):,}")
        st.write(f"Percentage of outliers: {(len(outliers) / len(df) * 100):.2f}%")
    
    with col2:
        # Box plot for outliers
        fig = px.box(df, y=column, title=f"Box Plot of {column}")
        st.plotly_chart(fig, use_container_width=True)
    
    if len(outliers) > 0:
        with st.expander(f"View Outliers ({len(outliers):,} rows)"):
            if len(outliers) > 1000:
                st.warning(f"Showing only the first 1,000 of {len(outliers):,} outliers.")
                st.dataframe(outliers.head(1000))
            else:
                st.dataframe(outliers)

def analyze_correlations_with_column(df, selected_column, numeric_columns):
    """Analyze correlations between selected column and other numeric columns"""
    with st.expander("Select columns for correlation analysis", expanded=True):
        corr_cols = st.multiselect(
            "Select columns:",
            numeric_columns,
            default=[selected_column] + [col for col in numeric_columns if col != selected_column][:min(4, len(numeric_columns)-1)]
        )
        
        if len(corr_cols) > 1:
            corr_method = st.radio(
                "Correlation method:",
                ["Pearson", "Spearman", "Kendall"],
                horizontal=True
            )
            
            # Check for missing values in correlation columns
            corr_df = df[corr_cols].copy()
            missing_in_corr = corr_df.isnull().sum().sum()
            if missing_in_corr > 0:
                st.warning(f"Found {missing_in_corr:,} missing values across selected columns. Using pairwise deletion for correlation.")
            
            try:
                corr_matrix = corr_df.corr(method=corr_method.lower())
                
                fig = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    color_continuous_scale="RdBu_r",
                    title=f"{corr_method} Correlation Matrix"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show correlation with selected column
                if len(corr_cols) > 2 and selected_column in corr_cols:
                    st.write(f"### Correlation with {selected_column}")
                    corr_with_selected = corr_matrix[selected_column].drop(selected_column).sort_values(ascending=False)
                    
                    fig = px.bar(
                        x=corr_with_selected.index,
                        y=corr_with_selected.values,
                        title=f"Correlation with {selected_column}",
                        labels={"x": "Column", "y": f"{corr_method} Correlation"}
                    )
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error calculating correlation: {str(e)}")
