import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px
import plotly.figure_factory as ff
import gc
import time

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

# Helper functions
def create_distribution_plot(df, column, title, max_rows=10000):
    """Helper function to create distribution plots with sampling for large datasets"""
    if len(df) > max_rows:
        plot_df = df.sample(max_rows, random_state=42)
        st.info(f"Dataset is large. Visualizing a random sample of {max_rows:,} rows.")
    else:
        plot_df = df
    
    fig = px.histogram(plot_df, x=column, marginal="box", title=title)
    return fig

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
        st.error(f"Column '{column}' contains only missing values.")
        return None
    elif len(data_no_na) < len(df[column]):
        st.warning(f"Column '{column}' contains {len(df[column]) - len(data_no_na):,} missing values that were excluded from {operation_name}.")
    return data_no_na

def run_app():
    # Check if we should use sample data
    if st.session_state.use_sample_data and st.session_state.sample_data is not None:
        df = st.session_state.sample_data
        st.sidebar.success("Using sample data!")
        analyze_data(df)
    elif uploaded_file is not None:
        # Load data
        try:
            with st.spinner("Loading data..."):
                df = pd.read_csv(uploaded_file)
            st.sidebar.success("File successfully loaded!")
            analyze_data(df)
        except Exception as e:
            st.error(f"Error loading the data: {str(e)}")
    else:
        # Display welcome message and sample data option when no file is uploaded
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
                # Generate sample data
                np.random.seed(42)
                sample_size = 1000
                
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
                
                # Store in session state
                st.session_state.sample_data = sample_df
                st.session_state.use_sample_data = True
                
                # Display success message
                st.success("Sample data loaded successfully!")
                st.rerun()

def analyze_data(df):
    """Main function to analyze the loaded data"""
    # Display basic information
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

def analyze_numeric_column(df, selected_column, numeric_columns):
    """Analyze a numeric column"""
    # Check for missing values
    data_no_na = handle_missing_values(df, selected_column)
    if data_no_na is None:
        return
    
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
        fig = create_distribution_plot(df, selected_column, f"Distribution of {selected_column}")
        st.plotly_chart(fig, use_container_width=True)
    
    # Normality tests
    st.subheader("Normality Tests")
    col1, col2 = st.columns(2)
    
    with col1:
        # Shapiro-Wilk test (limited to 5000 samples)
        sample_size = min(5000, len(data_no_na))
        if sample_size < 3:
            st.warning("Not enough non-missing values for Shapiro-Wilk test (need at least 3).")
        else:
            sample = data_no_na.sample(sample_size, random_state=42) if len(data_no_na) > sample_size else data_no_na
            
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
        if len(data_no_na) >= 3:  # Need at least 3 points for Q-Q plot
            st.write("**Q-Q Plot**")
            try:
                fig, ax = plt.subplots(figsize=(8, 4))
                stats.probplot(data_no_na, plot=ax)
                st.pyplot(fig)
                plt.close(fig)  # Close the figure to free memory
            except Exception as e:
                st.warning(f"Could not generate Q-Q plot: {str(e)}")
        else:
            st.warning("Not enough non-missing values for Q-Q plot (need at least 3).")
    
    # Outlier detection
    st.subheader("Outlier Detection")
    
    q1 = data_no_na.quantile(0.25)
    q3 = data_no_na.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = df[(df[selected_column] < lower_bound) | (df[selected_column] > upper_bound)]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"IQR Method (1.5 × IQR)")
        st.write(f"Lower bound: {lower_bound:.4f}")
        st.write(f"Upper bound: {upper_bound:.4f}")
        st.write(f"Number of outliers: {len(outliers):,}")
        st.write(f"Percentage of outliers: {(len(outliers) / len(df) * 100):.2f}%")
    
    with col2:
        # Box plot for outliers
        fig = px.box(df, y=selected_column, title=f"Box Plot of {selected_column}")
        st.plotly_chart(fig, use_container_width=True)
    
    if len(outliers) > 0:
        with st.expander(f"View Outliers ({len(outliers):,} rows)"):
            if len(outliers) > 1000:
                st.warning(f"Showing only the first 1,000 of {len(outliers):,} outliers.")
                st.dataframe(outliers.head(1000))
            else:
                st.dataframe(outliers)
    
    # Correlation analysis
    if len(numeric_columns) > 1:
        st.subheader("Correlation Analysis")
        
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

def analyze_categorical_column(df, selected_column, numeric_columns, categorical_columns):
    """Analyze a categorical column"""
    # Check for missing values
    data_no_na = handle_missing_values(df, selected_column)
    if data_no_na is None:
        return
    
    st.subheader("Frequency Distribution")
    
    # Get value counts and handle potential memory issues for large datasets
    if len(data_no_na.unique()) > 1000:
        st.warning(f"Column has {len(data_no_na.unique()):,} unique values, which may be too many for a categorical variable.")
    
    value_counts = data_no_na.value_counts().reset_index()
    value_counts.columns = [selected_column, 'Count']
    value_counts['Percentage'] = (value_counts['Count'] / value_counts['Count'].sum() * 100).round(2)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if len(value_counts) > 50:
            st.warning(f"Showing only top 50 of {len(value_counts):,} categories.")
            st.dataframe(value_counts.head(50))
        else:
            st.dataframe(value_counts)
        st.write(f"Number of unique values: {data_no_na.nunique():,}")
    
    with col2:
        # Bar chart of frequency (limit to top 20 for readability)
        if len(value_counts) > 20:
            plot_data = value_counts.head(20)
            title = f"Top 20 Categories in {selected_column}"
        else:
            plot_data = value_counts
            title = f"Frequency Distribution of {selected_column}"
        
        fig = px.bar(
            plot_data,
            x=selected_column,
            y='Count',
            title=title
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Pie chart (only if 10 or fewer categories for readability)
    if data_no_na.nunique() <= 10:
        fig = px.pie(
            value_counts, 
            values='Count', 
            names=selected_column,
            title=f"Proportion of Categories in {selected_column}"
        )
        st.plotly_chart(fig, use_container_width=True)
    elif data_no_na.nunique() <= 20:
        with st.expander("Show Pie Chart (many categories)"):
            fig = px.pie(
                value_counts.head(20), 
                values='Count', 
                names=selected_column,
                title=f"Top 20 Categories in {selected_column}"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Relationship with numeric columns
    if numeric_columns:
        st.subheader(f"Relationship with Numeric Variables")
        
        numeric_col = st.selectbox(
            "Select a numeric column to analyze relationship:",
            numeric_columns
        )
        
        # Check if categorical column has too many unique values
        if data_no_na.nunique() > 20:
            st.warning(f"Column has {data_no_na.nunique():,} categories. Showing analysis for top 20 categories by frequency.")
            top_cats = data_no_na.value_counts().head(20).index.tolist()
            filtered_df = df[df[selected_column].isin(top_cats)]
        else:
            filtered_df = df
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Box plot
            fig = px.box(
                filtered_df,
                x=selected_column,
                y=numeric_col,
                title=f"{numeric_col} by {selected_column}"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Bar chart with mean
            try:
                agg_df = filtered_df.groupby(selected_column)[numeric_col].agg(['mean', 'median', 'count']).reset_index()
                agg_df = agg_df.sort_values('mean', ascending=False)
                
                fig = px.bar(
                    agg_df,
                    x=selected_column,
                    y='mean',
                    error_y=agg_df['median'],
                    title=f"Mean {numeric_col} by {selected_column}",
                    hover_data=['count']
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating aggregate plot: {str(e)}")
        
        # ANOVA test if there are enough categories
        if 2 <= data_no_na.nunique() <= 20:
            with st.expander("ANOVA Test", expanded=True):
                st.write("Testing if the mean of the numeric variable differs between categories")
                
                # Get groups, handling potential issues
                groups = []
                group_names = []
                
                for cat in data_no_na.unique():
                    group_data = df[df[selected_column] == cat][numeric_col].dropna()
                    if len(group_data) > 0:
                        groups.append(group_data)
                        group_names.append(cat)
                
                # Only perform ANOVA if we have valid groups
                if len(groups) < 2:
                    st.warning("Cannot perform ANOVA: need at least 2 groups with data.")
                else:
                    try:
                        f_stat, p_val = stats.f_oneway(*groups)
                        
                        st.write(f"F-statistic: {f_stat:.4f}")
                        st.write(f"p-value: {p_val:.4f}")
                        
                        if p_val < 0.05:
                            st.write("Conclusion: There is a **significant difference** between groups (p < 0.05)")
                        else:
                            st.write("Conclusion: There is no significant difference between groups (p >= 0.05)")
                            
                        # Show group statistics
                        group_stats = []
                        for i, group in enumerate(groups):
                            group_stats.append({
                                'Group': group_names[i],
                                'Count': len(group),
                                'Mean': group.mean(),
                                'Std Dev': group.std()
                            })
                        
                        st.write("#### Group Statistics")
                        group_stats_df = pd.DataFrame(group_stats)
                        st.dataframe(group_stats_df.style.format({
                            'Mean': '{:.4f}',
                            'Std Dev': '{:.4f}'
                        }))
                        
                    except Exception as e:
                        st.error(f"Error performing ANOVA test: {str(e)}")

def explore_custom_visualization(df, numeric_columns, categorical_columns):
    """Create custom visualizations"""
    st.subheader("Create Custom Visualization")
    
    viz_type = st.selectbox(
        "Select visualization type:",
        ["Scatter Plot", "Line Chart", "Bar Chart", "Histogram", "Box Plot", "Violin Plot", "Heatmap"]
    )
    
    try:
        if viz_type == "Scatter Plot":
            if len(numeric_columns) < 2:
                st.warning("Need at least 2 numeric columns for a scatter plot.")
                return
                
            x_col = st.selectbox("Select X-axis column:", numeric_columns)
            y_col = st.selectbox("Select Y-axis column:", 
                                [col for col in numeric_columns if col != x_col])
            color_col = st.selectbox("Color by (optional):", 
                                    ["None"] + categorical_columns)
            
            # Sample large datasets for better performance
            plot_df = df
            if len(df) > 5000:
                plot_df = df.sample(5000, random_state=42)
                st.info(f"Dataset is large. Visualizing a random sample of 5,000 rows.")
            
            fig = px.scatter(
                plot_df,
                x=x_col,
                y=y_col,
                color=None if color_col == "None" else color_col,
                title=f"{y_col} vs {x_col}",
                trendline="ols" if st.checkbox("Add trendline") else None
            )
            st.plotly_chart(fig, use_container_width=True)
            
            if st.checkbox("Show correlation coefficient"):
                corr = df[[x_col, y_col]].corr().iloc[0, 1]
                st.write(f"Pearson correlation coefficient: {corr:.4f}")
        
        elif viz_type == "Line Chart":
            x_col = st.selectbox("Select X-axis column:", df.columns)
            y_cols = st.multiselect("Select Y-axis column(s):", numeric_columns)
            
            if y_cols:
                # Handle datetime x-axis if applicable
                if pd.api.types.is_datetime64_any_dtype(df[x_col]):
                    plot_df = df.sort_values(x_col)
                else:
                    plot_df = df
                
                # Sample if dataset is very large
                if len(plot_df) > 10000:
                    st.warning(f"Dataset is large. Sampling to 10,000 points for better performance.")
                    plot_df = plot_df.sample(10000, random_state=42).sort_values(x_col)
                
                fig = px.line(
                    plot_df,
                    x=x_col,
                    y=y_cols,
                    title=f"Line Chart"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Bar Chart":
            x_col = st.selectbox("Select X-axis column:", df.columns)
            y_col = st.selectbox("Select Y-axis column:", numeric_columns)
            
            agg_func = st.selectbox(
                "Aggregation function:",
                ["Mean", "Sum", "Count", "Median", "Min", "Max"]
            )
            
            # Handle columns with too many unique values
            if df[x_col].nunique() > 50:
                st.warning(f"Column {x_col} has {df[x_col].nunique():,} unique values. Showing top 50 by {agg_func.lower()}.")
                
                # Group and aggregate
                agg_df = df.groupby(x_col)[y_col].agg(agg_func.lower()).reset_index()
                
                # Sort and limit to top 50
                if agg_func.lower() in ['mean', 'sum', 'median', 'max']:
                    agg_df = agg_df.sort_values(y_col, ascending=False).head(50)
                elif agg_func.lower() == 'min':
                    agg_df = agg_df.sort_values(y_col, ascending=True).head(50)
                else:  # count
                    agg_df = agg_df.sort_values(y_col, ascending=False).head(50)
            else:
                # Group and aggregate
                agg_df = df.groupby(x_col)[y_col].agg(agg_func.lower()).reset_index()
            
            fig = px.bar(
                agg_df,
                x=x_col,
                y=y_col,
                title=f"{agg_func} of {y_col} by {x_col}"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Histogram":
            hist_col = st.selectbox("Select column for histogram:", numeric_columns)
            bins = st.slider("Number of bins:", 5, 100, 20)
            
            fig = px.histogram(
                df,
                x=hist_col,
                nbins=bins,
                marginal="box",
                title=f"Histogram of {hist_col}"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Box Plot":
            y_col = st.selectbox("Select column for box plot:", numeric_columns)
            x_col = st.selectbox("Group by (optional):", ["None"] + categorical_columns)
            
            # Handle categorical columns with too many unique values
            if x_col != "None" and df[x_col].nunique() > 20:
                st.warning(f"Column {x_col} has {df[x_col].nunique():,} unique values. Showing top 20 categories by frequency.")
                top_cats = df[x_col].value_counts().head(20).index.tolist()
                plot_df = df[df[x_col].isin(top_cats)]
            else:
                plot_df = df
            
            fig = px.box(
                plot_df,
                y=y_col,
                x=None if x_col == "None" else x_col,
                title=f"Box Plot of {y_col}" + (f" by {x_col}" if x_col != "None" else "")
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Violin Plot":
            y_col = st.selectbox("Select column for violin plot:", numeric_columns)
            x_col = st.selectbox("Group by (optional):", ["None"] + categorical_columns)
            
            # Handle categorical columns with too many unique values
            if x_col != "None" and df[x_col].nunique() > 20:
                st.warning(f"Column {x_col} has {df[x_col].nunique():,} unique values. Showing top 20 categories by frequency.")
                top_cats = df[x_col].value_counts().head(20).index.tolist()
                plot_df = df[df[x_col].isin(top_cats)]
            else:
                plot_df = df
            
            fig = px.violin(
                plot_df,
                y=y_col,
                x=None if x_col == "None" else x_col,
                box=True,
                title=f"Violin Plot of {y_col}" + (f" by {x_col}" if x_col != "None" else "")
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Heatmap":
            if len(numeric_columns) < 2:
                st.warning("Need at least 2 numeric columns for a heatmap.")
                return
                
            corr_cols = st.multiselect(
                "Select columns for heatmap:",
                numeric_columns,
                default=numeric_columns[:min(5, len(numeric_columns))]
            )
            
            if len(corr_cols) > 1:
                corr_method = st.radio(
                    "Correlation method:",
                    ["Pearson", "Spearman", "Kendall"],
                    horizontal=True
                )
                
                try:
                    corr_df = df[corr_cols].corr(method=corr_method.lower())
                    
                    fig = px.imshow(
                        corr_df,
                        text_auto=True,
                        color_continuous_scale="RdBu_r",
                        title=f"{corr_method} Correlation Heatmap"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating heatmap: {str(e)}")
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")

def explore_correlation_matrix(df, numeric_columns):
    """Explore correlation matrix"""
    st.subheader("Correlation Matrix")
    
    if len(numeric_columns) < 2:
        st.warning("Need at least 2 numeric columns for correlation analysis.")
        return
        
    corr_method = st.radio(
        "Select correlation method:",
        ["Pearson", "Spearman", "Kendall"],
        horizontal=True,
        key="corr_method_tab2"
    )
    
    num_cols_to_show = st.slider(
        "Number of columns to include:",
        min_value=2,
        max_value=min(20, len(numeric_columns)),
        value=min(10, len(numeric_columns))
    )
    
    selected_num_cols = st.multiselect(
        "Select columns for correlation matrix:",
        numeric_columns,
        default=numeric_columns[:num_cols_to_show]
    )
    
    if len(selected_num_cols) > 1:
        try:
            with st.spinner("Calculating correlations..."):
                corr_df = df[selected_num_cols].corr(method=corr_method.lower())
                
                fig = px.imshow(
                    corr_df,
                    text_auto=True,
                    color_continuous_scale="RdBu_r",
                    title=f"{corr_method} Correlation Matrix"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show strongest correlations
                st.subheader("Strongest Correlations")
                
                # Create a DataFrame with all pairwise correlations
                corr_pairs = []
                for i in range(len(selected_num_cols)):
                    for j in range(i+1, len(selected_num_cols)):
                        col1 = selected_num_cols[i]
                        col2 = selected_num_cols[j]
                        corr_value = corr_df.loc[col1, col2]
                        corr_pairs.append({
                            'Variable 1': col1,
                            'Variable 2': col2,
                            'Correlation': corr_value,
                            'Abs Correlation': abs(corr_value)
                        })
                
                if corr_pairs:
                    corr_pairs_df = pd.DataFrame(corr_pairs)
                    top_corr = corr_pairs_df.sort_values('Abs Correlation', ascending=False).head(10)
                    
                    st.dataframe(
                        top_corr[['Variable 1', 'Variable 2', 'Correlation']]
                        .style.background_gradient(cmap='RdBu_r', subset=['Correlation'])
                        .format({'Correlation': '{:.4f}'})
                    )
                    
                    # Visualize top correlation as scatter plot
                    if len(corr_pairs) > 0 and st.checkbox("Visualize top correlation"):
                        top_pair = top_corr.iloc[0]
                        var1 = top_pair['Variable 1']
                        var2 = top_pair['Variable 2']
                        corr_val = top_pair['Correlation']
                        
                        # Sample for large datasets
                        plot_df = df
                        if len(df) > 5000:
                            plot_df = df.sample(5000, random_state=42)
                            st.info("Showing a sample of 5,000 points for better performance.")
                        
                        fig = px.scatter(
                            plot_df,
                            x=var1,
                            y=var2,
                            trendline="ols",
                            title=f"Strongest Correlation: {var1} vs {var2} (r = {corr_val:.4f})"
                        )
                        st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error in correlation analysis: {str(e)}")
    else:
        st.info("Please select at least 2 columns for correlation analysis.")

def explore_statistical_tests(df, numeric_columns, categorical_columns):
    """Run statistical tests"""
    st.subheader("Statistical Tests")
    
    test_type = st.selectbox(
        "Select test type:",
        ["T-Test (One Sample)", "T-Test (Two Samples)", 
         "Chi-Square Test", "ANOVA", "Normality Tests"]
    )
    
    try:
        if test_type == "T-Test (One Sample)":
            if not numeric_columns:
                st.warning("No numeric columns available for t-test.")
                return
                
            col = st.selectbox("Select column:", numeric_columns)
            mu = st.number_input("Hypothesized mean (μ₀):", value=0.0)
            
            sample = df[col].dropna()
            if len(sample) < 2:
                st.error(f"Not enough non-missing values in {col} for t-test.")
                return
                
            t_stat, p_val = stats.ttest_1samp(sample, mu)
            
            st.write("### One-Sample T-Test Results")
            st.write(f"Null hypothesis (H₀): μ = {mu}")
            st.write(f"Alternative hypothesis (H₁): μ ≠ {mu}")
            st.write(f"Sample mean: {sample.mean():.4f}")
            st.write(f"Sample size: {len(sample):,}")
            st.write(f"T-statistic: {t_stat:.4f}")
            st.write(f"P-value: {p_val:.4f}")
            
            if p_val < 0.05:
                st.write("Conclusion: **Reject** the null hypothesis (p < 0.05)")
            else:
                st.write("Conclusion: **Fail to reject** the null hypothesis (p >= 0.05)")
            
            # Visualize the sample distribution
            fig = px.histogram(
                sample, 
                title=f"Distribution of {col}",
                marginal="box"
            )
            # Add a vertical line at the hypothesized mean
            fig.add_vline(x=mu, line_dash="dash", line_color="red", 
                         annotation_text=f"μ₀ = {mu}", annotation_position="top right")
            st.plotly_chart(fig, use_container_width=True)
        
        elif test_type == "T-Test (Two Samples)":
            if not numeric_columns:
                st.warning("No numeric columns available for t-test.")
                return
            if not categorical_columns:
                st.warning("No categorical columns available for grouping.")
                return
                
            col = st.selectbox("Select numeric column:", numeric_columns)
            group_col = st.selectbox("Select grouping column:", categorical_columns)
            
            # Get unique values in the grouping column
            unique_groups = df[group_col].dropna().unique()
            
            if len(unique_groups) < 2:
                st.error(f"Need at least 2 groups in {group_col} for two-sample t-test.")
                return
                
            group1 = st.selectbox("Select first group:", unique_groups)
            remaining_groups = [g for g in unique_groups if g != group1]
            group2 = st.selectbox("Select second group:", remaining_groups)
            
            sample1 = df[df[group_col] == group1][col].dropna()
            sample2 = df[df[group_col] == group2][col].dropna()
            
            if len(sample1) < 2 or len(sample2) < 2:
                st.error(f"Not enough non-missing values in one or both groups for t-test.")
                return
                
            equal_var = st.checkbox("Assume equal variances", value=True)
            
            t_stat, p_val = stats.ttest_ind(sample1, sample2, equal_var=equal_var)
            
            st.write("### Two-Sample T-Test Results")
            st.write(f"Null hypothesis (H₀): μ₁ = μ₂")
            st.write(f"Alternative hypothesis (H₁): μ₁ ≠ μ₂")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Group 1 ({group1})**")
                st.write(f"Mean: {sample1.mean():.4f}")
                st.write(f"Size: {len(sample1):,}")
                st.write(f"Std Dev: {sample1.std():.4f}")
            
            with col2:
                st.write(f"**Group 2 ({group2})**")
                st.write(f"Mean: {sample2.mean():.4f}")
                st.write(f"Size: {len(sample2):,}")
                st.write(f"Std Dev: {sample2.std():.4f}")
            
            st.write(f"T-statistic: {t_stat:.4f}")
            st.write(f"P-value: {p_val:.4f}")
            
            if p_val < 0.05:
                st.write("Conclusion: **Reject** the null hypothesis (p < 0.05)")
                st.write("There is a significant difference between at least two group means.")
            else:
                st.write("Conclusion: **Fail to reject** the null hypothesis (p >= 0.05)")
                st.write("There is no significant difference between group means.")
            
            # Visualize the groups
            fig = px.box(
                df,
                x=cat_col,
                y=num_col,
                title=f"Distribution of {num_col} by {cat_col} groups"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Post-hoc test (Tukey's HSD) if ANOVA is significant
            if p_val < 0.05 and len(groups) > 2:
                st.write("#### Post-hoc Analysis (Tukey's HSD)")
                st.write("Since ANOVA is significant, performing pairwise comparisons:")
                
                try:
                    # Create a DataFrame in the format needed for Tukey's test
                    tukey_data = []
                    for i, group in enumerate(groups):
                        for value in group:
                            tukey_data.append({
                                'group': group_names[i],
                                'value': value
                            })
                    
                    tukey_df = pd.DataFrame(tukey_data)
                    
                    # Perform Tukey's test
                    from statsmodels.stats.multicomp import pairwise_tukeyhsd
                    tukey_result = pairwise_tukeyhsd(
                        tukey_df['value'],
                        tukey_df['group'],
                        alpha=0.05
                    )
                    
                    # Display results
                    tukey_summary = pd.DataFrame(
                        data=tukey_result._results_table.data[1:],
                        columns=tukey_result._results_table.data[0]
                    )
                    
                    st.dataframe(tukey_summary)
                    
                    st.write("Pairs with p-adj < 0.05 have significantly different means.")
                except Exception as e:
                    st.error(f"Error in post-hoc analysis: {str(e)}")
        
        elif test_type == "Normality Tests":
            if not numeric_columns:
                st.warning("No numeric columns available for normality tests.")
                return
                
            col = st.selectbox("Select column to test for normality:", numeric_columns)
            
            sample = df[col].dropna()
            if len(sample) < 3:
                st.error(f"Not enough non-missing values in {col} for normality tests (need at least 3).")
                return
                
            st.write("### Normality Test Results")
            
            # Shapiro-Wilk test (limited to 5000 samples)
            if len(sample) > 5000:
                st.write("Note: Shapiro-Wilk test is limited to 5000 samples. Using a random subset.")
                sample_shapiro = sample.sample(5000, random_state=42)
            else:
                sample_shapiro = sample
            
            shapiro_result, shapiro_error = safe_statistical_test(
                stats.shapiro, sample_shapiro, test_name="Shapiro-Wilk test"
            )
            
            # D'Agostino's K^2 test
            k2_result, k2_error = safe_statistical_test(
                stats.normaltest, sample, test_name="D'Agostino's K^2 test"
            )
            
            # Kolmogorov-Smirnov test
            ks_result, ks_error = safe_statistical_test(
                lambda x: stats.kstest(
                    (x - x.mean()) / x.std() if x.std() > 0 else x - x.mean(),
                    'norm'
                ),
                sample,
                test_name="Kolmogorov-Smirnov test"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("#### Test Statistics")
                test_results = []
                
                if not shapiro_error:
                    shapiro_stat, shapiro_p = shapiro_result
                    test_results.append({
                        'Test': 'Shapiro-Wilk',
                        'Statistic': shapiro_stat,
                        'p-value': shapiro_p,
                        'Normal?': 'No' if shapiro_p < 0.05 else 'Yes'
                    })
                else:
                    st.warning(shapiro_error)
                
                if not k2_error:
                    k2_stat, k2_p = k2_result
                    test_results.append({
                        'Test': "D'Agostino's K²",
                        'Statistic': k2_stat,
                        'p-value': k2_p,
                        'Normal?': 'No' if k2_p < 0.05 else 'Yes'
                    })
                else:
                    st.warning(k2_error)
                
                if not ks_error:
                    ks_stat, ks_p = ks_result
                    test_results.append({
                        'Test': 'Kolmogorov-Smirnov',
                        'Statistic': ks_stat,
                        'p-value': ks_p,
                        'Normal?': 'No' if ks_p < 0.05 else 'Yes'
                    })
                else:
                    st.warning(ks_error)
                
                if test_results:
                    test_results_df = pd.DataFrame(test_results)
                    st.dataframe(test_results_df.style.format({
                        'Statistic': '{:.4f}',
                        'p-value': '{:.4f}'
                    }))
                    
                    st.write("Note: p < 0.05 suggests the data is not normally distributed.")
                else:
                    st.error("All normality tests failed. Please check your data.")
            
            with col2:
                # Q-Q Plot
                try:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    stats.probplot(sample, plot=ax)
                    ax.set_title(f"Q-Q Plot for {col}")
                    st.pyplot(fig)
                    plt.close(fig)  # Close the figure to free memory
                except Exception as e:
                    st.warning(f"Could not generate Q-Q plot: {str(e)}")
            
            # Histogram with normal curve
            try:
                fig = plt.figure(figsize=(10, 6))
                sns.histplot(sample, kde=True, stat="density")
                
                # Add a normal curve
                xmin, xmax = plt.xlim()
                x = np.linspace(xmin, xmax, 100)
                p = stats.norm.pdf(x, sample.mean(), sample.std())
                plt.plot(x, p, 'k', linewidth=2)
                
                plt.title(f"Histogram with Normal Curve for {col}")
                st.pyplot(fig)
                plt.close(fig)  # Close the figure to free memory
            except Exception as e:
                st.warning(f"Could not generate histogram: {str(e)}")
            
            # Skewness and Kurtosis
            skewness = stats.skew(sample)
            kurtosis = stats.kurtosis(sample)
            
            st.write("#### Additional Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Skewness", f"{skewness:.4f}")
                if abs(skewness) < 0.5:
                    st.write("Approximately symmetric")
                elif abs(skewness) < 1:
                    st.write("Moderately skewed")
                else:
                    st.write("Highly skewed")
            
            with col2:
                st.metric("Kurtosis", f"{kurtosis:.4f}")
                if kurtosis < -0.5:
                    st.write("Platykurtic (flatter)")
                elif kurtosis > 0.5:
                    st.write("Leptokurtic (more peaked)")
                else:
                    st.write("Mesokurtic (normal-like)")
            
            with col3:
                st.metric("Sample Size", f"{len(sample):,}")
    
    except Exception as e:
        st.error(f"Error in statistical test: {str(e)}")

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
        data_no_na = handle_missing_values(df, transform_col, "transformation")
        if data_no_na is None:
            return
            
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
                
                elif advanced_transform == "Yeo-Johnson Transformation":
                    pt = preprocessing.PowerTransformer(method='yeo-johnson')
                    transformed_data = pt.fit_transform(data).flatten()
                    st.info(f"Optimal λ (lambda) value: {pt.lambdas_[0]:.4f}")
                
                elif advanced_transform == "Quantile Transformation":
                    output_dist = st.radio("Output distribution:", ["normal", "uniform"], horizontal=True)
                    qt = preprocessing.QuantileTransformer(output_distribution=output_dist, random_state=42)
                    transformed_data = qt.fit_transform(data).flatten()
                
                elif advanced_transform == "Power Transformation":
                    power = st.slider("Power value:", -3.0, 3.0, 1.0, 0.1)
                    if power == 0:
                        transformed_data = np.log(data.flatten())
                        st.info("Power = 0 corresponds to log transformation")
                    else:
                        transformed_data = np.power(data.flatten(), power)
                
                elif advanced_transform == "Robust Scaling":
                    rs = preprocessing.RobustScaler()
                    transformed_data = rs.fit_transform(data).flatten()
                    st.info("Scaled using median and interquartile range (robust to outliers)")
                
                # Plot transformed data
                fig = px.histogram(
                    transformed_data,
                    title=f"{advanced_transform} of {transform_col}"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Check normality of transformed data
                if len(transformed_data) > 3:  # Minimum sample size for Shapiro-Wilk test
                    sample_size = min(5000, len(transformed_data))
                    sample = np.random.choice(transformed_data, size=sample_size, replace=False)
                    
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
            
            except Exception as e:
                st.error(f"Error during transformation: {str(e)}")
                st.error(f"Details: {type(e).__name__}")
                

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
        data_no_na = handle_missing_values(df, transform_col, "transformation")
        if data_no_na is None:
            return
            
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("#### Original Data")
            fig = px.histogram(
                df,
                x=transform_col,
                title=f"Original Distribution of {transform_col}"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("#### Transformed Data")
            
            # Get data and handle NaNs
            data = data_no_na.values.reshape(-1, 1)
            
            try:
                from scipy import stats
                from sklearn import preprocessing
                
                if advanced_transform == "Box-Cox Transformation":
                    # Box-Cox requires positive data
                    min_val = data.min()
                    if min_val <= 0:
                        offset = abs(min_val) + 1
                        st.info(f"Adding {offset} to all values before Box-Cox transformation to handle non-positive values.")
                        data = data + offset
                    
                    transformed_data, lambda_value = stats.boxcox(data.flatten())
                    st.info(f"Optimal λ (lambda) value: {lambda_value:.4f}")
                
                elif advanced_transform == "Yeo-Johnson Transformation":
                    pt = preprocessing.PowerTransformer(method='yeo-johnson')
                    transformed_data = pt.fit_transform(data).flatten()
                    st.info(f"Optimal λ (lambda) value: {pt.lambdas_[0]:.4f}")
                
                elif advanced_transform == "Quantile Transformation":
                    output_dist = st.radio("Output distribution:", ["normal", "uniform"], horizontal=True)
                    qt = preprocessing.QuantileTransformer(output_distribution=output_dist, random_state=42)
                    transformed_data = qt.fit_transform(data).flatten()
                
                elif advanced_transform == "Power Transformation":
                    power = st.slider("Power value:", -3.0, 3.0, 1.0, 0.1)
                    if power == 0:
                        transformed_data = np.log(data.flatten())
                        st.info("Power = 0 corresponds to log transformation")
                    else:
                        transformed_data = np.power(data.flatten(), power)
                
                elif advanced_transform == "Robust Scaling":
                    rs = preprocessing.RobustScaler()
                    transformed_data = rs.fit_transform(data).flatten()
                    st.info("Scaled using median and interquartile range (robust to outliers)")
                
                # Plot transformed data
                fig = px.histogram(
                    transformed_data,
                    title=f"{advanced_transform} of {transform_col}"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Check normality of transformed data
                if len(transformed_data) > 3:  # Minimum sample size for Shapiro-Wilk test
                    sample_size = min(5000, len(transformed_data))
                    sample = np.random.choice(transformed_data, size=sample_size, replace=False)
                    
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
                
                # Add Q-Q plot for transformed data
                if st.checkbox("Show Q-Q plot of transformed data"):
                    try:
                        fig, ax = plt.subplots(figsize=(8, 4))
                        stats.probplot(transformed_data, plot=ax)
                        ax.set_title(f"Q-Q Plot for Transformed {transform_col}")
                        st.pyplot(fig)
                        plt.close(fig)  # Close the figure to free memory
                    except Exception as e:
                        st.warning(f"Could not generate Q-Q plot: {str(e)}")
                
                # Compare original and transformed data
                if st.checkbox("Compare original and transformed distributions"):
                    compare_df = pd.DataFrame({
                        'Original': data_no_na,
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
                            data_no_na.mean(),
                            np.median(data_no_na),
                            data_no_na.std(),
                            stats.skew(data_no_na),
                            stats.kurtosis(data_no_na)
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
            
            except Exception as e:
                st.error(f"Error during transformation: {str(e)}")
                st.error(f"Details: {type(e).__name__}")
                
                # More detailed error information for debugging
                import traceback
                st.error(f"Traceback: {traceback.format_exc()}")


def generate_transformed_dataset(df, numeric_columns):
    """Generate and download transformed dataset"""
    st.subheader("Download Transformed Data")
    
    if not numeric_columns:
        st.warning("No numeric columns available for transformation.")
        return
        
    cols_to_transform = st.multiselect(
        "Select columns to transform:",
        numeric_columns
    )
    
    if not cols_to_transform:
        st.info("Please select at least one column to transform.")
        return
        
    transform_options = {
        "None": lambda x: x,
        "Log": lambda x: np.log(x - x.min() + 1 if x.min() <= 0 else x),
        "Square Root": lambda x: np.sqrt(x - x.min() + 0.01 if x.min() < 0 else x),
        "Square": lambda x: x ** 2,
        "Cube": lambda x: x ** 3,
        "Z-Score": lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x - x.mean(),
        "Min-Max Scaling": lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else x
    }
    
    transformed_df = df.copy()
    transformation_log = []
    
    for col in cols_to_transform:
        transform_type = st.selectbox(
            f"Transformation for {col}:",
            list(transform_options.keys()),
            key=f"transform_{col}"
        )
        
        if transform_type != "None":
            try:
                # Check for missing values and inform user
                missing_count = df[col].isnull().sum()
                if missing_count > 0:
                    st.warning(f"Column {col} has {missing_count:,} missing values that will remain missing after transformation.")
                
                # Apply transformation
                new_col_name = f"{col}_{transform_type}"
                transformed_df[new_col_name] = transform_options[transform_type](df[col])
                
                # Log the transformation
                transformation_log.append(f"- Added column '{new_col_name}' using {transform_type} transformation")
                
                # Check for issues in transformed data
                if transformed_df[new_col_name].isnull().sum() > missing_count:
                    st.warning(f"Transformation created additional missing values in {new_col_name}. Check for invalid inputs like negative values for log transformation.")
                
                if np.isinf(transformed_df[new_col_name]).any():
                    st.warning(f"Transformation created infinite values in {new_col_name}. These will be replaced with NaN.")
                    transformed_df[new_col_name] = transformed_df[new_col_name].replace([np.inf, -np.inf], np.nan)
                
            except Exception as e:
                st.error(f"Error transforming {col}: {str(e)}")
    
    if transformation_log:
        st.write("### Transformation Summary")
        for log in transformation_log:
            st.write(log)
        
        # Show preview of transformed data
        st.write("### Preview of Transformed Data")
        st.dataframe(transformed_df.head())
        
        # Convert to CSV
        try:
            csv = transformed_df.to_csv(index=False)
            
            st.download_button(
                label="Download Transformed Data as CSV",
                data=csv,
                file_name="transformed_data.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Error creating CSV file: {str(e)}")
            
        # Clean up to free memory
        del transformed_df
        gc.collect()
    else:
        st.info("No transformations were applied. Select columns and transformations to continue.")

# Run the app
if __name__ == "__main__":
    run_app()
