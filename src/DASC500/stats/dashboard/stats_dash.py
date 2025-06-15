import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px
import plotly.figure_factory as ff

# Set page configuration
st.set_page_config(page_title="Statistical Analysis Dashboard", layout="wide")

# Add a title and description
st.title("📊 Statistical Analysis Dashboard")
st.markdown("""
This dashboard allows you to upload a CSV file and perform various statistical analyses on your data.
Select columns to analyze, visualize distributions, and run statistical tests.
""")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=['csv'])

# Main function to run the app
def run_app():
    if uploaded_file is not None:
        # Load data
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success("File successfully loaded!")
            
            # Display basic information
            with st.expander("Dataset Overview"):
                st.write("### Data Preview")
                st.dataframe(df.head())
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("### Dataset Shape")
                    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
                
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
            
            st.sidebar.header("Column Selection")
            selected_column = st.sidebar.selectbox("Select a column for detailed analysis:", df.columns)
            
            # Determine column type
            if selected_column in numeric_columns:
                column_type = "numeric"
            else:
                column_type = "categorical"
            
            # Display summary statistics
            st.header(f"Analysis of: {selected_column}")
            
            if column_type == "numeric":
                # Numeric column analysis
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Summary Statistics")
                    stats_df = pd.DataFrame({
                        'Statistic': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Q1 (25%)', 'Q3 (75%)', 'IQR', 'Skewness', 'Kurtosis'],
                        'Value': [
                            df[selected_column].mean(),
                            df[selected_column].median(),
                            df[selected_column].std(),
                            df[selected_column].min(),
                            df[selected_column].max(),
                            df[selected_column].quantile(0.25),
                            df[selected_column].quantile(0.75),
                            df[selected_column].quantile(0.75) - df[selected_column].quantile(0.25),
                            df[selected_column].skew(),
                            df[selected_column].kurt()
                        ]
                    })
                    st.dataframe(stats_df)
                
                with col2:
                    st.subheader("Distribution Plot")
                    fig = px.histogram(df, x=selected_column, marginal="box", 
                                       title=f"Distribution of {selected_column}")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Normality tests
                st.subheader("Normality Tests")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Shapiro-Wilk test
                    sample = df[selected_column].dropna()
                    # Limit sample size for Shapiro-Wilk test (max 5000)
                    if len(sample) > 5000:
                        sample = sample.sample(5000, random_state=42)
                    
                    shapiro_test = stats.shapiro(sample)
                    st.write("**Shapiro-Wilk Test**")
                    st.write(f"Statistic: {shapiro_test[0]:.4f}")
                    st.write(f"p-value: {shapiro_test[1]:.4f}")
                    if shapiro_test[1] < 0.05:
                        st.write("Conclusion: Data is **not normally distributed** (p < 0.05)")
                    else:
                        st.write("Conclusion: Data appears to be normally distributed (p >= 0.05)")
                
                with col2:
                    # Q-Q Plot
                    st.write("**Q-Q Plot**")
                    fig, ax = plt.subplots(figsize=(8, 4))
                    stats.probplot(df[selected_column].dropna(), plot=ax)
                    st.pyplot(fig)
                
                # Outlier detection
                st.subheader("Outlier Detection")
                
                q1 = df[selected_column].quantile(0.25)
                q3 = df[selected_column].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers = df[(df[selected_column] < lower_bound) | (df[selected_column] > upper_bound)]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"IQR Method (1.5 × IQR)")
                    st.write(f"Lower bound: {lower_bound:.4f}")
                    st.write(f"Upper bound: {upper_bound:.4f}")
                    st.write(f"Number of outliers: {len(outliers)}")
                    st.write(f"Percentage of outliers: {(len(outliers) / len(df) * 100):.2f}%")
                
                with col2:
                    # Box plot for outliers
                    fig = px.box(df, y=selected_column, title=f"Box Plot of {selected_column}")
                    st.plotly_chart(fig, use_container_width=True)
                
                if len(outliers) > 0 and len(outliers) < 100:
                    with st.expander("View Outliers"):
                        st.dataframe(outliers)
                
                # Correlation analysis
                if len(numeric_columns) > 1:
                    st.subheader("Correlation Analysis")
                    
                    corr_cols = st.multiselect(
                        "Select columns for correlation analysis:",
                        numeric_columns,
                        default=[selected_column] + [col for col in numeric_columns if col != selected_column][:4]
                    )
                    
                    if len(corr_cols) > 1:
                        corr_method = st.radio(
                            "Correlation method:",
                            ["Pearson", "Spearman", "Kendall"],
                            horizontal=True
                        )
                        
                        corr_df = df[corr_cols].corr(method=corr_method.lower())
                        
                        fig = px.imshow(
                            corr_df,
                            text_auto=True,
                            color_continuous_scale="RdBu_r",
                            title=f"{corr_method} Correlation Matrix"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show correlation with selected column
                        if len(corr_cols) > 2:
                            st.write(f"### Correlation with {selected_column}")
                            corr_with_selected = corr_df[selected_column].drop(selected_column).sort_values(ascending=False)
                            
                            fig = px.bar(
                                x=corr_with_selected.index,
                                y=corr_with_selected.values,
                                title=f"Correlation with {selected_column}",
                                labels={"x": "Column", "y": f"{corr_method} Correlation"}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                
            else:
                # Categorical column analysis
                st.subheader("Frequency Distribution")
                
                value_counts = df[selected_column].value_counts().reset_index()
                value_counts.columns = [selected_column, 'Count']
                value_counts['Percentage'] = (value_counts['Count'] / value_counts['Count'].sum() * 100).round(2)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.dataframe(value_counts)
                    st.write(f"Number of unique values: {df[selected_column].nunique()}")
                
                with col2:
                    # Bar chart of frequency
                    fig = px.bar(
                        value_counts,
                        x=selected_column,
                        y='Count',
                        title=f"Frequency Distribution of {selected_column}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Pie chart
                if df[selected_column].nunique() <= 10:
                    fig = px.pie(
                        value_counts, 
                        values='Count', 
                        names=selected_column,
                        title=f"Proportion of Categories in {selected_column}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Relationship with numeric columns
                if numeric_columns:
                    st.subheader(f"Relationship with Numeric Variables")
                    
                    numeric_col = st.selectbox(
                        "Select a numeric column to analyze relationship:",
                        numeric_columns
                    )
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Box plot
                        fig = px.box(
                            df,
                            x=selected_column,
                            y=numeric_col,
                            title=f"{numeric_col} by {selected_column}"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Bar chart with mean
                        agg_df = df.groupby(selected_column)[numeric_col].agg(['mean', 'median']).reset_index()
                        
                        fig = px.bar(
                            agg_df,
                            x=selected_column,
                            y='mean',
                            error_y=agg_df['median'],
                            title=f"Mean {numeric_col} by {selected_column}"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # ANOVA test if there are enough categories
                    if 2 <= df[selected_column].nunique() <= 10:
                        st.subheader("ANOVA Test")
                        st.write("Testing if the mean of the numeric variable differs between categories")
                        
                        groups = [df[df[selected_column] == cat][numeric_col].dropna() 
                                 for cat in df[selected_column].unique()]
                        
                        # Only perform ANOVA if we have valid groups
                        valid_groups = [g for g in groups if len(g) > 0]
                        
                        if len(valid_groups) >= 2:
                            f_stat, p_val = stats.f_oneway(*valid_groups)
                            
                            st.write(f"F-statistic: {f_stat:.4f}")
                            st.write(f"p-value: {p_val:.4f}")
                            
                            if p_val < 0.05:
                                st.write("Conclusion: There is a **significant difference** between groups (p < 0.05)")
                            else:
                                st.write("Conclusion: There is no significant difference between groups (p >= 0.05)")
                        else:
                            st.write("Cannot perform ANOVA: not enough valid groups")
            
            # Data exploration section
            st.header("Advanced Data Exploration")
            
            exploration_tab1, exploration_tab2, exploration_tab3 = st.tabs([
                "Custom Visualization", "Correlation Matrix", "Statistical Tests"
            ])
            
            with exploration_tab1:
                st.subheader("Create Custom Visualization")
                
                viz_type = st.selectbox(
                    "Select visualization type:",
                    ["Scatter Plot", "Line Chart", "Bar Chart", "Histogram", "Box Plot", "Violin Plot", "Heatmap"]
                )
                
                if viz_type == "Scatter Plot":
                    x_col = st.selectbox("Select X-axis column:", numeric_columns)
                    y_col = st.selectbox("Select Y-axis column:", 
                                         [col for col in numeric_columns if col != x_col])
                    color_col = st.selectbox("Color by (optional):", 
                                             ["None"] + categorical_columns)
                    
                    fig = px.scatter(
                        df,
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
                        fig = px.line(
                            df,
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
                    
                    fig = px.box(
                        df,
                        y=y_col,
                        x=None if x_col == "None" else x_col,
                        title=f"Box Plot of {y_col}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                elif viz_type == "Violin Plot":
                    y_col = st.selectbox("Select column for violin plot:", numeric_columns)
                    x_col = st.selectbox("Group by (optional):", ["None"] + categorical_columns)
                    
                    fig = px.violin(
                        df,
                        y=y_col,
                        x=None if x_col == "None" else x_col,
                        box=True,
                        title=f"Violin Plot of {y_col}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                elif viz_type == "Heatmap":
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
                        
                        corr_df = df[corr_cols].corr(method=corr_method.lower())
                        
                        fig = px.imshow(
                            corr_df,
                            text_auto=True,
                            color_continuous_scale="RdBu_r",
                            title=f"{corr_method} Correlation Heatmap"
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            with exploration_tab2:
                st.subheader("Correlation Matrix")
                
                if len(numeric_columns) > 1:
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
                else:
                    st.write("Not enough numeric columns for correlation analysis.")
            
            with exploration_tab3:
                st.subheader("Statistical Tests")
                
                test_type = st.selectbox(
                    "Select test type:",
                    ["T-Test (One Sample)", "T-Test (Two Samples)", 
                     "Chi-Square Test", "ANOVA", "Normality Tests"]
                )
                
                if test_type == "T-Test (One Sample)":
                    col = st.selectbox("Select column:", numeric_columns)
                    mu = st.number_input("Hypothesized mean (μ₀):", value=0.0)
                    
                    sample = df[col].dropna()
                    t_stat, p_val = stats.ttest_1samp(sample, mu)
                    
                    st.write("### One-Sample T-Test Results")
                    st.write(f"Null hypothesis (H₀): μ = {mu}")
                    st.write(f"Alternative hypothesis (H₁): μ ≠ {mu}")
                    st.write(f"Sample mean: {sample.mean():.4f}")
                    st.write(f"Sample size: {len(sample)}")
                    st.write(f"T-statistic: {t_stat:.4f}")
                    st.write(f"P-value: {p_val:.4f}")
                    
                    if p_val < 0.05:
                        st.write("Conclusion: **Reject** the null hypothesis (p < 0.05)")
                    else:
                        st.write("Conclusion: **Fail to reject** the null hypothesis (p >= 0.05)")
                
                elif test_type == "T-Test (Two Samples)":
                    col = st.selectbox("Select numeric column:", numeric_columns)
                    group_col = st.selectbox("Select grouping column:", categorical_columns)
                    
                    # Get unique values in the grouping column
                    unique_groups = df[group_col].unique()
                    
                    if len(unique_groups) < 2:
                        st.error("Need at least 2 groups for two-sample t-test")
                    else:
                        group1 = st.selectbox("Select first group:", unique_groups)
                        remaining_groups = [g for g in unique_groups if g != group1]
                        group2 = st.selectbox("Select second group:", remaining_groups)
                        
                        sample1 = df[df[group_col] == group1][col].dropna()
                        sample2 = df[df[group_col] == group2][col].dropna()
                        
                        equal_var = st.checkbox("Assume equal variances", value=True)
                        
                        t_stat, p_val = stats.ttest_ind(sample1, sample2, equal_var=equal_var)
                        
                        st.write("### Two-Sample T-Test Results")
                        st.write(f"Null hypothesis (H₀): μ₁ = μ₂")
                        st.write(f"Alternative hypothesis (H₁): μ₁ ≠ μ₂")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Group 1 ({group1})**")
                            st.write(f"Mean: {sample1.mean():.4f}")
                            st.write(f"Size: {len(sample1)}")
                            st.write(f"Std Dev: {sample1.std():.4f}")
                        
                        with col2:
                            st.write(f"**Group 2 ({group2})**")
                            st.write(f"Mean: {sample2.mean():.4f}")
                            st.write(f"Size: {len(sample2)}")
                            st.write(f"Std Dev: {sample2.std():.4f}")
                        
                        st.write(f"T-statistic: {t_stat:.4f}")
                        st.write(f"P-value: {p_val:.4f}")
                        
                        if p_val < 0.05:
                            st.write("Conclusion: **Reject** the null hypothesis (p < 0.05)")
                            st.write("There is a significant difference between the two groups.")
                        else:
                            st.write("Conclusion: **Fail to reject** the null hypothesis (p >= 0.05)")
                            st.write("There is no significant difference between the two groups.")
                        
                        # Visualize the comparison
                        fig = px.box(
                            df[df[group_col].isin([group1, group2])],
                            x=group_col,
                            y=col,
                            title=f"Comparison of {col} between groups"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                elif test_type == "Chi-Square Test":
                    if len(categorical_columns) < 2:
                        st.error("Need at least 2 categorical columns for Chi-Square test")
                    else:
                        col1 = st.selectbox("Select first categorical column:", categorical_columns)
                        remaining_cols = [c for c in categorical_columns if c != col1]
                        col2 = st.selectbox("Select second categorical column:", remaining_cols)
                        
                        # Create contingency table
                        contingency_table = pd.crosstab(df[col1], df[col2])
                        
                        # Run Chi-Square test
                        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
                        
                        st.write("### Chi-Square Test of Independence")
                        st.write(f"Null hypothesis (H₀): {col1} and {col2} are independent")
                        st.write(f"Alternative hypothesis (H₁): {col1} and {col2} are not independent")
                        
                        st.write("#### Contingency Table (Observed Frequencies)")
                        st.dataframe(contingency_table)
                        
                        st.write("#### Expected Frequencies (if variables were independent)")
                        st.dataframe(pd.DataFrame(
                            expected, 
                            index=contingency_table.index, 
                            columns=contingency_table.columns
                        ).round(2))
                        
                        st.write(f"Chi-square statistic: {chi2:.4f}")
                        st.write(f"Degrees of freedom: {dof}")
                        st.write(f"P-value: {p:.4f}")
                        
                        if p < 0.05:
                            st.write("Conclusion: **Reject** the null hypothesis (p < 0.05)")
                            st.write("There is a significant association between the variables.")
                        else:
                            st.write("Conclusion: **Fail to reject** the null hypothesis (p >= 0.05)")
                            st.write("There is no significant association between the variables.")
                        
                        # Visualize the relationship
                        st.write("#### Visualization")
                        
                        # Normalize the contingency table to show percentages
                        normalized_table = contingency_table.div(contingency_table.sum(axis=1), axis=0) * 100
                        
                        fig = px.imshow(
                            normalized_table,
                            text_auto='.1f',
                            labels=dict(x=col2, y=col1, color="Percentage (%)"),
                            title=f"Heatmap of {col1} vs {col2}",
                            color_continuous_scale="Blues"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                elif test_type == "ANOVA":
                    if len(categorical_columns) < 1:
                        st.error("Need at least 1 categorical column for ANOVA")
                    else:
                        num_col = st.selectbox("Select numeric column (dependent variable):", numeric_columns)
                        cat_col = st.selectbox("Select categorical column (groups):", categorical_columns)
                        
                        # Get groups
                        groups = []
                        group_names = []
                        
                        for group_name in df[cat_col].unique():
                            group_data = df[df[cat_col] == group_name][num_col].dropna()
                            if len(group_data) > 0:
                                groups.append(group_data)
                                group_names.append(group_name)
                        
                        if len(groups) < 2:
                            st.error("Need at least 2 groups with data for ANOVA")
                        else:
                            # Run ANOVA
                            f_stat, p_val = stats.f_oneway(*groups)
                            
                            st.write("### One-way ANOVA Results")
                            st.write(f"Null hypothesis (H₀): All group means are equal")
                            st.write(f"Alternative hypothesis (H₁): At least one group mean is different")
                            
                            # Summary statistics by group
                            summary_data = []
                            for i, group in enumerate(groups):
                                summary_data.append({
                                    'Group': group_names[i],
                                    'Count': len(group),
                                    'Mean': group.mean(),
                                    'Std Dev': group.std(),
                                    'Min': group.min(),
                                    'Max': group.max()
                                })
                            
                            summary_df = pd.DataFrame(summary_data)
                            st.write("#### Group Statistics")
                            st.dataframe(summary_df.style.format({
                                'Mean': '{:.4f}',
                                'Std Dev': '{:.4f}',
                                'Min': '{:.4f}',
                                'Max': '{:.4f}'
                            }))
                            
                            st.write(f"F-statistic: {f_stat:.4f}")
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
                
                elif test_type == "Normality Tests":
                    col = st.selectbox("Select column to test for normality:", numeric_columns)
                    
                    sample = df[col].dropna()
                    
                    st.write("### Normality Test Results")
                    
                    # Shapiro-Wilk test (limited to 5000 samples)
                    if len(sample) > 5000:
                        st.write("Note: Shapiro-Wilk test is limited to 5000 samples. Using a random subset.")
                        sample_shapiro = sample.sample(5000, random_state=42)
                    else:
                        sample_shapiro = sample
                    
                    shapiro_stat, shapiro_p = stats.shapiro(sample_shapiro)
                    
                    # D'Agostino's K^2 test
                    k2_stat, k2_p = stats.normaltest(sample)
                    
                    # Kolmogorov-Smirnov test
                    ks_stat, ks_p = stats.kstest(
                        (sample - sample.mean()) / sample.std(),
                        'norm'
                    )
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("#### Test Statistics")
                        test_results = pd.DataFrame({
                            'Test': ['Shapiro-Wilk', "D'Agostino's K²", 'Kolmogorov-Smirnov'],
                            'Statistic': [shapiro_stat, k2_stat, ks_stat],
                            'p-value': [shapiro_p, k2_p, ks_p],
                            'Normal?': [
                                'No' if shapiro_p < 0.05 else 'Yes',
                                'No' if k2_p < 0.05 else 'Yes',
                                'No' if ks_p < 0.05 else 'Yes'
                            ]
                        })
                        
                        st.dataframe(test_results.style.format({
                            'Statistic': '{:.4f}',
                            'p-value': '{:.4f}'
                        }))
                        
                        st.write("Note: p < 0.05 suggests the data is not normally distributed.")
                    
                    with col2:
                        # Q-Q Plot
                        fig, ax = plt.subplots(figsize=(10, 6))
                        stats.probplot(sample, plot=ax)
                        ax.set_title(f"Q-Q Plot for {col}")
                        st.pyplot(fig)
                    
                    # Histogram with normal curve
                    fig = plt.figure(figsize=(10, 6))
                    sns.histplot(sample, kde=True, stat="density")
                    
                    # Add a normal curve
                    xmin, xmax = plt.xlim()
                    x = np.linspace(xmin, xmax, 100)
                    p = stats.norm.pdf(x, sample.mean(), sample.std())
                    plt.plot(x, p, 'k', linewidth=2)
                    
                    plt.title(f"Histogram with Normal Curve for {col}")
                    st.pyplot(fig)
                    
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
                        st.metric("Sample Size", f"{len(sample)}")
            
            # Data transformation section
            st.header("Data Transformation")
            
            transform_tab1, transform_tab2 = st.tabs(["Basic Transformations", "Advanced Transformations"])
            
            with transform_tab1:
                st.subheader("Apply Basic Transformations")
                
                transform_col = st.selectbox("Select column to transform:", numeric_columns)
                transform_type = st.selectbox(
                    "Select transformation:",
                    ["None", "Log", "Square Root", "Square", "Cube", "Z-Score", "Min-Max Scaling"]
                )
                
                if transform_type != "None":
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
                        
                        # Apply transformation
                        if transform_type == "Log":
                            # Handle negative or zero values
                            min_val = df[transform_col].min()
                            offset = 0
                            if min_val <= 0:
                                offset = abs(min_val) + 1
                                st.info(f"Adding {offset} to all values before log transformation to handle non-positive values.")
                            
                            transformed = np.log(df[transform_col] + offset)
                            transform_name = f"Log({transform_col})"
                        
                        elif transform_type == "Square Root":
                            # Handle negative values
                            min_val = df[transform_col].min()
                            offset = 0
                            if min_val < 0:
                                offset = abs(min_val) + 0.01
                                st.info(f"Adding {offset} to all values before square root transformation to handle negative values.")
                            
                            transformed = np.sqrt(df[transform_col] + offset)
                            transform_name = f"Sqrt({transform_col})"
                        
                        elif transform_type == "Square":
                            transformed = df[transform_col] ** 2
                            transform_name = f"{transform_col}²"
                        
                        elif transform_type == "Cube":
                            transformed = df[transform_col] ** 3
                            transform_name = f"{transform_col}³"
                        
                        elif transform_type == "Z-Score":
                            transformed = (df[transform_col] - df[transform_col].mean()) / df[transform_col].std()
                            transform_name = f"Z({transform_col})"
                        
                        elif transform_type == "Min-Max Scaling":
                            min_val = df[transform_col].min()
                            max_val = df[transform_col].max()
                            transformed = (df[transform_col] - min_val) / (max_val - min_val)
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
                            shapiro_stat, shapiro_p = stats.shapiro(transformed.dropna().sample(
                                min(5000, len(transformed.dropna())), 
                                random_state=42
                            ))
                            
                            st.write(f"Shapiro-Wilk Test: statistic={shapiro_stat:.4f}, p-value={shapiro_p:.4f}")
                            
                            if shapiro_p < 0.05:
                                st.write("The transformed data is still not normally distributed (p < 0.05).")
                            else:
                                st.write("The transformed data appears to be normally distributed (p >= 0.05).")
            
            with transform_tab2:
                st.subheader("Advanced Transformations")
                
                advanced_transform = st.selectbox(
                    "Select advanced transformation:",
                    ["Box-Cox Transformation", "Yeo-Johnson Transformation", "Quantile Transformation", 
                     "Power Transformation", "Robust Scaling"]
                )
                
                transform_col = st.selectbox("Select column to transform:", numeric_columns, key="adv_transform_col")
                
                if advanced_transform:
                    from scipy import stats
                    from sklearn import preprocessing
                    
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
                        data = df[transform_col].dropna().values.reshape(-1, 1)
                        
                        try:
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
                                shapiro_stat, shapiro_p = stats.shapiro(sample)
                                
                                st.write(f"Shapiro-Wilk Test: statistic={shapiro_stat:.4f}, p-value={shapiro_p:.4f}")
                                
                                if shapiro_p < 0.05:
                                    st.write("The transformed data is still not normally distributed (p < 0.05).")
                                else:
                                    st.write("The transformed data appears to be normally distributed (p >= 0.05).")
                        
                        except Exception as e:
                            st.error(f"Error during transformation: {str(e)}")
            
            # Download transformed data
            if st.button("Generate Transformed Dataset"):
                st.session_state.show_download = True
            
            if st.session_state.get('show_download', False):
                st.subheader("Download Transformed Data")
                
                cols_to_transform = st.multiselect(
                    "Select columns to transform:",
                    numeric_columns
                )
                
                transform_options = {
                    "None": lambda x: x,
                    "Log": lambda x: np.log(x - x.min() + 1 if x.min() <= 0 else x),
                    "Square Root": lambda x: np.sqrt(x - x.min() + 0.01 if x.min() < 0 else x),
                    "Square": lambda x: x ** 2,
                    "Cube": lambda x: x ** 3,
                    "Z-Score": lambda x: (x - x.mean()) / x.std(),
                    "Min-Max Scaling": lambda x: (x - x.min()) / (x.max() - x.min())
                }
                
                transformed_df = df.copy()
                
                for col in cols_to_transform:
                    transform_type = st.selectbox(
                        f"Transformation for {col}:",
                        list(transform_options.keys()),
                        key=f"transform_{col}"
                    )
                    
                    if transform_type != "None":
                        transformed_df[f"{col}_{transform_type}"] = transform_options[transform_type](df[col])
                
                # Convert to CSV
                csv = transformed_df.to_csv(index=False)
                
                st.download_button(
                    label="Download Transformed Data as CSV",
                    data=csv,
                    file_name="transformed_data.csv",
                    mime="text/csv"
                )
        
        except Exception as e:
            st.error(f"Error processing the data: {str(e)}")
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
            
            # Display success message
            st.success("Sample data loaded! You can now explore the dashboard features.")
            
            # Display data preview
            st.write("### Sample Data Preview")
            st.dataframe(sample_df.head())
            
            # Offer option to continue with sample data
            if st.button("Analyze Sample Data"):
                # Use the sample data directly
                st.session_state.use_sample_data = True
                st.experimental_rerun()

# Initialize session state variables
if 'show_download' not in st.session_state:
    st.session_state.show_download = False

if 'use_sample_data' not in st.session_state:
    st.session_state.use_sample_data = False

# Let's refactor the analysis code into a separate function
def analyze_data(df):
    st.sidebar.success("Data successfully loaded!")
    
    # Display basic information
    with st.expander("Dataset Overview"):
        st.write("### Data Preview")
        st.dataframe(df.head())
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("### Dataset Shape")
            st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        
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
    
    st.sidebar.header("Column Selection")
    selected_column = st.sidebar.selectbox("Select a column for detailed analysis:", df.columns)
    
    # Determine column type
    if selected_column in numeric_columns:
        column_type = "numeric"
    else:
        column_type = "categorical"
    
    # Display summary statistics
    st.header(f"Analysis of: {selected_column}")
    
    if column_type == "numeric":
        # Numeric column analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Summary Statistics")
            stats_df = pd.DataFrame({
                'Statistic': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Q1 (25%)', 'Q3 (75%)', 'IQR', 'Skewness', 'Kurtosis'],
                'Value': [
                    df[selected_column].mean(),
                    df[selected_column].median(),
                    df[selected_column].std(),
                    df[selected_column].min(),
                    df[selected_column].max(),
                    df[selected_column].quantile(0.25),
                    df[selected_column].quantile(0.75),
                    df[selected_column].quantile(0.75) - df[selected_column].quantile(0.25),
                    df[selected_column].skew(),
                    df[selected_column].kurt()
                ]
            })
            st.dataframe(stats_df)
        
        with col2:
            st.subheader("Distribution Plot")
            fig = px.histogram(df, x=selected_column, marginal="box", 
                               title=f"Distribution of {selected_column}")
            st.plotly_chart(fig, use_container_width=True)
        
        # Normality tests
        st.subheader("Normality Tests")
        col1, col2 = st.columns(2)
        
        with col1:
            # Shapiro-Wilk test
            sample = df[selected_column].dropna()
            # Limit sample size for Shapiro-Wilk test (max 5000)
            if len(sample) > 5000:
                sample = sample.sample(5000, random_state=42)
            
            shapiro_test = stats.shapiro(sample)
            st.write("**Shapiro-Wilk Test**")
            st.write(f"Statistic: {shapiro_test[0]:.4f}")
            st.write(f"p-value: {shapiro_test[1]:.4f}")
            if shapiro_test[1] < 0.05:
                st.write("Conclusion: Data is **not normally distributed** (p < 0.05)")
            else:
                st.write("Conclusion: Data appears to be normally distributed (p >= 0.05)")
        
        with col2:
            # Q-Q Plot
            st.write("**Q-Q Plot**")
            fig, ax = plt.subplots(figsize=(8, 4))
            stats.probplot(df[selected_column].dropna(), plot=ax)
            st.pyplot(fig)
        
        # Outlier detection
        st.subheader("Outlier Detection")
        
        q1 = df[selected_column].quantile(0.25)
        q3 = df[selected_column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = df[(df[selected_column] < lower_bound) | (df[selected_column] > upper_bound)]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"IQR Method (1.5 × IQR)")
            st.write(f"Lower bound: {lower_bound:.4f}")
            st.write(f"Upper bound: {upper_bound:.4f}")
            st.write(f"Number of outliers: {len(outliers)}")
            st.write(f"Percentage of outliers: {(len(outliers) / len(df) * 100):.2f}%")
        
        with col2:
            # Box plot for outliers
            fig = px.box(df, y=selected_column, title=f"Box Plot of {selected_column}")
            st.plotly_chart(fig, use_container_width=True)
        
        if len(outliers) > 0 and len(outliers) < 100:
            with st.expander("View Outliers"):
                st.dataframe(outliers)
        
        # Correlation analysis
        if len(numeric_columns) > 1:
            st.subheader("Correlation Analysis")
            
            corr_cols = st.multiselect(
                "Select columns for correlation analysis:",
                numeric_columns,
                default=[selected_column] + [col for col in numeric_columns if col != selected_column][:4]
            )
            
            if len(corr_cols) > 1:
                corr_method = st.radio(
                    "Correlation method:",
                    ["Pearson", "Spearman", "Kendall"],
                    horizontal=True
                )
                
                corr_df = df[corr_cols].corr(method=corr_method.lower())
                
                fig = px.imshow(
                    corr_df,
                    text_auto=True,
                    color_continuous_scale="RdBu_r",
                    title=f"{corr_method} Correlation Matrix"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show correlation with selected column
                if len(corr_cols) > 2:
                    st.write(f"### Correlation with {selected_column}")
                    corr_with_selected = corr_df[selected_column].drop(selected_column).sort_values(ascending=False)
                    
                    fig = px.bar(
                        x=corr_with_selected.index,
                        y=corr_with_selected.values,
                        title=f"Correlation with {selected_column}",
                        labels={"x": "Column", "y": f"{corr_method} Correlation"}
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
    else:
        # Categorical column analysis
        st.subheader("Frequency Distribution")
        
        value_counts = df[selected_column].value_counts().reset_index()
        value_counts.columns = [selected_column, 'Count']
        value_counts['Percentage'] = (value_counts['Count'] / value_counts['Count'].sum() * 100).round(2)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(value_counts)
            st.write(f"Number of unique values: {df[selected_column].nunique()}")
        
        with col2:
            # Bar chart of frequency
            fig = px.bar(
                value_counts,
                x=selected_column,
                y='Count',
                title=f"Frequency Distribution of {selected_column}"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Pie chart
        if df[selected_column].nunique() <= 10:
            fig = px.pie(
                value_counts, 
                values='Count', 
                names=selected_column,
                title=f"Proportion of Categories in {selected_column}"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Relationship with numeric columns
        if numeric_columns:
            st.subheader(f"Relationship with Numeric Variables")
            
            numeric_col = st.selectbox(
                "Select a numeric column to analyze relationship:",
                numeric_columns
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Box plot
                fig = px.box(
                    df,
                    x=selected_column,
                    y=numeric_col,
                    title=f"{numeric_col} by {selected_column}"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Bar chart with mean
                agg_df = df.groupby(selected_column)[numeric_col].agg(['mean', 'median']).reset_index()
                
                fig = px.bar(
                    agg_df,
                    x=selected_column,
                    y='mean',
                    error_y=agg_df['median'],
                    title=f"Mean {numeric_col} by {selected_column}"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # ANOVA test if there are enough categories
            if 2 <= df[selected_column].nunique() <= 10:
                st.subheader("ANOVA Test")
                st.write("Testing if the mean of the numeric variable differs between categories")
                
                groups = [df[df[selected_column] == cat][numeric_col].dropna() 
                         for cat in df[selected_column].unique()]
                
                # Only perform ANOVA if we have valid groups
                valid_groups = [g for g in groups if len(g) > 0]
                
                if len(valid_groups) >= 2:
                    f_stat, p_val = stats.f_oneway(*valid_groups)
                    
                    st.write(f"F-statistic: {f_stat:.4f}")
                    st.write(f"p-value: {p_val:.4f}")
                    
                    if p_val < 0.05:
                        st.write("Conclusion: There is a **significant difference** between groups (p < 0.05)")
                    else:
                        st.write("Conclusion: There is no significant difference between groups (p >= 0.05)")
                else:
                    st.write("Cannot perform ANOVA: not enough valid groups")
    
    # Data exploration section
    st.header("Advanced Data Exploration")
    
    exploration_tab1, exploration_tab2, exploration_tab3 = st.tabs([
        "Custom Visualization", "Correlation Matrix", "Statistical Tests"
    ])
    
    # Add all the exploration tabs code here...
    # [Include all the code for the exploration tabs]
    
    # Data transformation section
    st.header("Data Transformation")
    
    transform_tab1, transform_tab2 = st.tabs(["Basic Transformations", "Advanced Transformations"])
    
    # Add all the transformation tabs code here...
    # [Include all the code for the transformation tabs]
    
    # Download transformed data section
    # [Include the download section code]


# Main function to run the app
def run_app():
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            analyze_data(df)
        except Exception as e:
            st.error(f"Error processing the data: {str(e)}")
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
            st.experimental_rerun()


# Initialize session state variables
if 'show_download' not in st.session_state:
    st.session_state.show_download = False

if 'use_sample_data' not in st.session_state:
    st.session_state.use_sample_data = False

# Run the app
if st.session_state.get('use_sample_data', False) and 'sample_data' in st.session_state:
    # If we're using sample data, process it directly
    df = st.session_state.sample_data
    analyze_data(df)
else:
    # Normal flow - will use uploaded_file from sidebar
    run_app()

