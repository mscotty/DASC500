import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from scipy import stats
import traceback

from utils.statistics import safe_statistical_test

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
            run_one_sample_ttest(df, numeric_columns)
        
        elif test_type == "T-Test (Two Samples)":
            run_two_sample_ttest(df, numeric_columns, categorical_columns)
        
        elif test_type == "Chi-Square Test":
            run_chi_square_test(df, categorical_columns)
        
        elif test_type == "ANOVA":
            run_anova_test(df, numeric_columns, categorical_columns)
        
        elif test_type == "Normality Tests":
            run_normality_tests(df, numeric_columns)
    
    except Exception as e:
        st.error(f"Error in statistical test: {str(e)}")
        st.error(f"Details: {traceback.format_exc()}")

def run_one_sample_ttest(df, numeric_columns):
    """Run one-sample t-test"""
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

def run_two_sample_ttest(df, numeric_columns, categorical_columns):
    """Run two-sample t-test"""
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
        st.write("There is a significant difference between the two groups.")
    else:
        st.write("Conclusion: **Fail to reject** the null hypothesis (p >= 0.05)")
        st.write("There is no significant difference between the two groups.")
    
    # Visualize the comparison
    fig = px.box(
        df[df[group_col].isin([group1, group2])],
        x=group_col,
        y=col,
        title=f"Comparison of {col} between groups",
        points="all" if max(len(sample1), len(sample2)) < 100 else "outliers"
    )
    st.plotly_chart(fig, use_container_width=True)

def run_chi_square_test(df, categorical_columns):
    """Run chi-square test of independence"""
    if len(categorical_columns) < 2:
        st.error("Need at least 2 categorical columns for Chi-Square test.")
        return
        
    col1 = st.selectbox("Select first categorical column:", categorical_columns)
    remaining_cols = [c for c in categorical_columns if c != col1]
    col2 = st.selectbox("Select second categorical column:", remaining_cols)
    
    # Check if columns have too many categories
    if df[col1].nunique() > 20 or df[col2].nunique() > 20:
        st.warning(f"One or both columns have many categories ({df[col1].nunique():,} and {df[col2].nunique():,}). Consider using columns with fewer categories for more meaningful results.")
    
    # Create contingency table
    contingency_table = pd.crosstab(df[col1], df[col2])
    
    # Run Chi-Square test
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    
    st.write("### Chi-Square Test of Independence")
    st.write(f"Null hypothesis (H₀): {col1} and {col2} are independent")
    st.write(f"Alternative hypothesis (H₁): {col1} and {col2} are not independent")
    
    with st.expander("View Contingency Table", expanded=True):
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

def run_anova_test(df, numeric_columns, categorical_columns):
    """Run one-way ANOVA test"""
    if not numeric_columns:
        st.warning("No numeric columns available for ANOVA.")
        return
    if not categorical_columns:
        st.warning("No categorical columns available for grouping.")
        return
        
    num_col = st.selectbox("Select numeric column (dependent variable):", numeric_columns)
    cat_col = st.selectbox("Select categorical column (groups):", categorical_columns)
    
    # Get groups
    groups = []
    group_names = []
    
    for group_name in df[cat_col].dropna().unique():
        group_data = df[df[cat_col] == group_name][num_col].dropna()
        if len(group_data) > 0:
            groups.append(group_data)
            group_names.append(group_name)
    
    if len(groups) < 2:
        st.error("Need at least 2 groups with data for ANOVA.")
        return
        
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
            'Std Dev': group.std()
        })
    
    summary_df = pd.DataFrame(summary_data)
    st.write("#### Group Statistics")
    st.dataframe(summary_df.style.format({
        'Mean': '{:.4f}',
        'Std Dev': '{:.4f}'
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

def run_normality_tests(df, numeric_columns):
    """Run normality tests on numeric columns"""
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
        import seaborn as sns
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
