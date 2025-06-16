import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats

from utils.statistics import handle_missing_values

def analyze_categorical_column(df, selected_column, numeric_columns, categorical_columns):
    """Analyze a categorical column"""
    # Check for missing values
    data_no_na, warning = handle_missing_values(df, selected_column)
    if data_no_na is None:
        st.error(warning)
        return
    if warning:
        st.warning(warning)
    
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
    analyze_relationship_with_numeric(df, selected_column, numeric_columns, data_no_na)

def analyze_relationship_with_numeric(df, selected_column, numeric_columns, data_no_na):
    """Analyze relationship between categorical column and numeric columns"""
    if not numeric_columns:
        return
        
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
    run_anova_test(df, selected_column, numeric_col)

def run_anova_test(df, categorical_col, numeric_col):
    """Run ANOVA test to compare means across categories"""
    if 2 <= df[categorical_col].nunique() <= 20:
        with st.expander("ANOVA Test", expanded=True):
            st.write("Testing if the mean of the numeric variable differs between categories")
            
            # Get groups, handling potential issues
            groups = []
            group_names = []
            
            for cat in df[categorical_col].dropna().unique():
                group_data = df[df[categorical_col] == cat][numeric_col].dropna()
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
                
                except Exception as e:
                    st.error(f"Error performing ANOVA test: {str(e)}")

