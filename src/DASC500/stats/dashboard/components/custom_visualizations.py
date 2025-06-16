import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

def create_custom_visualization(df, numeric_columns, categorical_columns):
    """Create custom visualizations"""
    st.subheader("Create Custom Visualization")
    
    viz_type = st.selectbox(
        "Select visualization type:",
        ["Scatter Plot", "Line Chart", "Bar Chart", "Histogram", "Box Plot", "Violin Plot", "Heatmap"]
    )
    
    try:
        if viz_type == "Scatter Plot":
            create_scatter_plot(df, numeric_columns, categorical_columns)
        
        elif viz_type == "Line Chart":
            create_line_chart(df, numeric_columns)
        
        elif viz_type == "Bar Chart":
            create_bar_chart(df, numeric_columns)
        
        elif viz_type == "Histogram":
            create_histogram(df, numeric_columns)
        
        elif viz_type == "Box Plot":
            create_box_plot(df, numeric_columns, categorical_columns)
        
        elif viz_type == "Violin Plot":
            create_violin_plot(df, numeric_columns, categorical_columns)
        
        elif viz_type == "Heatmap":
            create_heatmap(df, numeric_columns)
    
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")

def create_scatter_plot(df, numeric_columns, categorical_columns):
    """Create a scatter plot"""
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

def create_line_chart(df, numeric_columns):
    """Create a line chart"""
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

def create_bar_chart(df, numeric_columns):
    """Create a bar chart"""
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

def create_histogram(df, numeric_columns):
    """Create a histogram"""
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

def create_box_plot(df, numeric_columns, categorical_columns):
    """Create a box plot"""
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

def create_violin_plot(df, numeric_columns, categorical_columns):
    """Create a violin plot"""
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
