import os
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def compute_basic_statistics(df, columns=None):
    """
    Computes basic descriptive statistics for numerical columns.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The DataFrame to analyze
    columns : list of str, optional
        Specific columns to analyze (default: all numerical columns)
        
    Returns:
    --------
    pandas DataFrame
        DataFrame containing basic statistics
    """
    # Select numerical columns if not specified
    if columns is None:
        num_df = df.select_dtypes(include=np.number)
    else:
        # Filter to only include numerical columns that exist
        valid_cols = [col for col in columns if col in df.columns and 
                     np.issubdtype(df[col].dtype, np.number)]
        num_df = df[valid_cols]
    
    if num_df.empty:
        return pd.DataFrame()
    
    # Calculate basic statistics
    desc_stats = num_df.describe().T
    
    return desc_stats


def compute_advanced_statistics(df, columns=None):
    """
    Computes advanced descriptive statistics (skewness, kurtosis, etc.) for numerical columns.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The DataFrame to analyze
    columns : list of str, optional
        Specific columns to analyze (default: all numerical columns)
        
    Returns:
    --------
    pandas DataFrame
        DataFrame containing advanced statistics
    """
    # Select numerical columns if not specified
    if columns is None:
        num_df = df.select_dtypes(include=np.number)
    else:
        # Filter to only include numerical columns that exist
        valid_cols = [col for col in columns if col in df.columns and 
                     np.issubdtype(df[col].dtype, np.number)]
        num_df = df[valid_cols]
    
    if num_df.empty:
        return pd.DataFrame()
    
    # Start with basic statistics
    stats_df = compute_basic_statistics(df, columns)
    
    # Add advanced statistics
    stats_df['skewness'] = num_df.skew()
    stats_df['kurtosis'] = num_df.kurt()
    stats_df['cv'] = stats_df['std'] / stats_df['mean']  # Coefficient of variation
    
    # Add percentiles
    for q_val in [0.01, 0.05, 0.95, 0.99]:
        stats_df[f'{q_val*100:.0f}%'] = num_df.quantile(q_val)
    
    # Add range and IQR
    stats_df['range'] = stats_df['max'] - stats_df['min']
    stats_df['iqr'] = stats_df['75%'] - stats_df['25%']
    
    # Add missing value count and percentage
    stats_df['missing_count'] = num_df.isna().sum()
    stats_df['missing_percent'] = (num_df.isna().sum() / len(num_df)) * 100
    
    return stats_df


def test_normality(df, columns=None, alpha=0.05):
    """
    Tests for normality using Shapiro-Wilk test.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The DataFrame to analyze
    columns : list of str, optional
        Specific columns to test (default: all numerical columns)
    alpha : float, default=0.05
        Significance level for the test
        
    Returns:
    --------
    pandas DataFrame
        DataFrame containing normality test results
    """
    # Select numerical columns if not specified
    if columns is None:
        num_df = df.select_dtypes(include=np.number)
    else:
        # Filter to only include numerical columns that exist
        valid_cols = [col for col in columns if col in df.columns and 
                     np.issubdtype(df[col].dtype, np.number)]
        num_df = df[valid_cols]
    
    if num_df.empty:
        return pd.DataFrame()
    
    # Initialize results DataFrame
    results = pd.DataFrame(index=num_df.columns, 
                          columns=['shapiro_stat', 'shapiro_p_value', 'is_normal'])
    
    # Perform Shapiro-Wilk test for each column
    for col in num_df.columns:
        data = num_df[col].dropna()
        
        # Shapiro-Wilk test requires at least 3 samples
        if len(data) >= 3:
            stat, p_value = stats.shapiro(data)
            results.loc[col, 'shapiro_stat'] = stat
            results.loc[col, 'shapiro_p_value'] = p_value
            results.loc[col, 'is_normal'] = p_value > alpha
        else:
            results.loc[col, 'shapiro_stat'] = np.nan
            results.loc[col, 'shapiro_p_value'] = np.nan
            results.loc[col, 'is_normal'] = np.nan
    
    return results


def create_boxplots(df, columns=None, output_dir=None, show_plots=True, figsize=(15, 10)):
    """
    Creates boxplots for numerical columns.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The DataFrame to analyze
    columns : list of str, optional
        Specific columns to plot (default: all numerical columns)
    output_dir : str, optional
        Directory to save output files
    show_plots : bool, default=True
        Whether to display plots
    figsize : tuple, default=(15, 10)
        Figure size for the plot
        
    Returns:
    --------
    dict
        Dictionary with plot paths
    """
    # Select numerical columns if not specified
    if columns is None:
        num_df = df.select_dtypes(include=np.number)
    else:
        # Filter to only include numerical columns that exist
        valid_cols = [col for col in columns if col in df.columns and 
                     np.issubdtype(df[col].dtype, np.number)]
        num_df = df[valid_cols]
    
    if num_df.empty:
        return {}
    
    plot_paths = {}
    
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Calculate number of rows and columns for subplots
    n_cols = min(3, len(num_df.columns))
    n_rows = (len(num_df.columns) + n_cols - 1) // n_cols
    
    # Create figure and subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    # Create boxplots
    for i, col in enumerate(num_df.columns):
        if i < len(axes):
            sns.boxplot(y=num_df[col], ax=axes[i])
            axes[i].set_title(f'Box Plot of {col}')
            axes[i].set_ylabel(col)
    
    # Hide unused subplots
    for i in range(len(num_df.columns), len(axes)):
        axes[i].set_visible(False)
    
    fig.tight_layout()
    
    # Save or show plot
    if output_dir:
        plot_path = os.path.join(output_dir, 'boxplots.png')
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        plot_paths['boxplots'] = plot_path
    
    if show_plots:
        plt.show()
    else:
        plt.close(fig)
    
    return plot_paths


def create_histograms(df, columns=None, bins=30, kde=True, output_dir=None, show_plots=True, figsize=(15, 10)):
    """
    Creates histograms with optional KDE for numerical columns.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The DataFrame to analyze
    columns : list of str, optional
        Specific columns to plot (default: all numerical columns)
    bins : int, default=30
        Number of bins for histograms
    kde : bool, default=True
        Whether to include KDE curve
    output_dir : str, optional
        Directory to save output files
    show_plots : bool, default=True
        Whether to display plots
    figsize : tuple, default=(15, 10)
        Figure size for the plot
        
    Returns:
    --------
    dict
        Dictionary with plot paths
    """
    # Select numerical columns if not specified
    if columns is None:
        num_df = df.select_dtypes(include=np.number)
    else:
        # Filter to only include numerical columns that exist
        valid_cols = [col for col in columns if col in df.columns and 
                     np.issubdtype(df[col].dtype, np.number)]
        num_df = df[valid_cols]
    
    if num_df.empty:
        return {}
    
    plot_paths = {}
    
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Calculate number of rows and columns for subplots
    n_cols = min(3, len(num_df.columns))
    n_rows = (len(num_df.columns) + n_cols - 1) // n_cols
    
    # Create figure and subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    # Create histograms
    for i, col in enumerate(num_df.columns):
        if i < len(axes):
            sns.histplot(num_df[col].dropna(), bins=bins, kde=kde, ax=axes[i])
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(col)
    
    # Hide unused subplots
    for i in range(len(num_df.columns), len(axes)):
        axes[i].set_visible(False)
    
    fig.tight_layout()
    
    # Save or show plot
    if output_dir:
        plot_path = os.path.join(output_dir, 'histograms.png')
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        plot_paths['histograms'] = plot_path
    
    if show_plots:
        plt.show()
    else:
        plt.close(fig)
    
    return plot_paths


def create_violin_plots(df, columns=None, output_dir=None, show_plots=True, figsize=(15, 10)):
    """
    Creates violin plots for numerical columns.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The DataFrame to analyze
    columns : list of str, optional
        Specific columns to plot (default: all numerical columns)
    output_dir : str, optional
        Directory to save output files
    show_plots : bool, default=True
        Whether to display plots
    figsize : tuple, default=(15, 10)
        Figure size for the plot
        
    Returns:
    --------
    dict
        Dictionary with plot paths
    """
    # Select numerical columns if not specified
    if columns is None:
        num_df = df.select_dtypes(include=np.number)
    else:
        # Filter to only include numerical columns that exist
        valid_cols = [col for col in columns if col in df.columns and 
                     np.issubdtype(df[col].dtype, np.number)]
        num_df = df[valid_cols]
    
    if num_df.empty:
        return {}
    
    plot_paths = {}
    
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Calculate number of rows and columns for subplots
    n_cols = min(3, len(num_df.columns))
    n_rows = (len(num_df.columns) + n_cols - 1) // n_cols
    
    # Create figure and subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    # Create violin plots
    for i, col in enumerate(num_df.columns):
        if i < len(axes):
            sns.violinplot(y=num_df[col], ax=axes[i])
            axes[i].set_title(f'Violin Plot of {col}')
            axes[i].set_ylabel(col)
    
    # Hide unused subplots
    for i in range(len(num_df.columns), len(axes)):
        axes[i].set_visible(False)
    
    fig.tight_layout()
    
    # Save or show plot
    if output_dir:
        plot_path = os.path.join(output_dir, 'violin_plots.png')
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        plot_paths['violin_plots'] = plot_path
    
    if show_plots:
        plt.show()
    else:
        plt.close(fig)
    
    return plot_paths


def calculate_descriptive_statistics(df, columns=None, output_dir=None, 
                                    create_plots=True, plot_types=None,
                                    test_for_normality=True, show_plots=True):
    """
    Comprehensive descriptive statistics analysis with visualization options.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The DataFrame to analyze
    columns : list of str, optional
        Specific columns to analyze (default: all numerical columns)
    output_dir : str, optional
        Directory to save output files
    create_plots : bool, default=True
        Whether to create visualizations
    plot_types : list of str, optional
        Types of plots to create: 'boxplot', 'histogram', 'violin'
        (default: all plot types)
    test_for_normality : bool, default=True
        Whether to perform normality tests
    show_plots : bool, default=True
        Whether to display plots
        
    Returns:
    --------
    dict
        Dictionary containing analysis results
    """
    import os
    import logging
    
    # Set up logging
    logger = logging.getLogger('descriptive_stats')
    logger.setLevel(logging.INFO)
    
    # Create console handler if not already present
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Add file handler if output_dir is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(output_dir, 'descriptive_stats.log'))
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    logger.info("\n=== Enhanced Descriptive Statistics Analysis ===")
    
        # Select numerical columns if not specified
    if columns is None:
        num_df = df.select_dtypes(include=np.number)
        columns = num_df.columns.tolist()
    else:
        # Filter to only include numerical columns that exist
        columns = [col for col in columns if col in df.columns and 
                  np.issubdtype(df[col].dtype, np.number)]
        num_df = df[columns]
    
    if num_df.empty:
        logger.info("No numerical columns for statistics.")
        return {"stats": pd.DataFrame()}
    
    logger.info(f"Analyzing {len(columns)} numerical columns")
    
    # Initialize results dictionary
    results = {}
    
    # Compute advanced statistics
    stats_df = compute_advanced_statistics(df, columns)
    results["stats"] = stats_df
    
    # Log statistics
    logger.info(f"\nDescriptive Statistics:\n{stats_df}")
    
    # Save statistics to CSV if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        stats_path = os.path.join(output_dir, 'descriptive_statistics.csv')
        stats_df.to_csv(stats_path)
        logger.info(f"Statistics saved to: {stats_path}")
    
    # Test for normality if requested
    if test_for_normality:
        normality_df = test_normality(df, columns)
        results["normality_tests"] = normality_df
        
        # Log normality test results
        logger.info("\nNormality Test Results:")
        for col in normality_df.index:
            p_value = normality_df.loc[col, 'shapiro_p_value']
            is_normal = normality_df.loc[col, 'is_normal']
            
            if not np.isnan(p_value):
                logger.info(f"  {col}: p-value = {p_value:.4f} - {'Normal' if is_normal else 'Not normal'}")
            else:
                logger.info(f"  {col}: Insufficient data for normality test")
        
        # Save normality tests to CSV if output directory is provided
        if output_dir:
            normality_path = os.path.join(output_dir, 'normality_tests.csv')
            normality_df.to_csv(normality_path)
            logger.info(f"Normality tests saved to: {normality_path}")
    
    # Create visualizations if requested
    if create_plots:
        plot_results = {}
        
        # Determine which plot types to create
        if plot_types is None:
            plot_types = ['boxplot', 'histogram', 'violin']
        
        # Create plots directory
        plots_dir = os.path.join(output_dir, 'plots') if output_dir else None
        
        # Create boxplots
        if 'boxplot' in plot_types:
            logger.info("\nCreating boxplots...")
            boxplot_paths = create_boxplots(df, columns, plots_dir, show_plots)
            plot_results['boxplots'] = boxplot_paths
        
        # Create histograms
        if 'histogram' in plot_types:
            logger.info("\nCreating histograms...")
            histogram_paths = create_histograms(df, columns, bins=30, kde=True, 
                                               output_dir=plots_dir, show_plots=show_plots)
            plot_results['histograms'] = histogram_paths
        
        # Create violin plots
        if 'violin' in plot_types:
            logger.info("\nCreating violin plots...")
            violin_paths = create_violin_plots(df, columns, plots_dir, show_plots)
            plot_results['violin_plots'] = violin_paths
        
        results['plots'] = plot_results
    
    logger.info("\n=== Descriptive Statistics Analysis Complete ===")
    
    return results
