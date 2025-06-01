import numpy as np
import pandas as pd
from scipy.stats import zscore
from copy import deepcopy
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

def compute_z_scores_df(df, columns):
    """
    Computes the z-scores for the specified columns in the DataFrame.

    Parameters:
    -----------
    df : pandas DataFrame
        The input DataFrame
    columns : list of str
        The columns for which to compute z-scores

    Returns:
    --------
    pandas DataFrame
        A DataFrame containing the z-scores of the specified columns
    """
    # Filter to only include columns that exist in the DataFrame
    valid_columns = [col for col in columns if col in df.columns]
    
    if not valid_columns:
        return pd.DataFrame()
    
    # Handle potential NaN values by computing z-scores for each column separately
    z_scores_df = pd.DataFrame(index=df.index)
    
    for col in valid_columns:
        # Skip columns with constant values (std=0)
        if df[col].std() == 0:
            z_scores_df[col] = 0
        else:
            # Calculate z-scores, handling NaN values
            z_scores_df[col] = (df[col] - df[col].mean()) / df[col].std()
    
    return z_scores_df


def flag_outliers_using_z_score(df, columns, threshold=3.0, output_column='is_outlier'):
    """
    Flags outliers in the DataFrame based on z-score analysis, only updating the output column 
    if its current value is False.

    Parameters:
    -----------
    df : pandas DataFrame
        The input DataFrame
    columns : list of str
        The columns to analyze for outliers
    threshold : float, default=3.0
        The z-score threshold to determine outliers
    output_column : str, default='is_outlier'
        The column in which to store flagged outliers

    Returns:
    --------
    pandas DataFrame
        A DataFrame with the updated outlier flagging
    """
    df_flagged = deepcopy(df)
    
    # Initialize output column if it doesn't exist
    if output_column not in df_flagged.columns:
        df_flagged[output_column] = False
    
    # Compute z-scores for the specified columns
    z_scores = compute_z_scores_df(df_flagged, columns)
    
    if z_scores.empty:
        return df_flagged
    
    # Determine which rows exceed the threshold
    flagged_rows = (np.abs(z_scores) > threshold).any(axis=1)
    
    # Update the output column only if it was previously False
    df_flagged.loc[~df_flagged[output_column], output_column] = flagged_rows[~df_flagged[output_column]]
    
    return df_flagged


def identify_outliers_by_zscore(df, columns, threshold=3.0):
    """
    Identifies outliers in the DataFrame based on z-score analysis.

    Parameters:
    -----------
    df : pandas DataFrame
        The input DataFrame
    columns : list of str
        The columns to analyze for outliers
    threshold : float, default=3.0
        The z-score threshold to determine outliers

    Returns:
    --------
    pandas DataFrame
        A DataFrame containing only the outlier rows with additional columns for z-scores
    """
    # Compute z-scores for the specified columns
    z_scores = compute_z_scores_df(df, columns)
    
    if z_scores.empty:
        return pd.DataFrame()
    
    # Identify which z-scores exceed the threshold
    outlier_mask = (np.abs(z_scores) > threshold).any(axis=1)
    
    if not outlier_mask.any():
        return pd.DataFrame()
    
    # Create a new DataFrame with original values and their z-scores
    outliers_df = df.loc[outlier_mask, columns].copy()
    
    # Add z-score columns
    for col in columns:
        if col in z_scores.columns:
            outliers_df[f'{col}_zscore'] = z_scores.loc[outlier_mask, col]
    
    # Add a column indicating which values are outliers
    for col in columns:
        if col in z_scores.columns:
            outliers_df[f'{col}_is_outlier'] = np.abs(z_scores.loc[outlier_mask, col]) > threshold
    
    return outliers_df


def iterative_z_score_outlier_removal(series, z_threshold=3.0, max_iterations=10, 
                                     high_side_iterations=5, convergence_tol=1e-3):
    """
    Iteratively removes outliers using a z-score approach, prioritizing high-side outliers first.
    Useful for aerodynamic parameters like dynamic pressure.

    Parameters:
    -----------
    series : pandas Series
        The data series to analyze
    z_threshold : float, default=3.0
        Z-score threshold for outlier detection
    max_iterations : int, default=10
        Maximum iterations for full z-score filtering
    high_side_iterations : int, default=5
        Number of iterations dedicated to filtering only high-side outliers
    convergence_tol : float, default=1e-3
        Tolerance for stopping early if min/max values stabilize

    Returns:
    --------
    tuple
        (min_value, max_value) - The final data range after outlier removal
    """
    data = series.dropna().copy()
    
    if len(data) < 3:  # Need at least 3 points for meaningful statistics
        return data.min() if not data.empty else None, data.max() if not data.empty else None
    
    prev_min, prev_max = None, None  # Track previous min/max for convergence check

    # Step 1: Remove high-side outliers first
    for _ in range(high_side_iterations):
        mean = np.mean(data)
        std = np.std(data)

        if std == 0:
            break  # Stop if no variation remains

        z_scores = (data - mean) / std
        mask = z_scores <= z_threshold  # Remove only high-side outliers

        if not mask.any():  # Avoid empty dataset
            break

        new_min, new_max = data[mask].min(), data[mask].max()
        
        # Check for convergence (if min/max stops changing)
        if prev_min is not None and abs(new_max - prev_max) < convergence_tol:
            break

        data = data[mask]  # Update dataset for next iteration
        prev_min, prev_max = new_min, new_max

    # Step 2: Full symmetric z-score filtering
    for _ in range(max_iterations):
        mean = np.mean(data)
        std = np.std(data)

        if std == 0:
            break  # Stop if no variation remains

        z_scores = (data - mean) / std
        mask = np.abs(z_scores) <= z_threshold  # Remove both high and low outliers

        if not mask.any():  # Avoid empty dataset
            break

        new_min, new_max = data[mask].min(), data[mask].max()
        
        # Check for convergence (if min/max stops changing)
        if prev_min is not None and abs(new_min - prev_min) < convergence_tol and abs(new_max - prev_max) < convergence_tol:
            break

        data = data[mask]  # Update dataset for next iteration
        prev_min, prev_max = new_min, new_max

    return data.min(), data.max()


def compute_adaptive_lower_bound(series, percentile=10):
    """
    Computes an adaptive lower bound for a series based on a small percentile 
    of the data, ensuring it does not approach zero.
    Useful for parameters like dynamic pressure in aerodynamic data.
    
    Parameters:
    -----------
    series : pandas Series
        The data series to analyze
    percentile : float, default=10
        Percentile value for the lower bound calculation
        
    Returns:
    --------
    float
        Lower bound for the series
    """
    clean_series = series.dropna()
    
    if clean_series.empty:
        return None
    
    min_reasonable_val = np.percentile(clean_series, percentile)
    return max(min_reasonable_val, 1.0)  # Ensure a physically reasonable lower bound


def plot_z_score_distribution(df, columns, output_dir=None, show_plots=True):
    """
    Creates plots showing the distribution of z-scores for selected columns.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The input DataFrame
    columns : list of str
        The columns to analyze
    output_dir : str, optional
        Directory to save output files and plots
    show_plots : bool, default=True
        Whether to display plots
        
    Returns:
    --------
    dict
        Dictionary with plot paths
    """
    plot_paths = {}
    
    # Compute z-scores
    z_scores = compute_z_scores_df(df, columns)
    
    if z_scores.empty:
        return plot_paths
    
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Create distribution plot for each column
    for col in z_scores.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot histogram with KDE
        sns.histplot(z_scores[col].dropna(), kde=True, ax=ax)
        
        # Add vertical lines at common thresholds
        ax.axvline(x=2, color='orange', linestyle='--', alpha=0.7, label='Z = 2')
        ax.axvline(x=3, color='red', linestyle='--', alpha=0.7, label='Z = 3')
        ax.axvline(x=-2, color='orange', linestyle='--', alpha=0.7)
        ax.axvline(x=-3, color='red', linestyle='--', alpha=0.7)
        
        ax.set_title(f'Z-Score Distribution for {col}')
        ax.set_xlabel('Z-Score')
        ax.set_ylabel('Frequency')
        ax.legend()
        
        # Save plot if output directory is provided
        if output_dir:
            plot_path = os.path.join(output_dir, f'zscore_dist_{col}.png')
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            plot_paths[col] = plot_path
        
        if show_plots:
            plt.show()
        else:
            plt.close(fig)
    
    return plot_paths


def detect_outliers_zscore_workflow(df, columns=None, threshold=3.0, output_dir=None, 
                                   save_csv=True, plot_distributions=True, show_plots=True):
    """
    Complete workflow for detecting outliers using z-score method.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The input DataFrame
    columns : list of str, optional
        The columns to analyze (default: all numerical columns)
    threshold : float, default=3.0
        Z-score threshold for outlier detection
    output_dir : str, optional
        Directory to save output files and plots
    save_csv : bool, default=True
        Whether to save outliers to CSV file
    plot_distributions : bool, default=True
        Whether to create z-score distribution plots
    show_plots : bool, default=True
        Whether to display plots
        
    Returns:
    --------
    dict
        Dictionary containing outlier analysis results
    """
    # Set up logging
    logger = logging.getLogger('zscore_outlier_detection')
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
            file_handler = logging.FileHandler(os.path.join(output_dir, 'zscore_outlier_detection.log'))
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    logger.info("\n=== Z-Score Outlier Detection ===")
    
    # Determine columns to analyze
    if columns is None:
        columns = df.select_dtypes(include=np.number).columns.tolist()
    else:
        # Filter to only include numerical columns that exist in the DataFrame
        columns = [col for col in columns if col in df.columns and 
                  np.issubdtype(df[col].dtype, np.number)]
    
    if not columns:
        logger.info("No valid numerical columns found for z-score analysis.")
        return {"outliers_df": pd.DataFrame(), "columns_analyzed": []}
    
    logger.info(f"Analyzing {len(columns)} columns for outliers using z-score threshold: {threshold}")
    
    # Identify outliers
    outliers_df = identify_outliers_by_zscore(df, columns, threshold)
    
    # Log results
    if outliers_df.empty:
        logger.info("No outliers detected.")
    else:
        outlier_count = len(outliers_df)
        logger.info(f"Detected {outlier_count} outliers ({outlier_count/len(df)*100:.2f}% of data)")
        
        # Count outliers by column
        outlier_counts_by_column = {}
        for col in columns:
            outlier_col = f"{col}_is_outlier"
            if outlier_col in outliers_df.columns:
                col_outliers = outliers_df[outlier_col].sum()
                outlier_counts_by_column[col] = col_outliers
                logger.info(f"  - {col}: {col_outliers} outliers")
        
        # Save to CSV if requested
        if save_csv and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            csv_path = os.path.join(output_dir, "zscore_outliers.csv")
            outliers_df.to_csv(csv_path, index=True)
            logger.info(f"Outliers saved to: {csv_path}")
    
    # Create z-score distribution plots if requested
    plot_paths = {}
    if plot_distributions:
        logger.info("\nCreating z-score distribution plots...")
        plot_paths = plot_z_score_distribution(df, columns, output_dir, show_plots)
        if output_dir and plot_paths:
            logger.info(f"Z-score distribution plots saved to: {output_dir}")
    
    # Create summary of results
    results = {
        "outliers_df": outliers_df,
        "columns_analyzed": columns,
        "outlier_count": len(outliers_df) if not outliers_df.empty else 0,
        "outlier_percentage": len(outliers_df)/len(df)*100 if not outliers_df.empty else 0,
        "plot_paths": plot_paths
    }
    
    # Add column-specific outlier counts
    if not outliers_df.empty:
        column_outlier_counts = {}
        for col in columns:
            outlier_col = f"{col}_is_outlier"
            if outlier_col in outliers_df.columns:
                column_outlier_counts[col] = outliers_df[outlier_col].sum()
        results["column_outlier_counts"] = column_outlier_counts
    
    logger.info("\n=== Z-Score Outlier Detection Complete ===")
    
    return results

