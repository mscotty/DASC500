import os
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def compute_iqr_bounds(data, k=1.5):
    """
    Computes the IQR bounds for outlier detection.
    
    Parameters:
    -----------
    data : array-like
        Data to analyze
    k : float, default=1.5
        Multiplier for IQR (typically 1.5 for outliers, 3 for extreme outliers)
        
    Returns:
    --------
    tuple
        (q1, q3, lower_bound, upper_bound)
    """
    # Drop NaN values
    clean_data = pd.Series(data).dropna()
    
    if len(clean_data) < 4:  # Need at least a few points for quartiles
        return None, None, None, None
    
    # Compute quartiles
    q1 = clean_data.quantile(0.25)
    q3 = clean_data.quantile(0.75)
    
    # Compute IQR
    iqr = q3 - q1
    
    # Compute bounds
    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr
    
    return q1, q3, lower_bound, upper_bound


def identify_iqr_outliers(data, column_name=None, k=1.5):
    """
    Identifies outliers in a series using the IQR method.
    
    Parameters:
    -----------
    data : array-like
        Data to analyze
    column_name : str, optional
        Name of the column/variable
    k : float, default=1.5
        Multiplier for IQR (typically 1.5 for outliers, 3 for extreme outliers)
        
    Returns:
    --------
    dict
        Dictionary with outlier information
    """
    # Convert to Series if not already
    if not isinstance(data, pd.Series):
        data = pd.Series(data)
    
    # Use column name if provided, otherwise use Series name
    col_name = column_name if column_name is not None else (data.name if data.name else "Value")
    
    # Compute IQR bounds
    q1, q3, lower_bound, upper_bound = compute_iqr_bounds(data, k)
    
    if q1 is None:
        return {
            'column': col_name,
            'q1': None,
            'q3': None,
            'iqr': None,
            'lower_bound': None,
            'upper_bound': None,
            'outliers': [],
            'outlier_indices': [],
            'outlier_count': 0,
            'outlier_percentage': 0
        }
    
    # Identify outliers
    outlier_mask = (data < lower_bound) | (data > upper_bound)
    outlier_indices = data.index[outlier_mask].tolist()
    outliers = data[outlier_mask].tolist()
    
    # Create result dictionary
    result = {
        'column': col_name,
        'q1': q1,
        'q3': q3,
        'iqr': q3 - q1,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'outliers': outliers,
        'outlier_indices': outlier_indices,
        'outlier_count': len(outliers),
        'outlier_percentage': (len(outliers) / len(data.dropna())) * 100 if len(data.dropna()) > 0 else 0
    }
    
    return result


def flag_iqr_outliers_in_dataframe(df, columns=None, k=1.5, output_column='is_iqr_outlier'):
    """
    Flags outliers in a DataFrame based on IQR analysis.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The DataFrame to analyze
    columns : list of str, optional
        Specific columns to analyze (default: all numerical columns)
    k : float, default=1.5
        Multiplier for IQR
    output_column : str, default='is_iqr_outlier'
        Column name for outlier flag
        
    Returns:
    --------
    pandas DataFrame
        DataFrame with outlier flags
    """
    # Create a copy of the DataFrame
    df_flagged = df.copy()
    
    # Initialize outlier flag column
    if output_column not in df_flagged.columns:
        df_flagged[output_column] = False
    
    # Select numerical columns if not specified
    if columns is None:
        columns = df.select_dtypes(include=np.number).columns.tolist()
    else:
        # Filter to only include numerical columns that exist
        columns = [col for col in columns if col in df.columns and 
                  np.issubdtype(df[col].dtype, np.number)]
    
    # Flag outliers for each column
    for col in columns:
        # Compute IQR bounds
        q1, q3, lower_bound, upper_bound = compute_iqr_bounds(df[col], k)
        
        if q1 is not None:
            # Identify outliers
            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            
            # Update outlier flag column
            df_flagged.loc[outlier_mask, output_column] = True
    
    return df_flagged


def create_iqr_boxplot(data, column_name=None, k=1.5, output_dir=None, 
                     filename=None, show_plot=True):
    """
    Creates a boxplot highlighting IQR outliers.
    
    Parameters:
    -----------
    data : array-like
        Data to plot
    column_name : str, optional
        Name of the column/variable
    k : float, default=1.5
        Multiplier for IQR
    output_dir : str, optional
        Directory to save the plot
    filename : str, optional
        Filename for the saved plot (default: 'iqr_boxplot_{column_name}.png')
    show_plot : bool, default=True
        Whether to display the plot
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    # Convert to Series if not already
    if not isinstance(data, pd.Series):
        data = pd.Series(data)
    
    # Use column name if provided, otherwise use Series name
    col_name = column_name if column_name is not None else (data.name if data.name else "Value")
    
    # Generate filename if not provided
    if filename is None:
        safe_col_name = col_name.replace('/', '_').replace('\\', '_')
        filename = f'iqr_boxplot_{safe_col_name}.png'
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create boxplot
    sns.boxplot(x=data, ax=ax)
    
    # Compute IQR bounds
    q1, q3, lower_bound, upper_bound = compute_iqr_bounds(data, k)
    
    if q1 is not None:
        # Identify outliers
        outlier_mask = (data < lower_bound) | (data > upper_bound)
        outliers = data[outlier_mask]
        
        # Add scatter points for outliers
        if not outliers.empty:
            ax.scatter(outliers, [0] * len(outliers), color='red', s=50, zorder=10)
        
        # Add vertical lines for bounds
        ax.axvline(x=lower_bound, color='red', linestyle='--', alpha=0.7, 
                  label=f'Lower Bound ({lower_bound:.2f})')
        ax.axvline(x=upper_bound, color='red', linestyle='--', alpha=0.7,
                  label=f'Upper Bound ({upper_bound:.2f})')
        
        # Add legend
        ax.legend()
    
    # Set title and labels
    ax.set_title(f'Boxplot with IQR Outliers for {col_name} (k={k})')
    ax.set_xlabel(col_name)
    
    # Save plot if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, filename)
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    # Show or close plot
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def create_iqr_histogram(data, column_name=None, k=1.5, bins=30,
                        output_dir=None, filename=None, show_plot=True):
    """
    Creates a histogram highlighting IQR outliers.
    
    Parameters:
    -----------
    data : array-like
        Data to plot
    column_name : str, optional
        Name of the column/variable
    k : float, default=1.5
        Multiplier for IQR
    bins : int, default=30
        Number of bins for histogram
    output_dir : str, optional
        Directory to save the plot
    filename : str, optional
        Filename for the saved plot (default: 'iqr_histogram_{column_name}.png')
    show_plot : bool, default=True
        Whether to display the plot
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    # Convert to Series if not already
    if not isinstance(data, pd.Series):
        data = pd.Series(data)
    
    # Use column name if provided, otherwise use Series name
    col_name = column_name if column_name is not None else (data.name if data.name else "Value")
    
    # Generate filename if not provided
    if filename is None:
        safe_col_name = col_name.replace('/', '_').replace('\\', '_')
        filename = f'iqr_histogram_{safe_col_name}.png'
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Compute IQR bounds
    q1, q3, lower_bound, upper_bound = compute_iqr_bounds(data, k)
    
    if q1 is not None:
        # Identify outliers and non-outliers
        outlier_mask = (data < lower_bound) | (data > upper_bound)
        outliers = data[outlier_mask]
        non_outliers = data[~outlier_mask]
        
        # Plot histogram for non-outliers
        sns.histplot(non_outliers, bins=bins, ax=ax, color='blue', alpha=0.7, 
                    label='Normal Values')
        
        # Plot histogram for outliers
        if not outliers.empty:
            sns.histplot(outliers, bins=bins, ax=ax, color='red', alpha=0.7, 
                        label='Outliers')
        
        # Add vertical lines for bounds
        ax.axvline(x=lower_bound, color='red', linestyle='--', alpha=0.7, 
                  label=f'Lower Bound ({lower_bound:.2f})')
        ax.axvline(x=upper_bound, color='red', linestyle='--', alpha=0.7,
                  label=f'Upper Bound ({upper_bound:.2f})')
        
        # Add legend
        ax.legend()
    else:
        # If IQR bounds couldn't be computed, just plot regular histogram
        sns.histplot(data, bins=bins, ax=ax)
    
    # Set title and labels
    ax.set_title(f'Histogram with IQR Outliers for {col_name} (k={k})')
    ax.set_xlabel(col_name)
    ax.set_ylabel('Frequency')
    
    # Save plot if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, filename)
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    # Show or close plot
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def create_multi_column_iqr_boxplot(df, columns=None, k=1.5,
                                  output_dir=None, filename='multi_column_iqr_boxplot.png',
                                  show_plot=True):
    """
    Creates boxplots for multiple columns highlighting IQR outliers.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The DataFrame to analyze
    columns : list of str, optional
        Specific columns to plot (default: all numerical columns)
    k : float, default=1.5
        Multiplier for IQR
    output_dir : str, optional
        Directory to save the plot
    filename : str, default='multi_column_iqr_boxplot.png'
        Filename for the saved plot
    show_plot : bool, default=True
        Whether to display the plot
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    # Select numerical columns if not specified
    if columns is None:
        columns = df.select_dtypes(include=np.number).columns.tolist()
    else:
        # Filter to only include numerical columns that exist
        columns = [col for col in columns if col in df.columns and 
                  np.issubdtype(df[col].dtype, np.number)]
    
    if not columns:
        return None
    
    # Create figure
    fig, axes = plt.subplots(len(columns), 1, figsize=(12, 4 * len(columns)))
    
    # Handle case with single column
    if len(columns) == 1:
        axes = [axes]
    
    # Create boxplots for each column
    for i, col in enumerate(columns):
        ax = axes[i]
        
        # Create boxplot
        sns.boxplot(x=df[col].dropna(), ax=ax)
        
        # Compute IQR bounds
        q1, q3, lower_bound, upper_bound = compute_iqr_bounds(df[col], k)
        
        if q1 is not None:
            # Identify outliers
            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            outliers = df[col][outlier_mask]
            
            # Add scatter points for outliers
            if not outliers.empty:
                ax.scatter(outliers, [0] * len(outliers), color='red', s=50, zorder=10)
            
            # Add vertical lines for bounds
            ax.axvline(x=lower_bound, color='red', linestyle='--', alpha=0.7, 
                      label=f'Lower Bound ({lower_bound:.2f})')
            ax.axvline(x=upper_bound, color='red', linestyle='--', alpha=0.7,
                      label=f'Upper Bound ({upper_bound:.2f})')
            
            # Add legend
            ax.legend()
        
        # Set title and labels
        ax.set_title(f'Boxplot with IQR Outliers for {col} (k={k})')
        ax.set_xlabel(col)
    
    plt.tight_layout()
    
        # Save plot if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, filename)
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    # Show or close plot
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def detect_outliers_iqr(df, columns=None, k=1.5, output_dir=None, 
                       save_csv=True, plot_boxplots=True, plot_histograms=True,
                       show_plots=True):
    """
    Comprehensive IQR-based outlier detection for numerical columns.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The DataFrame to analyze
    columns : list of str, optional
        Specific columns to analyze (default: all numerical columns)
    k : float, default=1.5
        Multiplier for IQR
    output_dir : str, optional
        Directory to save output files and plots
    save_csv : bool, default=True
        Whether to save outliers to CSV file
    plot_boxplots : bool, default=True
        Whether to create boxplots
    plot_histograms : bool, default=True
        Whether to create histograms
    show_plots : bool, default=True
        Whether to display plots
        
    Returns:
    --------
    dict
        Dictionary containing outlier analysis results
    """
    # Set up logging
    logger = logging.getLogger('iqr_outlier_detection')
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
            file_handler = logging.FileHandler(os.path.join(output_dir, 'iqr_outlier_detection.log'))
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    logger.info("\n=== IQR-Based Outlier Detection ===")
    
    # Select numerical columns if not specified
    if columns is None:
        columns = df.select_dtypes(include=np.number).columns.tolist()
    else:
        # Filter to only include numerical columns that exist
        columns = [col for col in columns if col in df.columns and 
                  np.issubdtype(df[col].dtype, np.number)]
    
    if not columns:
        logger.info("No numerical columns for IQR analysis.")
        return {}
    
    logger.info(f"Analyzing {len(columns)} columns for outliers using IQR method (k={k})")
    
    # Initialize results dictionary
    results = {
        'outliers_by_column': {},
        'flagged_dataframe': None,
        'plots': {}
    }
    
    # Analyze each column
    all_outliers = []
    
    for col in columns:
        logger.info(f"\nAnalyzing column: {col}")
        
        # Identify outliers
        outlier_info = identify_iqr_outliers(df[col], column_name=col, k=k)
        results['outliers_by_column'][col] = outlier_info
        
        # Log results
        if outlier_info['outlier_count'] > 0:
            logger.info(f"  Found {outlier_info['outlier_count']} outliers ({outlier_info['outlier_percentage']:.2f}% of data)")
            logger.info(f"  IQR: {outlier_info['iqr']:.4f}")
            logger.info(f"  Bounds: [{outlier_info['lower_bound']:.4f}, {outlier_info['upper_bound']:.4f}]")
            
            # Add to all outliers list
            for idx in outlier_info['outlier_indices']:
                all_outliers.append({
                    'Row Index': idx,
                    'Column': col,
                    'Value': df.loc[idx, col],
                    'Q1': outlier_info['q1'],
                    'Q3': outlier_info['q3'],
                    'IQR': outlier_info['iqr'],
                    'Lower Bound': outlier_info['lower_bound'],
                    'Upper Bound': outlier_info['upper_bound'],
                    'Type': 'Low' if df.loc[idx, col] < outlier_info['lower_bound'] else 'High'
                })
        else:
            logger.info("  No outliers detected.")
    
    # Create DataFrame with all outliers
    if all_outliers:
        outliers_df = pd.DataFrame(all_outliers)
        results['all_outliers_df'] = outliers_df
        
        # Log summary
        logger.info(f"\nTotal outliers detected: {len(outliers_df)}")
        
        # Count outliers by column
        outlier_counts = outliers_df['Column'].value_counts()
        logger.info("\nOutliers by column:")
        for col, count in outlier_counts.items():
            logger.info(f"  {col}: {count} outliers")
        
        # Save to CSV if requested
        if save_csv and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            csv_path = os.path.join(output_dir, "iqr_outliers.csv")
            outliers_df.to_csv(csv_path, index=False)
            logger.info(f"\nOutliers saved to: {csv_path}")
    else:
        logger.info("\nNo outliers detected in any column.")
        results['all_outliers_df'] = pd.DataFrame()
    
    # Flag outliers in DataFrame
    flagged_df = flag_iqr_outliers_in_dataframe(df, columns=columns, k=k)
    results['flagged_dataframe'] = flagged_df
    
    # Count total flagged rows
    flagged_count = flagged_df['is_iqr_outlier'].sum()
    logger.info(f"\nTotal rows flagged as containing outliers: {flagged_count} ({flagged_count/len(df)*100:.2f}% of data)")
    
    # Save flagged DataFrame to CSV if requested
    if save_csv and output_dir:
        flagged_csv_path = os.path.join(output_dir, "flagged_data_iqr.csv")
        flagged_df.to_csv(flagged_csv_path, index=True)
        logger.info(f"Flagged data saved to: {flagged_csv_path}")
    
    # Create plots directory if needed
    plots_dir = os.path.join(output_dir, 'iqr_plots') if output_dir else None
    
    # Create boxplots if requested
    if plot_boxplots:
        logger.info("\nCreating IQR boxplots...")
        
        # Create individual boxplots
        boxplot_paths = {}
        for col in columns:
            boxplot_fig = create_iqr_boxplot(
                df[col],
                column_name=col,
                k=k,
                output_dir=plots_dir,
                show_plot=show_plots
            )
            boxplot_paths[col] = boxplot_fig
        
        results['plots']['boxplots'] = boxplot_paths
        
        # Create multi-column boxplot
        multi_boxplot_fig = create_multi_column_iqr_boxplot(
            df,
            columns=columns,
            k=k,
            output_dir=plots_dir,
            show_plot=show_plots
        )
        results['plots']['multi_boxplot'] = multi_boxplot_fig
    
    # Create histograms if requested
    if plot_histograms:
        logger.info("\nCreating IQR histograms...")
        
        histogram_paths = {}
        for col in columns:
            hist_fig = create_iqr_histogram(
                df[col],
                column_name=col,
                k=k,
                output_dir=plots_dir,
                show_plot=show_plots
            )
            histogram_paths[col] = hist_fig
        
        results['plots']['histograms'] = histogram_paths
    
    logger.info("\n=== IQR-Based Outlier Detection Complete ===")
    
    return results

