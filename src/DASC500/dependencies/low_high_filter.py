import os
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec


def apply_low_pass_filter(data, threshold, inclusive=True):
    """
    Applies a low-pass filter to data, keeping values below the threshold.
    
    Parameters:
    -----------
    data : array-like
        Data to filter
    threshold : float
        Maximum value to keep
    inclusive : bool, default=True
        Whether to include values equal to the threshold
        
    Returns:
    --------
    pandas Series
        Filtered data
    """
    # Convert to Series if not already
    if not isinstance(data, pd.Series):
        data = pd.Series(data)
    
    # Apply filter
    if inclusive:
        filtered_data = data[data <= threshold]
    else:
        filtered_data = data[data < threshold]
    
    return filtered_data


def apply_high_pass_filter(data, threshold, inclusive=True):
    """
    Applies a high-pass filter to data, keeping values above the threshold.
    
    Parameters:
    -----------
    data : array-like
        Data to filter
    threshold : float
        Minimum value to keep
    inclusive : bool, default=True
        Whether to include values equal to the threshold
        
    Returns:
    --------
    pandas Series
        Filtered data
    """
    # Convert to Series if not already
    if not isinstance(data, pd.Series):
        data = pd.Series(data)
    
    # Apply filter
    if inclusive:
        filtered_data = data[data >= threshold]
    else:
        filtered_data = data[data > threshold]
    
    return filtered_data


def apply_band_pass_filter(data, low_threshold, high_threshold, inclusive=True):
    """
    Applies a band-pass filter to data, keeping values between thresholds.
    
    Parameters:
    -----------
    data : array-like
        Data to filter
    low_threshold : float
        Minimum value to keep
    high_threshold : float
        Maximum value to keep
    inclusive : bool, default=True
        Whether to include values equal to the thresholds
        
    Returns:
    --------
    pandas Series
        Filtered data
    """
    # Convert to Series if not already
    if not isinstance(data, pd.Series):
        data = pd.Series(data)
    
    # Apply filter
    if inclusive:
        filtered_data = data[(data >= low_threshold) & (data <= high_threshold)]
    else:
        filtered_data = data[(data > low_threshold) & (data < high_threshold)]
    
    return filtered_data


def apply_band_stop_filter(data, low_threshold, high_threshold, inclusive=True):
    """
    Applies a band-stop filter to data, removing values between thresholds.
    
    Parameters:
    -----------
    data : array-like
        Data to filter
    low_threshold : float
        Lower threshold of values to remove
    high_threshold : float
        Upper threshold of values to remove
    inclusive : bool, default=True
        Whether to exclude values equal to the thresholds
        
    Returns:
    --------
    pandas Series
        Filtered data
    """
    # Convert to Series if not already
    if not isinstance(data, pd.Series):
        data = pd.Series(data)
    
    # Apply filter
    if inclusive:
        filtered_data = data[(data <= low_threshold) | (data >= high_threshold)]
    else:
        filtered_data = data[(data < low_threshold) | (data > high_threshold)]
    
    return filtered_data


def filter_dataframe_by_column(df, column, filter_type, thresholds, inclusive=True):
    """
    Filters a DataFrame based on values in a specific column.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame to filter
    column : str
        Column name to apply filter to
    filter_type : str
        Type of filter ('low_pass', 'high_pass', 'band_pass', 'band_stop')
    thresholds : float or tuple
        Threshold value(s) for filtering
        - For low_pass/high_pass: single value
        - For band_pass/band_stop: tuple of (low, high) values
    inclusive : bool, default=True
        Whether to include values equal to the thresholds
        
    Returns:
    --------
    pandas DataFrame
        Filtered DataFrame
    """
    # Check if column exists
    if column not in df.columns:
        return df.copy()
    
    # Create a copy of the DataFrame
    filtered_df = df.copy()
    
    # Apply appropriate filter
    if filter_type == 'low_pass':
        if not isinstance(thresholds, (int, float)):
            raise ValueError("For low_pass filter, threshold must be a single value")
        
        if inclusive:
            mask = filtered_df[column] <= thresholds
        else:
            mask = filtered_df[column] < thresholds
    
    elif filter_type == 'high_pass':
        if not isinstance(thresholds, (int, float)):
            raise ValueError("For high_pass filter, threshold must be a single value")
        
        if inclusive:
            mask = filtered_df[column] >= thresholds
        else:
            mask = filtered_df[column] > thresholds
    
    elif filter_type == 'band_pass':
        if not isinstance(thresholds, tuple) or len(thresholds) != 2:
            raise ValueError("For band_pass filter, thresholds must be a tuple of (low, high)")
        
        low_threshold, high_threshold = thresholds
        
        if inclusive:
            mask = (filtered_df[column] >= low_threshold) & (filtered_df[column] <= high_threshold)
        else:
            mask = (filtered_df[column] > low_threshold) & (filtered_df[column] < high_threshold)
    
    elif filter_type == 'band_stop':
        if not isinstance(thresholds, tuple) or len(thresholds) != 2:
            raise ValueError("For band_stop filter, thresholds must be a tuple of (low, high)")
        
        low_threshold, high_threshold = thresholds
        
        if inclusive:
            mask = (filtered_df[column] <= low_threshold) | (filtered_df[column] >= high_threshold)
        else:
            mask = (filtered_df[column] < low_threshold) | (filtered_df[column] > high_threshold)
    
    else:
        raise ValueError("Invalid filter_type. Must be one of: 'low_pass', 'high_pass', 'band_pass', 'band_stop'")
    
    # Apply mask to filter DataFrame
    filtered_df = filtered_df[mask]
    
    return filtered_df


def create_filter_comparison_plot(data, column_name=None, filter_type='low_pass', 
                                thresholds=None, inclusive=True,
                                output_dir=None, filename=None, show_plot=True):
    """
    Creates a plot comparing original and filtered data.
    
    Parameters:
    -----------
    data : array-like
        Data to filter and plot
    column_name : str, optional
        Name of the column/variable
    filter_type : str, default='low_pass'
        Type of filter ('low_pass', 'high_pass', 'band_pass', 'band_stop')
    thresholds : float or tuple, optional
        Threshold value(s) for filtering
        - For low_pass/high_pass: single value
        - For band_pass/band_stop: tuple of (low, high) values
    inclusive : bool, default=True
        Whether to include values equal to the thresholds
    output_dir : str, optional
        Directory to save the plot
    filename : str, optional
        Filename for the saved plot
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
    
    # Set default thresholds if not provided
    if thresholds is None:
        if filter_type in ['low_pass', 'high_pass']:
            thresholds = data.median()
        else:  # band_pass or band_stop
            q1, q3 = data.quantile(0.25), data.quantile(0.75)
            thresholds = (q1, q3)
    
    # Generate filename if not provided
    if filename is None:
        safe_col_name = col_name.replace('/', '_').replace('\\', '_')
        filename = f'{filter_type}_filter_{safe_col_name}.png'
    
    # Apply filter
    if filter_type == 'low_pass':
        filtered_data = apply_low_pass_filter(data, thresholds, inclusive)
        removed_data = data[~data.index.isin(filtered_data.index)]
        threshold_label = f"Threshold: {thresholds:.4f}"
    
    elif filter_type == 'high_pass':
        filtered_data = apply_high_pass_filter(data, thresholds, inclusive)
        removed_data = data[~data.index.isin(filtered_data.index)]
        threshold_label = f"Threshold: {thresholds:.4f}"
    
    elif filter_type == 'band_pass':
        low_threshold, high_threshold = thresholds
        filtered_data = apply_band_pass_filter(data, low_threshold, high_threshold, inclusive)
        removed_data = data[~data.index.isin(filtered_data.index)]
        threshold_label = f"Thresholds: [{low_threshold:.4f}, {high_threshold:.4f}]"
    
    elif filter_type == 'band_stop':
        low_threshold, high_threshold = thresholds
        filtered_data = apply_band_stop_filter(data, low_threshold, high_threshold, inclusive)
        removed_data = data[~data.index.isin(filtered_data.index)]
        threshold_label = f"Thresholds: [{low_threshold:.4f}, {high_threshold:.4f}]"
    
    else:
        raise ValueError("Invalid filter_type. Must be one of: 'low_pass', 'high_pass', 'band_pass', 'band_stop'")
    
    # Create figure with 2 subplots (histogram and boxplot)
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(1, 2, width_ratios=[2, 1])
    
    # Histogram subplot
    ax1 = fig.add_subplot(gs[0])
    
    # Plot histograms
    sns.histplot(data, bins=30, alpha=0.5, label='Original Data', ax=ax1)
    sns.histplot(filtered_data, bins=30, alpha=0.5, label='Filtered Data', ax=ax1)
    
    # Add threshold lines
    if filter_type in ['low_pass', 'high_pass']:
        ax1.axvline(x=thresholds, color='red', linestyle='--', 
                   label=f"Threshold: {thresholds:.4f}")
    else:  # band_pass or band_stop
        low_threshold, high_threshold = thresholds
        ax1.axvline(x=low_threshold, color='red', linestyle='--', 
                   label=f"Low Threshold: {low_threshold:.4f}")
        ax1.axvline(x=high_threshold, color='red', linestyle='--', 
                   label=f"High Threshold: {high_threshold:.4f}")
    
    # Set labels and title for histogram
    ax1.set_xlabel(col_name)
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'Histogram: {filter_type.replace("_", " ").title()} Filter')
    ax1.legend()
    
    # Boxplot subplot
    ax2 = fig.add_subplot(gs[1])
    
    # Create boxplot data
    boxplot_data = [
        {'data': data, 'label': 'Original'},
        {'data': filtered_data, 'label': 'Filtered'},
        {'data': removed_data, 'label': 'Removed'}
    ]
    
    # Plot boxplots
    box_positions = [1, 2, 3]
    box_colors = ['blue', 'green', 'red']
    
    for i, item in enumerate(boxplot_data):
        ax2.boxplot(item['data'].dropna(), positions=[box_positions[i]], 
                   patch_artist=True, 
                   boxprops=dict(facecolor=box_colors[i], alpha=0.5),
                   medianprops=dict(color='black'))
    
    # Set labels and title for boxplot
    ax2.set_ylabel(col_name)
    ax2.set_xticks(box_positions)
    ax2.set_xticklabels([item['label'] for item in boxplot_data])
    ax2.set_title('Boxplot Comparison')
    
    # Add filter info
    plt.figtext(0.5, 0.01, 
               f"{filter_type.replace('_', ' ').title()} Filter | {threshold_label} | "
               f"Kept: {len(filtered_data)}/{len(data)} ({len(filtered_data)/len(data)*100:.1f}%)", 
               ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
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


def create_multi_threshold_comparison(data, column_name=None, filter_type='low_pass',
                                    thresholds_list=None, inclusive=True,
                                    output_dir=None, filename=None, show_plot=True):
    """
    Creates a plot comparing multiple threshold values for a filter.
    
    Parameters:
    -----------
    data : array-like
        Data to filter and plot
    column_name : str, optional
        Name of the column/variable
    filter_type : str, default='low_pass'
        Type of filter ('low_pass', 'high_pass')
    thresholds_list : list, optional
        List of threshold values to compare
    inclusive : bool, default=True
        Whether to include values equal to the thresholds
    output_dir : str, optional
        Directory to save the plot
    filename : str, optional
        Filename for the saved plot
    show_plot : bool, default=True
        Whether to display the plot
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    # Only support low_pass and high_pass for this function
    if filter_type not in ['low_pass', 'high_pass']:
        raise ValueError("This function only supports 'low_pass' and 'high_pass' filter types")
    
    # Convert to Series if not already
    if not isinstance(data, pd.Series):
        data = pd.Series(data)
    
    # Use column name if provided, otherwise use Series name
    col_name = column_name if column_name is not None else (data.name if data.name else "Value")
    
        # Set default thresholds if not provided
    if thresholds_list is None:
        # Create 5 thresholds based on percentiles
        thresholds_list = [
            data.quantile(0.1),
            data.quantile(0.25),
            data.quantile(0.5),
            data.quantile(0.75),
            data.quantile(0.9)
        ]
    
    # Generate filename if not provided
    if filename is None:
        safe_col_name = col_name.replace('/', '_').replace('\\', '_')
        filename = f'multi_threshold_{filter_type}_{safe_col_name}.png'
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    # Plot original data histogram
    sns.histplot(data, bins=30, alpha=0.3, color='gray', label='Original Data', ax=ax1)
    
    # Plot threshold lines and calculate stats
    threshold_stats = []
    colors = plt.cm.viridis(np.linspace(0, 1, len(thresholds_list)))
    
    for i, threshold in enumerate(thresholds_list):
        # Apply filter
        if filter_type == 'low_pass':
            filtered_data = apply_low_pass_filter(data, threshold, inclusive)
        else:  # high_pass
            filtered_data = apply_high_pass_filter(data, threshold, inclusive)
        
        # Calculate percentage kept
        pct_kept = len(filtered_data) / len(data) * 100
        threshold_stats.append({
            'threshold': threshold,
            'kept_count': len(filtered_data),
            'kept_pct': pct_kept
        })
        
        # Add threshold line to histogram
        ax1.axvline(x=threshold, color=colors[i], linestyle='--', 
                   label=f"{threshold:.4f} ({pct_kept:.1f}%)")
    
    # Set labels and title for histogram
    ax1.set_xlabel(col_name)
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'Histogram with {filter_type.replace("_", " ").title()} Filter Thresholds')
    ax1.legend(title='Threshold (% kept)')
    
    # Create percentage plot
    thresholds = [stat['threshold'] for stat in threshold_stats]
    pct_kept = [stat['kept_pct'] for stat in threshold_stats]
    
    ax2.plot(thresholds, pct_kept, 'o-', linewidth=2)
    
    # Add threshold markers
    for i, (threshold, pct) in enumerate(zip(thresholds, pct_kept)):
        ax2.scatter([threshold], [pct], color=colors[i], s=100, zorder=5)
        ax2.annotate(f"{pct:.1f}%", 
                    xy=(threshold, pct),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=9)
    
    # Set labels and title for percentage plot
    ax2.set_xlabel('Threshold Value')
    ax2.set_ylabel('Percentage of Data Kept')
    ax2.set_title(f'Data Retention vs Threshold ({filter_type.replace("_", " ").title()})')
    ax2.grid(True, alpha=0.3)
    
    # Set y-axis limits for percentage plot
    ax2.set_ylim(0, 105)
    
    # Add filter info
    plt.figtext(0.5, 0.01, 
               f"{filter_type.replace('_', ' ').title()} Filter | "
               f"Comparing {len(thresholds_list)} threshold values", 
               ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
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


def apply_data_filters(df, columns=None, filter_configs=None, output_dir=None,
                      create_plots=True, show_plots=True):
    """
    Comprehensive data filtering with multiple filter configurations.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The DataFrame to filter
    columns : list of str, optional
        Specific columns to filter (default: all numerical columns)
    filter_configs : list of dict, optional
        List of filter configurations, each with keys:
        - 'column': Column to filter
        - 'filter_type': Type of filter ('low_pass', 'high_pass', 'band_pass', 'band_stop')
        - 'thresholds': Threshold value(s)
        - 'inclusive': Whether to include threshold values (default: True)
    output_dir : str, optional
        Directory to save output files and plots
    create_plots : bool, default=True
        Whether to create comparison plots
    show_plots : bool, default=True
        Whether to display plots
        
    Returns:
    --------
    dict
        Dictionary containing filtering results
    """
    # Set up logging
    logger = logging.getLogger('data_filtering')
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
            file_handler = logging.FileHandler(os.path.join(output_dir, 'data_filtering.log'))
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    logger.info("\n=== Data Filtering Analysis ===")
    
    # Select numerical columns if not specified
    if columns is None:
        columns = df.select_dtypes(include=np.number).columns.tolist()
    else:
        # Filter to only include columns that exist
        columns = [col for col in columns if col in df.columns]
    
    if not columns:
        logger.info("No valid columns for filtering.")
        return {}
    
    logger.info(f"Available columns for filtering: {columns}")
    
    # Initialize results dictionary
    results = {
        'filtered_dataframes': {},
        'filter_stats': {},
        'plots': {}
    }
    
    # If no filter configs provided, create default ones
    if not filter_configs:
        logger.info("\nNo filter configurations provided. Creating default filters...")
        filter_configs = []
        
        for col in columns:
            # Get column data without NaNs
            col_data = df[col].dropna()
            
            if len(col_data) < 2:
                continue
            
            # Add low-pass filter at 90th percentile
            filter_configs.append({
                'column': col,
                'filter_type': 'low_pass',
                'thresholds': col_data.quantile(0.9),
                'inclusive': True
            })
            
            # Add high-pass filter at 10th percentile
            filter_configs.append({
                'column': col,
                'filter_type': 'high_pass',
                'thresholds': col_data.quantile(0.1),
                'inclusive': True
            })
    
    # Apply each filter configuration
    for i, config in enumerate(filter_configs):
        # Extract configuration parameters
        column = config.get('column')
        filter_type = config.get('filter_type', 'low_pass')
        thresholds = config.get('thresholds')
        inclusive = config.get('inclusive', True)
        
        # Skip if column not in DataFrame
        if column not in df.columns:
            logger.info(f"\nSkipping filter {i+1}: Column '{column}' not found in DataFrame")
            continue
        
        # Skip if thresholds not provided
        if thresholds is None:
            logger.info(f"\nSkipping filter {i+1}: No thresholds provided for column '{column}'")
            continue
        
        logger.info(f"\nApplying filter {i+1}:")
        logger.info(f"  Column: {column}")
        logger.info(f"  Filter Type: {filter_type}")
        
        if isinstance(thresholds, tuple):
            logger.info(f"  Thresholds: {thresholds[0]} to {thresholds[1]}")
        else:
            logger.info(f"  Threshold: {thresholds}")
        
        logger.info(f"  Inclusive: {inclusive}")
        
        # Apply filter
        filtered_df = filter_dataframe_by_column(df, column, filter_type, thresholds, inclusive)
        
        # Calculate statistics
        original_count = len(df)
        filtered_count = len(filtered_df)
        removed_count = original_count - filtered_count
        kept_pct = (filtered_count / original_count) * 100 if original_count > 0 else 0
        
        # Create filter ID
        filter_id = f"{column}_{filter_type}_{i+1}"
        
        # Store results
        results['filtered_dataframes'][filter_id] = filtered_df
        
        # Store statistics
        results['filter_stats'][filter_id] = {
            'column': column,
            'filter_type': filter_type,
            'thresholds': thresholds,
            'inclusive': inclusive,
            'original_count': original_count,
            'filtered_count': filtered_count,
            'removed_count': removed_count,
            'kept_percentage': kept_pct
        }
        
        # Log results
        logger.info(f"  Results: Kept {filtered_count} of {original_count} rows ({kept_pct:.2f}%)")
        
        # Create comparison plot if requested
        if create_plots:
            plots_dir = os.path.join(output_dir, 'filter_plots') if output_dir else None
            
            plot_fig = create_filter_comparison_plot(
                df[column],
                column_name=column,
                filter_type=filter_type,
                thresholds=thresholds,
                inclusive=inclusive,
                output_dir=plots_dir,
                show_plot=show_plots
            )
            
            results['plots'][filter_id] = plot_fig
    
    # Create multi-threshold comparisons for each column
    if create_plots:
        logger.info("\nCreating multi-threshold comparisons...")
        multi_threshold_plots = {}
        
        for col in columns:
            # Skip columns with insufficient data
            if df[col].dropna().shape[0] < 2:
                continue
            
            # Create low-pass comparison
            low_pass_fig = create_multi_threshold_comparison(
                df[col],
                column_name=col,
                filter_type='low_pass',
                output_dir=plots_dir,
                show_plot=show_plots
            )
            multi_threshold_plots[f"{col}_low_pass_comparison"] = low_pass_fig
            
            # Create high-pass comparison
            high_pass_fig = create_multi_threshold_comparison(
                df[col],
                column_name=col,
                filter_type='high_pass',
                output_dir=plots_dir,
                show_plot=show_plots
            )
            multi_threshold_plots[f"{col}_high_pass_comparison"] = high_pass_fig
        
        results['plots']['multi_threshold'] = multi_threshold_plots
    
    # Save filter statistics to CSV if output directory is provided
    if output_dir and results['filter_stats']:
        # Convert filter stats to DataFrame
        stats_rows = []
        for filter_id, stats in results['filter_stats'].items():
            row = {
                'filter_id': filter_id,
                'column': stats['column'],
                'filter_type': stats['filter_type'],
                'original_count': stats['original_count'],
                'filtered_count': stats['filtered_count'],
                'removed_count': stats['removed_count'],
                'kept_percentage': stats['kept_percentage']
            }
            
            # Add thresholds
            if isinstance(stats['thresholds'], tuple):
                row['low_threshold'] = stats['thresholds'][0]
                row['high_threshold'] = stats['thresholds'][1]
            else:
                row['threshold'] = stats['thresholds']
            
            stats_rows.append(row)
        
        stats_df = pd.DataFrame(stats_rows)
        stats_path = os.path.join(output_dir, "filter_statistics.csv")
        stats_df.to_csv(stats_path, index=False)
        logger.info(f"\nFilter statistics saved to: {stats_path}")
    
    logger.info("\n=== Data Filtering Complete ===")
    
    return results

