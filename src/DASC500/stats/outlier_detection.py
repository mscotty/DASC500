from datetime import datetime
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from copy import deepcopy

# For the Venn diagrams in combine_outlier_results
# Note: This is an optional dependency, the function handles its absence gracefully
try:
    from matplotlib_venn import venn2, venn3
except ImportError:
    pass

from DASC500.stats.z_score import detect_outliers_zscore_workflow
from DASC500.stats.iqr import detect_outliers_iqr
from DASC500.stats.pca import perform_pca_analysis, detect_pca_outliers
from DASC500.stats.calc_generic import calculate_descriptive_statistics
from DASC500.stats.html_report import create_html_report

def combine_outlier_results(df, results, output_dir=None, save_csv=True, create_plots=True, show_plots=False):
    """
    Combines outlier detection results from multiple methods.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The original DataFrame
    results : dict
        Dictionary containing results from different outlier detection methods
    output_dir : str, optional
        Directory to save output files
    save_csv : bool, default=True
        Whether to save combined results to CSV
    create_plots : bool, default=True
        Whether to create plots
    show_plots : bool, default=False
        Whether to display plots
        
    Returns:
    --------
    dict
        Dictionary with combined outlier information
    """
    # Initialize logger
    logger = logging.getLogger('outlier_detection')
    
    # Track which rows are identified as outliers by each method
    method_outliers = {}
    
    # Get Z-score outliers
    if 'zscore_results' in results and 'outliers_df' in results['zscore_results']:
        zscore_outliers = results['zscore_results']['outliers_df']
        if not zscore_outliers.empty:
            method_outliers['zscore'] = zscore_outliers.index.tolist()
        else:
            method_outliers['zscore'] = []
    
    # Get IQR outliers
    if 'iqr_results' in results and 'flagged_dataframe' in results['iqr_results']:
        iqr_df = results['iqr_results']['flagged_dataframe']
        if 'is_iqr_outlier' in iqr_df.columns:
            method_outliers['iqr'] = iqr_df[iqr_df['is_iqr_outlier']].index.tolist()
        else:
            method_outliers['iqr'] = []
    
    # Get PCA outliers
    if 'pca_results' in results and 'outliers' in results['pca_results']:
        pca_outliers = results['pca_results']['outliers']
        method_outliers['pca'] = pca_outliers.get('outlier_indices', [])
    
    # Find outliers identified by multiple methods
    all_outlier_indices = set()
    for indices in method_outliers.values():
        all_outlier_indices.update(indices)
    
    # Count how many methods identified each outlier
    agreement_counts = {}
    for idx in all_outlier_indices:
        count = sum(1 for method_indices in method_outliers.values() if idx in method_indices)
        if count not in agreement_counts:
            agreement_counts[count] = []
        agreement_counts[count].append(idx)
    
    # Create a DataFrame with consensus outliers
    consensus_rows = []
    
    for method_count, indices in sorted(agreement_counts.items(), reverse=True):
        for idx in indices:
            # Determine which methods flagged this row
            methods = [method for method, method_indices in method_outliers.items() 
                      if idx in method_indices]
            
            consensus_rows.append({
                'Index': idx,
                'Method_Count': method_count,
                'Methods': ', '.join(methods)
            })
    
    consensus_df = pd.DataFrame(consensus_rows)
    if not consensus_df.empty:
        consensus_df.set_index('Index', inplace=True)
    
    # Log results
    method_counts = {method: len(indices) for method, indices in method_outliers.items()}
    logger.info("\nOutliers by method:")
    for method, count in method_counts.items():
        logger.info(f"  {method}: {count} outliers")
    
    logger.info("\nMethod agreement:")
    for count, indices in sorted(agreement_counts.items(), reverse=True):
        logger.info(f"  {count} methods: {len(indices)} outliers")
    
    # Save to CSV if requested
    if save_csv and output_dir and not consensus_df.empty:
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, "consensus_outliers.csv")
        consensus_df.to_csv(csv_path)
        logger.info(f"\nConsensus outliers saved to: {csv_path}")
    
    # Create visualization if requested
    if create_plots and output_dir and len(method_outliers) > 1:
        try:
            # Create Venn diagram or bar chart
            if len(method_outliers) <= 3:  # Venn diagram for 2-3 methods
                
                fig, ax = plt.subplots(figsize=(10, 8))
                
                if len(method_outliers) == 2:
                    methods = list(method_outliers.keys())
                    venn2(
                        [set(method_outliers[methods[0]]), set(method_outliers[methods[1]])],
                        set_labels=methods,
                        ax=ax
                    )
                else:  # 3 methods
                    methods = list(method_outliers.keys())
                    venn3(
                        [set(method_outliers[methods[0]]), 
                         set(method_outliers[methods[1]]), 
                         set(method_outliers[methods[2]])],
                        set_labels=methods,
                        ax=ax
                    )
                
                ax.set_title('Outlier Detection Method Comparison')
                
                # Save plot
                plot_path = os.path.join(output_dir, 'method_comparison_venn.png')
                fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                
                if not show_plots:
                    plt.close(fig)
            else:  # Bar chart for more than 3 methods
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot counts by agreement level
                counts = [len(indices) for count, indices in sorted(agreement_counts.items())]
                labels = [f"{count} methods" for count in sorted(agreement_counts.keys())]
                
                bars = ax.bar(labels, counts, color='skyblue')
                
                # Add count labels on top of bars
                for bar, count in zip(bars, counts):
                    ax.text(
                        bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.1,
                        str(count),
                        ha='center'
                    )
                
                ax.set_title('Outliers by Method Agreement')
                ax.set_xlabel('Number of Methods in Agreement')
                ax.set_ylabel('Number of Outliers')
                
                # Save plot
                plot_path = os.path.join(output_dir, 'method_agreement_counts.png')
                fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                
                if not show_plots:
                    plt.close(fig)
        except Exception as e:
            logger.error(f"Error creating method comparison visualization: {e}")
    
    return {
        'agreement_counts': agreement_counts,
        'consensus_outliers': consensus_df,
        'method_counts': method_counts
    }


def run_outlier_detection(df, 
                         columns=None,
                         methods=['zscore', 'iqr', 'pca'],
                         zscore_threshold=3.0,
                         iqr_k=1.5,
                         pca_variance_threshold=0.95,
                         output_dir=None,
                         create_plots=True,
                         show_plots=False,
                         save_csv=True,
                         include_descriptive_stats=True,
                         replot_without_outliers=False,
                         outlier_removal_method='consensus',
                         min_consensus_methods=2):
    """
    Comprehensive outlier detection workflow that analyzes data using multiple methods.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The DataFrame to analyze
    columns : list, optional
        List of column names to analyze (default: all numerical columns)
    methods : list, default=['zscore', 'iqr', 'pca']
        List of outlier detection methods to use
    zscore_threshold : float, default=3.0
        Z-score threshold for outlier detection
    iqr_k : float, default=1.5
        Multiplier for IQR outlier detection
    pca_variance_threshold : float, default=0.95
        Threshold for explained variance in PCA analysis
    output_dir : str, optional
        Directory to save output files and plots
    create_plots : bool, default=True
        Whether to create plots
    show_plots : bool, default=False
        Whether to display plots (False is better for automated runs)
    save_csv : bool, default=True
        Whether to save outliers to CSV files
    include_descriptive_stats : bool, default=True
        Whether to include descriptive statistics analysis
    replot_without_outliers : bool, default=False
        Whether to create additional plots with outliers removed
    outlier_removal_method : str, default='consensus'
        Method for determining which outliers to remove:
        - 'consensus': Remove outliers identified by multiple methods
        - 'union': Remove all outliers identified by any method
        - 'zscore': Remove only Z-score outliers
        - 'iqr': Remove only IQR outliers
        - 'pca': Remove only PCA outliers
    min_consensus_methods : int, default=2
        Minimum number of methods that must agree to remove an outlier (only used with 'consensus')
        
    Returns:
    --------
    dict
        Dictionary containing all outlier detection results
    """
    # Create timestamp for run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory with timestamp if provided
    if output_dir:
        output_dir = os.path.join(output_dir, f"outlier_detection_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging
    logger = logging.getLogger('outlier_detection')
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    if logger.handlers:
        logger.handlers.clear()
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if output_dir is provided
    if output_dir:
        file_handler = logging.FileHandler(os.path.join(output_dir, 'outlier_detection.log'))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Initialize results dictionary
    results = {
        'timestamp': timestamp,
        'dataframe_shape': df.shape,
        'row_count': len(df),
        'column_count': len(df.columns),
        'methods': methods,
        'zscore_threshold': zscore_threshold,
        'iqr_k': iqr_k,
        'pca_variance_threshold': pca_variance_threshold,
        'replot_without_outliers': replot_without_outliers,
        'outlier_removal_method': outlier_removal_method,
        'min_consensus_methods': min_consensus_methods,
        'zscore_results': {},
        'iqr_results': {},
        'pca_results': {},
        'combined_results': {},
        'cleaned_data_results': {},  # New section for cleaned data analysis
        'plots': {}
    }
    
    logger.info("\n" + "="*80)
    logger.info(f"OUTLIER DETECTION REPORT - {timestamp}")
    logger.info("="*80)
    logger.info(f"DataFrame shape: {df.shape} (rows, columns)")
    logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")
    if replot_without_outliers:
        logger.info(f"Replotting enabled with method: {outlier_removal_method}")
        if outlier_removal_method == 'consensus':
            logger.info(f"Minimum consensus methods: {min_consensus_methods}")
    logger.info("="*80)
    
    # Determine columns to analyze
    if columns is None:
        columns = df.select_dtypes(include=np.number).columns.tolist()
    else:
        # Filter to only include numerical columns that exist in the DataFrame
        columns = [col for col in columns if col in df.columns and 
                np.issubdtype(df[col].dtype, np.number)]
    
    # Store analyzed columns
    results['columns_analyzed'] = columns
    
    if not columns:
        logger.info("No valid numerical columns found for outlier detection.")
        return results
    
    logger.info(f"Analyzing {len(columns)} numerical columns for outliers")
    logger.info(f"Methods: {', '.join(methods)}")
    logger.info("="*80)
    
    # After determining columns to analyze, add descriptive statistics if requested
    if include_descriptive_stats:
        logger.info("\n\n" + "="*50)
        logger.info("DESCRIPTIVE STATISTICS ANALYSIS")
        logger.info("="*50)
        
        # Create stats directory
        stats_dir = os.path.join(output_dir, 'descriptive_stats') if output_dir else None
        
        # Run descriptive statistics analysis
        stats_results = calculate_descriptive_statistics(
            df,
            columns=columns,
            output_dir=stats_dir,
            create_plots=create_plots,
            plot_types=['boxplot', 'histogram', 'violin'],
            test_for_normality=True,
            show_plots=show_plots
        )
        
        # Add stats results to the main results dictionary
        results['descriptive_stats'] = stats_results
    
    # Run Z-score outlier detection if requested
    if 'zscore' in methods:
        logger.info("\n\n" + "="*50)
        logger.info("Z-SCORE OUTLIER DETECTION")
        logger.info("="*50)
        
        zscore_results = detect_outliers_zscore_workflow(
            df,
            columns=columns,
            threshold=zscore_threshold,
            output_dir=os.path.join(output_dir, 'zscore') if output_dir else None,
            save_csv=save_csv,
            plot_distributions=create_plots,
            show_plots=show_plots
        )
        
        results['zscore_results'] = zscore_results
    
    # Run IQR outlier detection if requested
    if 'iqr' in methods:
        logger.info("\n\n" + "="*50)
        logger.info("IQR OUTLIER DETECTION")
        logger.info("="*50)
        
        iqr_results = detect_outliers_iqr(
            df,
            columns=columns,
            k=iqr_k,
            output_dir=os.path.join(output_dir, 'iqr') if output_dir else None,
            save_csv=save_csv,
            plot_boxplots=create_plots,
            plot_histograms=create_plots,
            show_plots=show_plots
        )
        
        results['iqr_results'] = iqr_results
    
    # Run PCA-based outlier detection if requested
    if 'pca' in methods:
        logger.info("\n\n" + "="*50)
        logger.info("PCA-BASED OUTLIER ANALYSIS")
        logger.info("="*50)
        
        pca_df, pca_model, loadings_df = perform_pca_analysis(
            df,
            columns=columns,
            variance_threshold=pca_variance_threshold,
            output_dir=os.path.join(output_dir, 'pca') if output_dir else None,
            plot_scree=create_plots,
            plot_scatter=create_plots,
            plot_loadings=create_plots,
            show_plots=show_plots
        )
        
        # Add PCA results to the main results dictionary
        if pca_df is not None:
            results['pca_results'] = {
                'pca_df': pca_df,
                'explained_variance': pca_model.explained_variance_ratio_.tolist() if pca_model else None,
                'loadings': loadings_df.to_dict() if loadings_df is not None else None
            }
            
            # Add PCA-based outlier detection using Mahalanobis distance
            if output_dir and create_plots:
                pca_outliers = detect_pca_outliers(
                    df, 
                    columns, 
                    output_dir=os.path.join(output_dir, 'pca'),
                    show_plots=show_plots
                )
                results['pca_results']['outliers'] = pca_outliers
    
    # Combine results from different methods (needed for replotting)
    if len(methods) > 1 or replot_without_outliers:
        logger.info("\n\n" + "="*50)
        logger.info("COMBINED OUTLIER ANALYSIS")
        logger.info("="*50)
        
        combined_results = combine_outlier_results(
            df,
            results,
            output_dir=output_dir if output_dir else None,
            save_csv=save_csv,
            create_plots=create_plots,
            show_plots=show_plots
        )
        
        results['combined_results'] = combined_results
    
    # NEW FUNCTIONALITY: Replot without outliers
    if replot_without_outliers and create_plots:
        logger.info("\n\n" + "="*50)
        logger.info("ANALYSIS WITHOUT OUTLIERS")
        logger.info("="*50)
        
        # Determine which outliers to remove
        outlier_indices = _get_outliers_to_remove(results, outlier_removal_method, min_consensus_methods, logger)
        
        if outlier_indices:
            # Create cleaned dataset
            cleaned_df = df.drop(index=outlier_indices).reset_index(drop=True)
            
            logger.info(f"Removed {len(outlier_indices)} outlier rows using '{outlier_removal_method}' method")
            logger.info(f"Original data shape: {df.shape}")
            logger.info(f"Cleaned data shape: {cleaned_df.shape}")
            logger.info(f"Reduction: {len(outlier_indices)} rows ({len(outlier_indices)/len(df)*100:.2f}%)")
            
            # Create subdirectory for cleaned data analysis
            cleaned_output_dir = os.path.join(output_dir, 'cleaned_data') if output_dir else None
            if cleaned_output_dir:
                os.makedirs(cleaned_output_dir, exist_ok=True)
            
            # Store information about the cleaning process
            results['cleaned_data_results']['removal_info'] = {
                'method': outlier_removal_method,
                'outliers_removed': len(outlier_indices),
                'outlier_indices': outlier_indices,
                'original_shape': df.shape,
                'cleaned_shape': cleaned_df.shape,
                'reduction_percentage': len(outlier_indices)/len(df)*100
            }
            
            # Run descriptive statistics on cleaned data
            if include_descriptive_stats:
                logger.info("\nRunning descriptive statistics on cleaned data...")
                cleaned_stats_dir = os.path.join(cleaned_output_dir, 'descriptive_stats') if cleaned_output_dir else None
                
                cleaned_stats_results = calculate_descriptive_statistics(
                    cleaned_df,
                    columns=columns,
                    output_dir=cleaned_stats_dir,
                    create_plots=True,  # Always create plots for comparison
                    plot_types=['boxplot', 'histogram', 'violin'],
                    test_for_normality=True,
                    show_plots=show_plots
                )
                
                results['cleaned_data_results']['descriptive_stats'] = cleaned_stats_results
            
            # Run outlier detection methods on cleaned data (for comparison)
            logger.info("\nRunning outlier detection on cleaned data for comparison...")
            
            # Z-score analysis on cleaned data
            if 'zscore' in methods:
                logger.info("  Z-score analysis on cleaned data...")
                cleaned_zscore_dir = os.path.join(cleaned_output_dir, 'zscore') if cleaned_output_dir else None
                
                cleaned_zscore_results = detect_outliers_zscore_workflow(
                    cleaned_df,
                    columns=columns,
                    threshold=zscore_threshold,
                    output_dir=cleaned_zscore_dir,
                    save_csv=save_csv,
                    plot_distributions=True,
                    show_plots=show_plots
                )
                
                results['cleaned_data_results']['zscore_results'] = cleaned_zscore_results
            
            # IQR analysis on cleaned data
            if 'iqr' in methods:
                logger.info("  IQR analysis on cleaned data...")
                cleaned_iqr_dir = os.path.join(cleaned_output_dir, 'iqr') if cleaned_output_dir else None
                
                cleaned_iqr_results = detect_outliers_iqr(
                    cleaned_df,
                    columns=columns,
                    k=iqr_k,
                    output_dir=cleaned_iqr_dir,
                    save_csv=save_csv,
                    plot_boxplots=True,
                    plot_histograms=True,
                    show_plots=show_plots
                )
                
                results['cleaned_data_results']['iqr_results'] = cleaned_iqr_results
            
            # PCA analysis on cleaned data
            if 'pca' in methods:
                logger.info("  PCA analysis on cleaned data...")
                cleaned_pca_dir = os.path.join(cleaned_output_dir, 'pca') if cleaned_output_dir else None
                
                cleaned_pca_df, cleaned_pca_model, cleaned_loadings_df = perform_pca_analysis(
                    cleaned_df,
                    columns=columns,
                    variance_threshold=pca_variance_threshold,
                    output_dir=cleaned_pca_dir,
                    plot_scree=True,
                    plot_scatter=True,
                    plot_loadings=True,
                    show_plots=show_plots
                )
                
                if cleaned_pca_df is not None:
                    results['cleaned_data_results']['pca_results'] = {
                        'pca_df': cleaned_pca_df,
                        'explained_variance': cleaned_pca_model.explained_variance_ratio_.tolist() if cleaned_pca_model else None,
                        'loadings': cleaned_loadings_df.to_dict() if cleaned_loadings_df is not None else None
                    }
                    
                    # Detect outliers in cleaned data
                    if cleaned_output_dir:
                        cleaned_pca_outliers = detect_pca_outliers(
                            cleaned_df, 
                            columns, 
                            output_dir=cleaned_pca_dir,
                            show_plots=show_plots
                        )
                        results['cleaned_data_results']['pca_results']['outliers'] = cleaned_pca_outliers
            
            # Create comparison plots
            if cleaned_output_dir:
                _create_before_after_comparison_plots(
                    df, cleaned_df, columns, outlier_indices, 
                    cleaned_output_dir, show_plots, logger
                )
        
        else:
            logger.info("No outliers found to remove based on the specified criteria.")
            results['cleaned_data_results']['removal_info'] = {
                'method': outlier_removal_method,
                'outliers_removed': 0,
                'message': 'No outliers met removal criteria'
            }
    
    # Generate summary report
    logger.info("\n\n" + "="*50)
    logger.info("OUTLIER DETECTION SUMMARY")
    logger.info("="*50)
    
    # Z-score summary
    if 'zscore' in methods and 'outlier_count' in results['zscore_results']:
        zscore_outlier_count = results['zscore_results']['outlier_count']
        zscore_percentage = results['zscore_results']['outlier_percentage']
        logger.info(f"\nZ-score outliers (threshold={zscore_threshold}):")
        logger.info(f"  Total: {zscore_outlier_count} rows ({zscore_percentage:.2f}% of data)")
        
        if 'column_outlier_counts' in results['zscore_results']:
            logger.info("  By column:")
            for col, count in results['zscore_results']['column_outlier_counts'].items():
                logger.info(f"    {col}: {count} outliers")
    
    # IQR summary
    if 'iqr' in methods and 'all_outliers_df' in results['iqr_results']:
        iqr_outlier_count = len(results['iqr_results']['all_outliers_df'])
        iqr_percentage = iqr_outlier_count / len(df) * 100 if iqr_outlier_count > 0 else 0
        logger.info(f"\nIQR outliers (k={iqr_k}):")
        logger.info(f"  Total: {iqr_outlier_count} values ({iqr_percentage:.2f}% of data cells)")
        
        if not results['iqr_results']['all_outliers_df'].empty:
            outlier_counts = results['iqr_results']['all_outliers_df']['Column'].value_counts()
            logger.info("  By column:")
            for col, count in outlier_counts.items():
                logger.info(f"    {col}: {count} outliers")
    
    # PCA summary
    if 'pca' in methods and 'pca_results' in results and 'outliers' in results['pca_results']:
        pca_outlier_count = len(results['pca_results']['outliers'].get('outlier_indices', []))
        pca_percentage = pca_outlier_count / len(df) * 100 if pca_outlier_count > 0 else 0
        logger.info(f"\nPCA-based outliers:")
        logger.info(f"  Total: {pca_outlier_count} rows ({pca_percentage:.2f}% of data)")
    
    # Combined summary
    if 'combined_results' in results and 'agreement_counts' in results['combined_results']:
        logger.info("\nMethod agreement summary:")
        agreement_counts = results['combined_results']['agreement_counts']
        
        for count, rows in sorted(agreement_counts.items(), reverse=True):
            logger.info(f"  Identified by {count} methods: {len(rows)} rows")
    
    # Cleaned data summary
    if replot_without_outliers and 'removal_info' in results.get('cleaned_data_results', {}):
        removal_info = results['cleaned_data_results']['removal_info']
        if removal_info['outliers_removed'] > 0:
            logger.info(f"\nCleaned data analysis:")
            logger.info(f"  Removal method: {removal_info['method']}")
            logger.info(f"  Outliers removed: {removal_info['outliers_removed']} rows")
            logger.info(f"  Data reduction: {removal_info['reduction_percentage']:.2f}%")
            
            # Compare outlier counts in original vs cleaned data
            if 'zscore_results' in results['cleaned_data_results']:
                cleaned_zscore_count = results['cleaned_data_results']['zscore_results'].get('outlier_count', 0)
                original_zscore_count = results['zscore_results'].get('outlier_count', 0)
                logger.info(f"  Z-score outliers: {original_zscore_count} -> {cleaned_zscore_count}")
            
            if 'iqr_results' in results['cleaned_data_results']:
                cleaned_iqr_count = len(results['cleaned_data_results']['iqr_results'].get('all_outliers_df', []))
                original_iqr_count = len(results['iqr_results'].get('all_outliers_df', []))
                logger.info(f"  IQR outliers: {original_iqr_count} -> {cleaned_iqr_count}")
    
    # Generate HTML report if output_dir is provided
    if output_dir:
        create_html_report(results, df, output_dir)
        logger.info(f"\nHTML report saved to: {os.path.join(output_dir, 'outlier_detection_report.html')}")
    
    logger.info("\n" + "="*80)
    logger.info("OUTLIER DETECTION COMPLETE")
    logger.info("="*80)
    
    return results


def _get_outliers_to_remove(results, removal_method, min_consensus_methods, logger):
    """
    Helper function to determine which outlier indices to remove based on the specified method.
    
    Parameters:
    -----------
    results : dict
        The results dictionary from outlier detection
    removal_method : str
        Method for determining which outliers to remove
    min_consensus_methods : int
        Minimum number of methods that must agree (for consensus method)
    logger : logging.Logger
        Logger instance
        
    Returns:
    --------
    list
        List of indices to remove
    """
    outlier_indices = []
    
    if removal_method == 'consensus':
        # Use consensus outliers from combined results
        if 'combined_results' in results and 'agreement_counts' in results['combined_results']:
            agreement_counts = results['combined_results']['agreement_counts']
            
            for method_count, indices in agreement_counts.items():
                if method_count >= min_consensus_methods:
                    outlier_indices.extend(indices)
            
            logger.info(f"Consensus method: found {len(outlier_indices)} outliers agreed upon by ≥{min_consensus_methods} methods")
    
    elif removal_method == 'union':
        # Remove all outliers identified by any method
        all_indices = set()
        
        # Z-score outliers
        if 'zscore_results' in results and 'outliers_df' in results['zscore_results']:
            zscore_outliers = results['zscore_results']['outliers_df']
            if not zscore_outliers.empty:
                all_indices.update(zscore_outliers.index.tolist())
        
        # IQR outliers (get unique row indices)
        if 'iqr_results' in results and 'flagged_dataframe' in results['iqr_results']:
            iqr_df = results['iqr_results']['flagged_dataframe']
            if 'is_iqr_outlier' in iqr_df.columns:
                all_indices.update(iqr_df[iqr_df['is_iqr_outlier']].index.tolist())
        
        # PCA outliers
        if 'pca_results' in results and 'outliers' in results['pca_results']:
            pca_outliers = results['pca_results']['outliers']
            all_indices.update(pca_outliers.get('outlier_indices', []))
        
        outlier_indices = list(all_indices)
        logger.info(f"Union method: found {len(outlier_indices)} outliers from all methods combined")
    
    elif removal_method == 'zscore':
        # Remove only Z-score outliers
        if 'zscore_results' in results and 'outliers_df' in results['zscore_results']:
            zscore_outliers = results['zscore_results']['outliers_df']
            if not zscore_outliers.empty:
                outlier_indices = zscore_outliers.index.tolist()
        logger.info(f"Z-score method: found {len(outlier_indices)} outliers")
    
    elif removal_method == 'iqr':
        # Remove only IQR outliers (get unique row indices)
        if 'iqr_results' in results and 'flagged_dataframe' in results['iqr_results']:
            iqr_df = results['iqr_results']['flagged_dataframe']
            if 'is_iqr_outlier' in iqr_df.columns:
                outlier_indices = iqr_df[iqr_df['is_iqr_outlier']].index.tolist()
        logger.info(f"IQR method: found {len(outlier_indices)} outliers")
    
    elif removal_method == 'pca':
        # Remove only PCA outliers
        if 'pca_results' in results and 'outliers' in results['pca_results']:
            pca_outliers = results['pca_results']['outliers']
            outlier_indices = pca_outliers.get('outlier_indices', [])
        logger.info(f"PCA method: found {len(outlier_indices)} outliers")
    
    return sorted(list(set(outlier_indices)))  # Remove duplicates and sort


def _create_before_after_comparison_plots(original_df, cleaned_df, columns, removed_indices, 
                                        output_dir, show_plots, logger):
    """
    Create before/after comparison plots to visualize the effect of outlier removal.
    
    Parameters:
    -----------
    original_df : pandas DataFrame
        Original dataframe with outliers
    cleaned_df : pandas DataFrame
        Cleaned dataframe without outliers
    columns : list
        List of columns to analyze
    removed_indices : list
        Indices of removed outliers
    output_dir : str
        Directory to save plots
    show_plots : bool
        Whether to display plots
    logger : logging.Logger
        Logger instance
    """
    try:
        logger.info("Creating before/after comparison plots...")
        
        # Create comparison directory
        comparison_dir = os.path.join(output_dir, 'before_after_comparison')
        os.makedirs(comparison_dir, exist_ok=True)
        
        # Create side-by-side box plots
        n_cols = min(3, len(columns))  # Max 3 columns per row
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(columns):
            if i < len(axes):
                ax = axes[i]
                
                # Create box plot comparison
                data_to_plot = [original_df[col].dropna(), cleaned_df[col].dropna()]
                box_plot = ax.boxplot(data_to_plot, labels=['Original', 'Cleaned'], patch_artist=True)
                
                # Color the boxes
                box_plot['boxes'][0].set_facecolor('lightcoral')
                box_plot['boxes'][1].set_facecolor('lightblue')
                
                ax.set_title(f'{col}\nBoxplot Comparison')
                ax.set_ylabel('Values')
                ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(len(columns), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(comparison_dir, 'boxplot_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        
        if not show_plots:
            plt.close(fig)
        
        # Create histogram comparisons
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(columns):
            if i < len(axes):
                ax = axes[i]
                
                # Plot histograms
                ax.hist(original_df[col].dropna(), bins=30, alpha=0.7, 
                       label='Original', color='lightcoral', density=True)
                ax.hist(cleaned_df[col].dropna(), bins=30, alpha=0.7, 
                       label='Cleaned', color='lightblue', density=True)
                
                ax.set_title(f'{col}\nHistogram Comparison')
                ax.set_xlabel('Values')
                ax.set_ylabel('Density')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(len(columns), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(comparison_dir, 'histogram_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        
        if not show_plots:
            plt.close(fig)
        
        # Create summary statistics comparison
        summary_comparison = pd.DataFrame({
            'Original_Mean': original_df[columns].mean(),
            'Cleaned_Mean': cleaned_df[columns].mean(),
            'Original_Std': original_df[columns].std(),
            'Cleaned_Std': cleaned_df[columns].std(),
            'Original_Median': original_df[columns].median(),
            'Cleaned_Median': cleaned_df[columns].median()
        })
        
        # Calculate percentage changes
        summary_comparison['Mean_Change_%'] = ((summary_comparison['Cleaned_Mean'] - 
                                              summary_comparison['Original_Mean']) / 
                                             summary_comparison['Original_Mean'] * 100)
        summary_comparison['Std_Change_%'] = ((summary_comparison['Cleaned_Std'] - 
                                             summary_comparison['Original_Std']) / 
                                            summary_comparison['Original_Std'] * 100)
        
        # Save summary comparison
        summary_comparison.to_csv(os.path.join(comparison_dir, 'summary_statistics_comparison.csv'))
        
        # Create a visual summary table
        fig, ax = plt.subplots(figsize=(12, max(6, len(columns) * 0.8)))
        ax.axis('tight')
        ax.axis('off')
        
        # Round values for display
        display_df = summary_comparison.round(3)
        
        table = ax.table(cellText=display_df.values,
                        rowLabels=display_df.index,
                        colLabels=display_df.columns,
                        cellLoc='center',
                        loc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        # Color code the percentage change columns
        for i in range(len(display_df)):
            # Mean change color
            mean_change = display_df.iloc[i]['Mean_Change_%']
            if abs(mean_change) > 5:  # Significant change
                color = 'lightcoral' if mean_change > 0 else 'lightgreen'
                table[(i + 1, 6)].set_facecolor(color)  # Mean_Change_% column
            
            # Std change color
            std_change = display_df.iloc[i]['Std_Change_%']
            if abs(std_change) > 5:  # Significant change
                color = 'lightcoral' if std_change > 0 else 'lightgreen'
                table[(i + 1, 7)].set_facecolor(color)  # Std_Change_% column
        
        plt.title('Summary Statistics Comparison: Original vs Cleaned Data', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.savefig(os.path.join(comparison_dir, 'summary_statistics_table.png'), 
                   dpi=300, bbox_inches='tight')
        
        if not show_plots:
            plt.close(fig)
        
        # Create scatter plots showing removed outliers
        if len(columns) >= 2:
            # Create pairwise scatter plots for first few columns
            max_pairs = min(6, len(columns) * (len(columns) - 1) // 2)  # Limit number of plots
            pair_count = 0
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            for i in range(len(columns)):
                for j in range(i + 1, len(columns)):
                    if pair_count >= max_pairs:
                        break
                    
                    ax = axes[pair_count]
                    
                    # Plot all points
                    ax.scatter(original_df[columns[i]], original_df[columns[j]], 
                             alpha=0.6, color='lightblue', s=20, label='Retained')
                    
                    # Highlight removed outliers
                    if len(removed_indices) > 0:
                        outlier_data = original_df.loc[removed_indices]
                        ax.scatter(outlier_data[columns[i]], outlier_data[columns[j]], 
                                 alpha=0.8, color='red', s=30, label='Removed', marker='x')
                    
                    ax.set_xlabel(columns[i])
                    ax.set_ylabel(columns[j])
                    ax.set_title(f'{columns[i]} vs {columns[j]}')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    pair_count += 1
                
                if pair_count >= max_pairs:
                    break
            
            # Hide empty subplots
            for i in range(pair_count, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(os.path.join(comparison_dir, 'outlier_scatter_plots.png'), 
                       dpi=300, bbox_inches='tight')
            
            if not show_plots:
                plt.close(fig)
        
        logger.info(f"Before/after comparison plots saved to: {comparison_dir}")
        
    except Exception as e:
        logger.error(f"Error creating before/after comparison plots: {e}")


def get_cleaned_dataframe(results, original_df, removal_method='consensus', min_consensus_methods=2):
    """
    Helper function to extract the cleaned dataframe from outlier detection results.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from run_outlier_detection
    original_df : pandas DataFrame
        Original dataframe
    removal_method : str, default='consensus'
        Method for determining which outliers to remove
    min_consensus_methods : int, default=2
        Minimum number of methods that must agree (for consensus method)
        
    Returns:
    --------
    pandas DataFrame
        Cleaned dataframe with outliers removed
    list
        List of removed indices
    """
    # Create a dummy logger for the helper function
    import logging
    logger = logging.getLogger('outlier_detection')
    
    # Get outliers to remove
    outlier_indices = _get_outliers_to_remove(results, removal_method, min_consensus_methods, logger)
    
    if outlier_indices:
        cleaned_df = original_df.drop(index=outlier_indices).reset_index(drop=True)
        return cleaned_df, outlier_indices
    else:
        return original_df.copy(), []


# Example usage function
def example_usage():
    """
    Example of how to use the upgraded outlier detection function.
    """
    # Create sample data with outliers
    np.random.seed(42)
    n_samples = 1000
    
    # Generate normal data
    data = {
        'feature_1': np.random.normal(50, 10, n_samples),
        'feature_2': np.random.normal(100, 15, n_samples),
        'feature_3': np.random.normal(25, 5, n_samples)
    }
    
    # Add some outliers
    outlier_indices = np.random.choice(n_samples, 20, replace=False)
    data['feature_1'][outlier_indices[:7]] += np.random.normal(100, 20, 7)  # High outliers
    data['feature_2'][outlier_indices[7:14]] -= np.random.normal(80, 15, 7)  # Low outliers
    data['feature_3'][outlier_indices[14:]] += np.random.normal(50, 10, 6)  # High outliers
    
    df = pd.DataFrame(data)
    
    print("Running outlier detection with replotting enabled...")
    
    # Run outlier detection with replotting
    results = run_outlier_detection(
        df=df,
        methods=['zscore', 'iqr', 'pca'],
        output_dir='./outlier_analysis_example',
        replot_without_outliers=True,  # Enable replotting
        outlier_removal_method='consensus',  # Use consensus method
        min_consensus_methods=2,  # Require at least 2 methods to agree
        create_plots=True,
        show_plots=False,
        save_csv=True
    )
    
    # Extract cleaned dataframe for further analysis
    cleaned_df, removed_indices = get_cleaned_dataframe(
        results, df, 
        removal_method='consensus', 
        min_consensus_methods=2
    )
    
    print(f"\nOriginal data shape: {df.shape}")
    print(f"Cleaned data shape: {cleaned_df.shape}")
    print(f"Removed {len(removed_indices)} outliers")
    
    return results, df, cleaned_df


if __name__ == "__main__":
    # Run example
    results, original_df, cleaned_df = example_usage()