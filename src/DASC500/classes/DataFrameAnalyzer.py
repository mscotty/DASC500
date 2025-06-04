# Consider organizing imports by standard library, third-party, and local modules
import os
import tempfile
import shutil
import unittest
import itertools
import functools
from typing import Dict, List, Tuple, Union, Optional, Any, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as scipy_stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from DASC500.stats import *  # Consider explicit imports instead of wildcard



class ComprehensiveDataFrameAnalyzer:
    """
    A class for performing various comprehensive analyses on pandas DataFrames.
    It provides methods for descriptive statistics, correlation analysis,
    distribution investigation, IQR and Z-score outlier detection,
    Principal Component Analysis (PCA), and threshold-based filtering.
    
    Examples:
    ---------
    >>> analyzer = ComprehensiveDataFrameAnalyzer()
    >>> stats_results = analyzer.perform_descriptive_analysis(df, columns=['age', 'income'])
    >>> outliers = analyzer.detect_outliers_with_iqr(df, columns=['salary'])
    """

    def __init__(self, show_plots_default: bool = True, output_dir_default: Optional[str] = None):
        """
        Initializes the analyzer.

        Parameters:
        -----------
        show_plots_default : bool, default=True
            Default behavior for showing plots if not specified in method calls.
        output_dir_default : str, optional
            Default directory to save plots and outputs if not specified in method calls.
        """
        self.show_plots_default = show_plots_default
        self.output_dir_default = output_dir_default
        if self.output_dir_default:
            os.makedirs(self.output_dir_default, exist_ok=True)
        
        # Default plot style
        self.set_plot_style()
        
        # For progress reporting
        self.verbose = True

    def set_plot_style(self, style_dict: Optional[Dict[str, Any]] = None) -> None:
        """
        Set default plotting style for all visualizations.
        
        Parameters:
        -----------
        style_dict : dict, optional
            Dictionary of matplotlib rcParams to customize plot appearance.
        """
        # Default style that works well for publications/reports
        default_style = {
            'figure.figsize': (10, 6),
            'axes.grid': True,
            'grid.alpha': 0.3,
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.titlesize': 18
        }
        
        # Update with user preferences if provided
        if style_dict:
            default_style.update(style_dict)
        
        plt.rcParams.update(default_style)
    
    def set_verbose(self, verbose: bool) -> None:
        """
        Set verbosity for progress reporting.
        
        Parameters:
        -----------
        verbose : bool
            Whether to print progress messages.
        """
        self.verbose = verbose
    
    def _log(self, message: str) -> None:
        """
        Print a log message if verbosity is enabled.
        
        Parameters:
        -----------
        message : str
            The message to print.
        """
        if self.verbose:
            print(message)

    def _prepare_output_dir(
        self, method_output_dir: Optional[str] = None, specific_subdir: Optional[str] = None
    ) -> Optional[str]:
        """
        Prepares and returns the appropriate output directory path for saving results or plots.

        This helper method determines the final output directory based on a method-specific
        override, the class's default output directory, and an optional specific subdirectory.
        It ensures the directory exists before returning its path.

        Parameters:
        -----------
        method_output_dir : str, optional
            A specific directory path provided for the current method call,
            which overrides the `output_dir_default` of the class.
        specific_subdir : str, optional
            A subdirectory to be created within the chosen base output directory.
            Useful for organizing plots by method or analysis type.

        Returns:
        --------
        str or None
            The path to the prepared output directory. Returns None if no base
            directory (neither `method_output_dir` nor `self.output_dir_default`)
            is specified.
        """
        base_dir = (
            method_output_dir
            if method_output_dir is not None
            else self.output_dir_default
        )
        if base_dir:
            if specific_subdir:
                final_dir = os.path.join(base_dir, specific_subdir)
            else:
                final_dir = base_dir
            os.makedirs(final_dir, exist_ok=True)
            return final_dir
        return None
    
    def _process_in_chunks(self, 
                          dataframe: pd.DataFrame, 
                          columns: List[str], 
                          chunk_size: int = 10000, 
                          func: Callable = None) -> pd.DataFrame:
        """
        Process large dataframes in chunks to reduce memory usage.
        
        Parameters:
        -----------
        dataframe : pd.DataFrame
            The DataFrame to process.
        columns : list of str
            Columns to process.
        chunk_size : int, default=10000
            Number of rows to process in each chunk.
        func : callable
            Function to apply to each chunk. Should take a DataFrame and return a DataFrame.
            
        Returns:
        --------
        pd.DataFrame
            Combined results from processing all chunks.
        """
        if len(dataframe) <= chunk_size:
            return func(dataframe[columns])
        
        results = []
        for i in range(0, len(dataframe), chunk_size):
            self._log(f"Processing chunk {i//chunk_size + 1}/{(len(dataframe)-1)//chunk_size + 1}")
            chunk = dataframe.iloc[i:i+chunk_size]
            result = func(chunk[columns])
            results.append(result)
        
        return pd.concat(results)
    
    def identify_constant_columns(self, dataframe: pd.DataFrame) -> List[str]:
        """
        Identify columns with zero variance (constant values).
        
        Parameters:
        -----------
        dataframe : pd.DataFrame
            The DataFrame to analyze.
            
        Returns:
        --------
        List[str]
            List of column names that have constant values.
        """
        return [col for col in dataframe.columns 
                if pd.api.types.is_numeric_dtype(dataframe[col]) and dataframe[col].nunique() == 1]
    
    def identify_duplicate_columns(self, dataframe: pd.DataFrame) -> List[Tuple[str, str]]:
        """
        Identify columns that have identical values.
        
        Parameters:
        -----------
        dataframe : pd.DataFrame
            The DataFrame to analyze.
            
        Returns:
        --------
        List[Tuple[str, str]]
            List of tuples containing pairs of column names with identical values.
        """
        duplicate_pairs = []
        columns = dataframe.columns
        
        for i in range(len(columns)):
            for j in range(i+1, len(columns)):
                if dataframe[columns[i]].equals(dataframe[columns[j]]):
                    duplicate_pairs.append((columns[i], columns[j]))
        
        return duplicate_pairs

    def error_handler(method: Callable) -> Callable:
        """
        Decorator to handle common errors in analysis methods.
        
        Parameters:
        -----------
        method : callable
            The method to wrap with error handling.
            
        Returns:
        --------
        callable
            Wrapped method with error handling.
        """
        @functools.wraps(method)
        def wrapper(self, dataframe: pd.DataFrame, *args, **kwargs):
            try:
                # Check if dataframe is valid
                if not isinstance(dataframe, pd.DataFrame):
                    return {"error": "Input must be a pandas DataFrame"}
                if dataframe.empty:
                    return {"error": "DataFrame is empty"}
                    
                return method(self, dataframe, *args, **kwargs)
            except Exception as e:
                import traceback
                return {
                    "error": f"An error occurred: {str(e)}", 
                    "exception": e,
                    "traceback": traceback.format_exc()
                }
        return wrapper

    @error_handler
    def perform_descriptive_analysis(
        self,
        dataframe: pd.DataFrame,
        columns: Optional[List[str]] = None,
        test_normality_alpha: float = 0.05,
        generate_plots: bool = False,
        plot_types: Optional[List[str]] = None,  # ['boxplot', 'histogram', 'violin']
        output_dir: Optional[str] = None,
        show_plots: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Computes comprehensive descriptive statistics for numerical columns,
        optionally tests for normality, and generates distribution plots.

        Parameters:
        -----------
        dataframe : pd.DataFrame
            The DataFrame to analyze.
        columns : list of str, optional
            Specific numerical columns to analyze (default: all numerical columns).
        test_normality_alpha : float, default=0.05
            Significance level for the Shapiro-Wilk normality test. Can be None to skip normality testing.
        generate_plots : bool, default=False
            Whether to generate and save/show plots.
        plot_types : list of str, optional
            Types of plots to generate (e.g., ['boxplot', 'histogram']).
            Defaults to ['boxplot', 'histogram'] if generate_plots is True.
        output_dir : str, optional
            Directory to save plots. Uses class default if None.
        show_plots : bool, optional
            Whether to display plots. Uses class default if None.

        Returns:
        --------
        dict
            A dictionary containing:
            - 'statistics_summary': DataFrame with descriptive statistics.
            - 'normality_tests' (optional): DataFrame with normality test results.
            - 'plot_paths' (optional): Dictionary of paths to saved plots.
            
        Examples:
        ---------
        >>> analyzer = ComprehensiveDataFrameAnalyzer()
        >>> results = analyzer.perform_descriptive_analysis(df, columns=['age', 'income'])
        >>> results['statistics_summary']
        """
        self._log("Starting descriptive analysis...")
        
        if columns is None:
            num_df = dataframe.select_dtypes(include=np.number)
            columns_to_analyze = num_df.columns.tolist()
        else:
            columns_to_analyze = [
                col
                for col in columns
                if col in dataframe.columns
                and pd.api.types.is_numeric_dtype(dataframe[col])
            ]

        if not columns_to_analyze:
            return {"error": "No numerical columns found or specified for analysis."}

        num_df_selected = dataframe[
            columns_to_analyze
        ].copy()  # Use .copy() to avoid SettingWithCopyWarning
        results = {}

        self._log(f"Computing statistics for {len(columns_to_analyze)} columns...")
        
        # Compute advanced statistics
        desc_stats = num_df_selected.describe().T
        desc_stats["skewness"] = num_df_selected.skew()
        desc_stats["kurtosis"] = num_df_selected.kurtosis()
        desc_stats["median"] = num_df_selected.median()
        
        # Use a robust MAD calculation
        try:
            desc_stats["mad"] = num_df_selected.apply(
                lambda x: scipy_stats.median_abs_deviation(x.dropna(), scale="normal"),
                axis=0,
            )
        except (AttributeError):  # Fallback for older scipy
            desc_stats["mad"] = (
                num_df_selected - num_df_selected.median()
            ).abs().median() * 1.4826

        desc_stats["sum"] = num_df_selected.sum()
        desc_stats["variance"] = num_df_selected.var()
        
        # Calculate quantiles more efficiently
        quantiles = num_df_selected.quantile([0.01, 0.05, 0.10, 0.25, 0.75, 0.90, 0.95, 0.99])
        for q_val in [0.01, 0.05, 0.10, 0.90, 0.95, 0.99]:
            desc_stats[f"{q_val*100:.0f}%"] = quantiles.loc[q_val]
            
        desc_stats["iqr"] = quantiles.loc[0.75] - quantiles.loc[0.25]
        desc_stats["range"] = desc_stats["max"] - desc_stats["min"]
        
        # Handle potential division by zero for cv if mean is zero
        desc_stats["cv"] = np.where(
            desc_stats["mean"] == 0, np.nan, desc_stats["std"] / desc_stats["mean"]
        )
        
        # Missing value statistics
        missing_counts = num_df_selected.isnull().sum()
        desc_stats["missing_count"] = missing_counts
        desc_stats["missing_percent"] = (missing_counts / len(num_df_selected)) * 100
        
        results["statistics_summary"] = desc_stats

        # Test for normality
        if test_normality_alpha is not None:
            self._log("Testing for normality...")
            normality_results = pd.DataFrame(
                index=columns_to_analyze,
                columns=["shapiro_statistic", "shapiro_p_value", "is_normal"],
            )
            for col in columns_to_analyze:
                data_clean = num_df_selected[col].dropna()
                if len(data_clean) >= 3:  # Shapiro-Wilk needs at least 3 samples
                    stat, p_val = scipy_stats.shapiro(data_clean)
                    normality_results.loc[col] = [
                        stat,
                        p_val,
                        p_val > test_normality_alpha,
                    ]
                else:
                    normality_results.loc[col] = [np.nan, np.nan, np.nan]
            results["normality_tests"] = normality_results

        # Generate plots
        if generate_plots:
            self._log("Generating plots...")
            current_show_plots = (
                show_plots if show_plots is not None else self.show_plots_default
            )
            current_output_dir = self._prepare_output_dir(
                output_dir, "descriptive_plots"
            )
            plot_paths = {}
            if plot_types is None:
                plot_types = ["boxplot", "histogram"]

            for col in columns_to_analyze:
                self._log(f"Creating plots for {col}...")
                col_plot_paths = {}
                data_to_plot = num_df_selected[col].dropna()  # Use dropna for plotting
                if data_to_plot.empty:  # Skip plotting if no data after dropna
                    continue

                if "boxplot" in plot_types:
                    fig, ax = plt.subplots()
                    sns.boxplot(y=data_to_plot, ax=ax, orientation="vertical")
                    ax.set_title(f"Box Plot of {col}")
                    if current_output_dir:
                        path = os.path.join(current_output_dir, f"{col}_boxplot.png")
                        fig.savefig(path, bbox_inches='tight', dpi=300)
                        col_plot_paths["boxplot"] = path
                    if current_show_plots:
                        plt.show()
                    plt.close(fig)  # Close figure after showing or saving

                if "histogram" in plot_types:
                    fig, ax = plt.subplots()
                    sns.histplot(data_to_plot, kde=True, ax=ax)
                    ax.set_title(f"Histogram of {col}")
                    if current_output_dir:
                        path = os.path.join(current_output_dir, f"{col}_histogram.png")
                        fig.savefig(path, bbox_inches='tight', dpi=300)
                        col_plot_paths["histogram"] = path
                    if current_show_plots:
                        plt.show()
                    plt.close(fig)

                if "violin" in plot_types:
                    fig, ax = plt.subplots()
                    sns.violinplot(y=data_to_plot, ax=ax)
                    ax.set_title(f"Violin Plot of {col}")
                    if current_output_dir:
                        path = os.path.join(current_output_dir, f"{col}_violinplot.png")
                        fig.savefig(path, bbox_inches='tight', dpi=300)
                        col_plot_paths["violin"] = path
                    if current_show_plots:
                        plt.show()
                    plt.close(fig)
                    
                # Add a new combined plot option
                if "combined" in plot_types:
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                    sns.boxplot(y=data_to_plot, ax=ax1)
                    ax1.set_title(f"Box Plot of {col}")
                    sns.histplot(data_to_plot, kde=True, ax=ax2)
                    ax2.set_title(f"Distribution of {col}")
                    plt.tight_layout()
                    if current_output_dir:
                        path = os.path.join(current_output_dir, f"{col}_combined.png")
                        fig.savefig(path, bbox_inches='tight', dpi=300)
                        col_plot_paths["combined"] = path
                    if current_show_plots:
                        plt.show()
                    plt.close(fig)
                    
                if col_plot_paths:
                    plot_paths[col] = col_plot_paths
            if plot_paths:
                results["plot_paths"] = plot_paths
                
        self._log("Descriptive analysis completed.")
        return results

    @error_handler
    def analyze_column_correlations(
        self,
        dataframe: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = "pearson",
        correlation_threshold: float = 0.7,
        generate_heatmap: bool = False,
        output_dir: Optional[str] = None,
        show_plots: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Computes correlation matrix, identifies highly correlated pairs,
        and optionally generates a heatmap.

        Parameters:
        -----------
        dataframe : pd.DataFrame
            The DataFrame to analyze.
        columns : list of str, optional
            Specific numerical columns for correlation (default: all numerical columns).
        method : str, default='pearson'
            Method of correlation ('pearson', 'kendall', 'spearman').
        correlation_threshold : float, default=0.7
            Absolute correlation value above which pairs are considered highly correlated.
        generate_heatmap : bool, default=False
            Whether to generate and save/show a correlation heatmap.
        output_dir : str, optional
            Directory to save the heatmap. Uses class default if None.
        show_plots : bool, optional
            Whether to display the heatmap. Uses class default if None.

        Returns:
        --------
        dict
            A dictionary containing:
            - 'correlation_matrix': DataFrame of correlations.
            - 'highly_correlated_pairs': DataFrame of variable pairs exceeding the threshold.
            - 'heatmap_path' (optional): Path to the saved heatmap image.
            
        Examples:
        ---------
        >>> analyzer = ComprehensiveDataFrameAnalyzer()
        >>> results = analyzer.analyze_column_correlations(
        ...     df, method='spearman', correlation_threshold=0.8
        ... )
        >>> highly_correlated = results['highly_correlated_pairs']
        """
        self._log(f"Starting correlation analysis using {method} method...")
        
        if columns is None:
            num_df = dataframe.select_dtypes(include=np.number)
        else:
            num_df = dataframe[
                [
                    col
                    for col in columns
                    if col in dataframe.columns
                    and pd.api.types.is_numeric_dtype(dataframe[col])
                ]
            ]

        if num_df.shape[1] < 2:
            return {
                "error": "Correlation analysis requires at least two numerical columns."
            }

        self._log(f"Computing correlation matrix for {num_df.shape[1]} columns...")
        corr_matrix = num_df.corr(method=method)
        results = {"correlation_matrix": corr_matrix}

        # Identify highly correlated pairs more efficiently
        # Create a mask for the upper triangle
        mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        
        # Apply the mask and find pairs exceeding threshold
        corr_values = corr_matrix.where(mask)
        
        # Find coordinates of high correlations
        highly_correlated = []
        high_corr_indices = np.where(np.abs(corr_values) >= correlation_threshold)
        
        for i, j in zip(*high_corr_indices):
            row_name = corr_matrix.index[i]
            col_name = corr_matrix.columns[j]
            correlation = corr_values.iloc[i, j]
            highly_correlated.append({
                "Variable1": row_name,
                "Variable2": col_name,
                "Correlation": correlation,
                "Abs_Correlation": abs(correlation)
            })
        
        # Sort by absolute correlation (strongest first)
        highly_correlated_df = pd.DataFrame(highly_correlated)
        if not highly_correlated_df.empty:
            highly_correlated_df = highly_correlated_df.sort_values(
                by="Abs_Correlation", ascending=False
            )
            # Remove the helper column after sorting
            highly_correlated_df = highly_correlated_df.drop(columns=["Abs_Correlation"])
            
        results["highly_correlated_pairs"] = highly_correlated_df

        if generate_heatmap:
            self._log("Generating correlation heatmap...")
            current_show_plots = (
                show_plots if show_plots is not None else self.show_plots_default
            )
            current_output_dir = self._prepare_output_dir(
                output_dir, "correlation_plots"
            )

            # Create a more visually appealing heatmap
            plt.figure(figsize=(
                max(8, corr_matrix.shape[1] * 0.6),
                max(6, corr_matrix.shape[0] * 0.6),
            ))
            
            # Use a mask to hide the upper triangle for a cleaner look
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            
            # Create the heatmap with improved aesthetics
            heatmap = sns.heatmap(
                corr_matrix,
                annot=True,
                cmap="coolwarm",
                fmt=".2f",
                linewidths=0.5,
                mask=mask,
                vmin=-1,
                vmax=1,
                center=0,
                square=True,
                cbar_kws={"shrink": .8}
            )
            
            plt.title(f"{method.capitalize()} Correlation Matrix", fontsize=16, pad=20)
            plt.tight_layout()
            
            if current_output_dir:
                path = os.path.join(
                    current_output_dir, f"{method}_correlation_heatmap.png"
                )
                plt.savefig(path, bbox_inches='tight', dpi=300)
                results["heatmap_path"] = path
                
                # Save a second version with highlighted high correlations
                plt.figure(figsize=(
                    max(8, corr_matrix.shape[1] * 0.6),
                    max(6, corr_matrix.shape[0] * 0.6),
                ))
                
                # Create a custom mask to highlight high correlations
                highlight_mask = np.abs(corr_matrix) < correlation_threshold
                
                sns.heatmap(
                    corr_matrix,
                    annot=True,
                    cmap="coolwarm",
                    fmt=".2f",
                    linewidths=0.5,
                    mask=highlight_mask,
                    vmin=-1,
                    vmax=1,
                    center=0,
                    square=True,
                    cbar_kws={"shrink": .8}
                )
                
                plt.title(f"High Correlations (|r| ≥ {correlation_threshold})", fontsize=16, pad=20)
                plt.tight_layout()
                
                highlighted_path = os.path.join(
                    current_output_dir, f"{method}_high_correlations.png"
                )
                plt.savefig(highlighted_path, bbox_inches='tight', dpi=300)
                results["highlighted_heatmap_path"] = highlighted_path
                
            if current_show_plots:
                plt.show()
            plt.close('all')
            
        self._log("Correlation analysis completed.")
        return results

    @error_handler
    def investigate_value_distribution(
        self,
        dataframe: pd.DataFrame,
        column_name: str,
        distributions_to_test: Optional[List[str]] = None,  # ['norm', 'expon', 'uniform', 'lognorm']
        alpha: float = 0.05,
        generate_plots: bool = False,
        output_dir: Optional[str] = None,
        show_plots: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Analyzes the distribution of a single numerical column, tests against specified
        distributions, calculates moments, and can generate relevant plots.

        Parameters:
        -----------
        dataframe : pd.DataFrame
            The DataFrame containing the column.
        column_name : str
            The name of the numerical column to analyze.
        distributions_to_test : list of str, optional
            List of distributions to test against (e.g., 'norm', 'expon').
            Defaults to ['norm', 'expon', 'uniform', 'lognorm'].
        alpha : float, default=0.05
            Significance level for distribution fit tests.
        generate_plots : bool, default=False
            Whether to generate Q-Q plots and distribution comparison plots.
        output_dir : str, optional
            Directory to save plots. Uses class default if None.
        show_plots : bool, optional
            Whether to display plots. Uses class default if None.

        Returns:
        --------
        dict
            A dictionary containing:
            - 'column_name': The analyzed column.
            - 'moments': Dictionary of (mean, variance, skewness, kurtosis).
            - 'distribution_fit_tests': Dictionary of test results for each distribution.
            - 'best_fit_distribution': Name of the best fitting distribution based on p-value (if applicable).
            - 'plot_paths' (optional): Dictionary of paths to saved plots.
            
        Examples:
        ---------
        >>> analyzer = ComprehensiveDataFrameAnalyzer()
        >>> results = analyzer.investigate_value_distribution(
        ...     df, 'income', distributions_to_test=['norm', 'lognorm']
        ... )
        >>> print(f"Best fit: {results['best_fit_distribution']}")
        """
        self._log(f"Starting distribution analysis for column '{column_name}'...")
        
        if distributions_to_test is None:
            distributions_to_test = ["norm", "expon", "uniform", "lognorm"]

        if column_name not in dataframe.columns or not pd.api.types.is_numeric_dtype(
            dataframe[column_name]
        ):
            return {"error": f"Column '{column_name}' not found or is not numerical."}

        data = dataframe[column_name].dropna()
        if len(data) < 5:  # Needs enough data for meaningful analysis
            return {
                "error": f"Column '{column_name}' has insufficient data points ({len(data)}) for distribution analysis."
            }

        results = {"column_name": column_name}

        # Calculate moments
        self._log("Calculating distribution moments...")
        results["moments"] = {
            "mean": np.mean(data),
            "variance": np.var(data),
            "skewness": scipy_stats.skew(data),
            "kurtosis": scipy_stats.kurtosis(data),  # Fisher's definition (normal=0)
        }

        # Test distribution fits
        self._log(f"Testing fit against {len(distributions_to_test)} distributions...")
        fit_tests = {}
        best_fit_p_value = -1
        best_fit_dist = None

        for dist_name in distributions_to_test:
            try:
                dist = getattr(scipy_stats, dist_name)
                # For uniform, K-S test needs loc and scale from data, not fitted params
                if dist_name == "uniform":
                    params = (
                        data.min(),
                        data.max() - data.min(),
                    )  # loc, scale for uniform
                else:
                    params = dist.fit(data)

                D, p_value = scipy_stats.kstest(data, dist_name, args=params)
                fit_tests[dist_name] = {
                    "statistic_D": D,
                    "p_value": p_value,
                    "fits": p_value > alpha,
                    "parameters": params,
                }
                if p_value > best_fit_p_value:
                    best_fit_p_value = p_value
                    best_fit_dist = dist_name
            except Exception as e:
                fit_tests[dist_name] = {
                    "error": str(e),
                    "p_value": -1,
                    "fits": False,
                }  # Ensure p_value exists for sorting
        results["distribution_fit_tests"] = fit_tests
        
        # Re-evaluate best_fit_dist based on actual p-values from tests
        valid_fits = {
            k: v
            for k, v in fit_tests.items()
            if "p_value" in v and v["p_value"] is not None
        }
        if valid_fits:
            best_fit_dist = max(valid_fits, key=lambda k: valid_fits[k]["p_value"])
            if (
                valid_fits[best_fit_dist]["p_value"] < 0
            ):  # if all fits failed or had errors
                best_fit_dist = None
        else:
            best_fit_dist = None

        results["best_fit_distribution"] = best_fit_dist

        if generate_plots:
            self._log("Generating distribution plots...")
            current_show_plots = (
                show_plots if show_plots is not None else self.show_plots_default
            )
            current_output_dir = self._prepare_output_dir(
                output_dir,
                f"distribution_plots/{column_name.replace('/', '_').replace(' ', '_')}",
            )  # Sanitize column name for path
            plot_paths = {}

            # Q-Q plot (against normal or best fit)
            target_dist_for_qq = (
                best_fit_dist
                if best_fit_dist and best_fit_dist != "uniform"
                else "norm"
            )

            fig, ax = plt.subplots(figsize=(8, 6))
            if (
                target_dist_for_qq == "uniform"
            ):  # Special handling for uniform if needed, or default to norm
                scipy_stats.probplot(
                    data, dist="norm", plot=ax
                )  # Fallback to normal for uniform for simplicity here
                ax.set_title(f"Q-Q Plot for {column_name} vs Normal (Uniform best fit)")
            else:
                scipy_stats.probplot(data, dist=target_dist_for_qq, plot=ax)
                ax.set_title(
                    f"Q-Q Plot for {column_name} vs {target_dist_for_qq.capitalize()}"
                )

            if current_output_dir:
                path = os.path.join(
                    current_output_dir,
                    f'{column_name.replace("/", "_").replace(" ", "_")}_qq_plot_{target_dist_for_qq}.png',
                )
                fig.savefig(path, bbox_inches='tight', dpi=300)
                plot_paths["qq_plot"] = path
            if current_show_plots:
                plt.show()
            plt.close(fig)

            # Enhanced histogram with fitted distributions
            fig, ax = plt.subplots(figsize=(10, 6))
            # Use more bins for large datasets
            n_bins = min(50, max(10, int(np.sqrt(len(data)))))
            
            # Plot histogram with KDE
            sns.histplot(
                data,
                kde=True,
                stat="density",
                label="Data Histogram",
                ax=ax,
                bins=n_bins,
                color='skyblue',
                edgecolor='black',
                alpha=0.7
            )
            
            # Set up x range for distribution curves
            x_min, x_max = ax.get_xlim()
            x_plot = np.linspace(x_min, x_max, 1000)
            
            # Use a colormap for different distributions
            colors = plt.cm.tab10(np.linspace(0, 1, len(distributions_to_test)))
            
            for idx, dist_name_plot in enumerate(distributions_to_test):
                if "error" not in fit_tests.get(dist_name_plot, {}):  # Check if fit was successful
                    try:
                        dist_plot = getattr(scipy_stats, dist_name_plot)
                        params_plot = fit_tests[dist_name_plot].get("parameters")
                        
                        if params_plot is not None:
                            pdf = dist_plot.pdf(x_plot, *params_plot)
                            
                            # Bold line for best fit
                            if dist_name_plot == best_fit_dist:
                                ax.plot(
                                    x_plot, 
                                    pdf, 
                                    label=f"{dist_name_plot.capitalize()} fit (Best)",
                                    color=colors[idx],
                                    linewidth=3,
                                    linestyle='-'
                                )
                            else:
                                ax.plot(
                                    x_plot, 
                                    pdf, 
                                    label=f"{dist_name_plot.capitalize()} fit",
                                    color=colors[idx],
                                    linewidth=1.5,
                                    linestyle='--'
                                )
                    except Exception as e:
                        self._log(f"Error plotting {dist_name_plot} distribution: {str(e)}")
            
            ax.set_title(f"Distribution Fits for {column_name}", fontsize=14)
            ax.set_xlabel(column_name, fontsize=12)
            ax.set_ylabel("Density", fontsize=12)
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            
            if current_output_dir:
                path = os.path.join(
                    current_output_dir,
                    f'{column_name.replace("/", "_").replace(" ", "_")}_distribution_comparison.png',
                )
                fig.savefig(path, bbox_inches='tight', dpi=300)
                plot_paths["distribution_comparison_plot"] = path
            if current_show_plots:
                plt.show()
            plt.close(fig)
            
            # Add an additional plot: Combined distribution analysis
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
            
            # Plot 1: Histogram with KDE
            sns.histplot(data, kde=True, ax=ax1, bins=n_bins, color='skyblue')
            ax1.set_title(f"Distribution of {column_name}")
            
            # Plot 2: Box plot
            sns.boxplot(y=data, ax=ax2, color='lightgreen')
            ax2.set_title(f"Box Plot of {column_name}")
            
            # Plot 3: Q-Q Plot
            scipy_stats.probplot(data, dist=target_dist_for_qq, plot=ax3)
            ax3.set_title(f"Q-Q Plot ({target_dist_for_qq.capitalize()})")
            
            plt.tight_layout()
            
            if current_output_dir:
                path = os.path.join(
                    current_output_dir,
                    f'{column_name.replace("/", "_").replace(" ", "_")}_combined_analysis.png',
                )
                fig.savefig(path, bbox_inches='tight', dpi=300)
                plot_paths["combined_analysis"] = path
            if current_show_plots:
                plt.show()
            plt.close(fig)

            if plot_paths:
                results["plot_paths"] = plot_paths
                
        self._log("Distribution analysis completed.")
        return results

    @error_handler
    def detect_outliers_with_iqr(
        self,
        dataframe: pd.DataFrame,
        columns: Optional[List[str]] = None,
        k_multiplier: float = 1.5,
        generate_plots: bool = False,
        output_dir: Optional[str] = None,
        show_plots: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Identifies outliers in numerical columns using the IQR method.

        Parameters:
        -----------
        dataframe : pd.DataFrame
            The DataFrame to analyze.
        columns : list of str, optional
            Specific numerical columns to check for outliers (default: all numerical columns).
        k_multiplier : float, default=1.5
            Multiplier for the IQR to determine outlier bounds (e.g., 1.5 or 3.0).
        generate_plots : bool, default=False
            Whether to generate boxplots showing outliers.
        output_dir : str, optional
            Directory to save plots. Uses class default if None.
        show_plots : bool, optional
            Whether to display plots. Uses class default if None.

        Returns:
        --------
        dict
            A dictionary where keys are column names. Each value is a dict with:
            - 'outlier_indices': List of indices of outliers.
            - 'outlier_values': List of outlier values.
            - 'bounds': Tuple (lower_bound, upper_bound).
            - 'Q1', 'Q3', 'IQR': Calculated IQR statistics.
            - 'plot_path' (optional): Path to the saved boxplot.
            
        Examples:
        ---------
        >>> analyzer = ComprehensiveDataFrameAnalyzer()
        >>> outliers = analyzer.detect_outliers_with_iqr(df, k_multiplier=2.0)
        >>> for col, info in outliers.items():
        ...     if info['outlier_indices']:
        ...         print(f"{col}: {len(info['outlier_indices'])} outliers detected")
        """
        self._log("Starting IQR outlier detection...")
        
        if columns is None:
            columns_to_analyze = dataframe.select_dtypes(
                include=np.number
            ).columns.tolist()
        else:
            columns_to_analyze = [
                col
                for col in columns
                if col in dataframe.columns
                and pd.api.types.is_numeric_dtype(dataframe[col])
            ]

        all_outlier_info = {}
        if not columns_to_analyze:
            return {"error": "No numerical columns to analyze for IQR outliers."}

        current_show_plots = (
            show_plots if show_plots is not None else self.show_plots_default
        )
        current_output_dir_iqr = self._prepare_output_dir(
            output_dir, "iqr_outlier_plots"
        )

        for col in columns_to_analyze:
            self._log(f"Analyzing column '{col}' for outliers...")
            data_col = dataframe[col].dropna()
            if len(data_col) < 4:  # Need some data for quartiles
                all_outlier_info[col] = {
                    "outlier_indices": [],
                    "outlier_values": [],
                    "bounds": (np.nan, np.nan),
                    "Q1": np.nan,
                    "Q3": np.nan,
                    "IQR": np.nan,
                    "message": "Insufficient data",
                }
                continue

            # Calculate quartiles and IQR
            Q1 = data_col.quantile(0.25)
            Q3 = data_col.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - k_multiplier * IQR
            upper_bound = Q3 + k_multiplier * IQR

            # Identify outliers
            outliers_mask = (data_col < lower_bound) | (data_col > upper_bound)
            outliers = data_col[outliers_mask]
            
            # Calculate percentage of outliers
            outlier_percentage = (len(outliers) / len(data_col)) * 100 if len(data_col) > 0 else 0
            
            all_outlier_info[col] = {
                "outlier_indices": outliers.index.tolist(),
                "outlier_values": outliers.tolist(),
                "bounds": (lower_bound, upper_bound),
                "Q1": Q1,
                "Q3": Q3,
                "IQR": IQR,
                "outlier_count": len(outliers),
                "outlier_percentage": outlier_percentage,
                "total_values": len(data_col)
            }

            if generate_plots:
                # Create enhanced boxplot
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot the boxplot with a specific style
                boxplot = sns.boxplot(
                    x=data_col, 
                    ax=ax, 
                    whis=k_multiplier, 
                    width=0.5,
                    palette="Set3"
                )
                
                # Add scatter points for outliers for better visibility
                if not outliers.empty:
                    sns.stripplot(
                        x=outliers, 
                        ax=ax,
                        color='red',
                        size=8,
                        jitter=0.1
                    )
                
                # Add bounds as vertical lines
                ax.axvline(
                    lower_bound,
                    color='red',
                    linestyle='--',
                    linewidth=2,
                    label=f"Lower bound: {lower_bound:.2f}"
                )
                ax.axvline(
                    upper_bound,
                    color='red',
                    linestyle='--',
                    linewidth=2,
                    label=f"Upper bound: {upper_bound:.2f}"
                )
                
                # Add title and labels
                ax.set_title(
                    f"IQR Outlier Detection for {col}\n"
                    f"(k={k_multiplier}, {len(outliers)} outliers, {outlier_percentage:.1f}%)",
                    fontsize=14
                )
                ax.set_xlabel(f"{col} Values", fontsize=12)
                ax.legend(loc='best')
                
                # Add text annotations for statistics
                stats_text = (
                    f"Q1: {Q1:.2f}\n"
                    f"Median: {data_col.median():.2f}\n"
                    f"Q3: {Q3:.2f}\n"
                    f"IQR: {IQR:.2f}"
                )
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                ax.text(
                    0.05, 0.95, stats_text, transform=ax.transAxes, 
                    fontsize=10, verticalalignment='top', bbox=props
                )
                
                if current_output_dir_iqr:
                    path = os.path.join(
                        current_output_dir_iqr,
                        f'{col.replace("/", "_").replace(" ", "_")}_iqr_boxplot.png',
                    )
                    fig.savefig(path, bbox_inches='tight', dpi=300)
                    all_outlier_info[col]["plot_path"] = path
                if current_show_plots:
                    plt.show()
                plt.close(fig)
                
                # Add a second plot: histogram with bounds
                if len(data_col) > 0:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Plot histogram
                    sns.histplot(
                        data_col, 
                        kde=True, 
                        ax=ax, 
                        bins='auto',
                        color='skyblue',
                        edgecolor='black',
                        alpha=0.7
                    )
                    
                    # Add bounds as vertical lines
                    ax.axvline(
                        lower_bound,
                        color='red',
                        linestyle='--',
                        linewidth=2,
                        label=f"Lower bound: {lower_bound:.2f}"
                    )
                    ax.axvline(
                        upper_bound,
                        color='red',
                        linestyle='--',
                        linewidth=2,
                        label=f"Upper bound: {upper_bound:.2f}"
                    )
                    
                    # Highlight outlier regions
                    x_min, x_max = ax.get_xlim()
                    y_min, y_max = ax.get_ylim()
                    
                    # Lower outlier region
                    if x_min < lower_bound:
                        ax.fill_betweenx(
                            [y_min, y_max], 
                            x_min, 
                            lower_bound, 
                            color='red', 
                            alpha=0.2
                        )
                    
                    # Upper outlier region
                    if x_max > upper_bound:
                        ax.fill_betweenx(
                            [y_min, y_max], 
                            upper_bound, 
                            x_max, 
                            color='red', 
                            alpha=0.2
                        )
                    
                    ax.set_title(
                        f"Distribution of {col} with IQR Bounds\n"
                        f"(k={k_multiplier}, {len(outliers)} outliers, {outlier_percentage:.1f}%)",
                        fontsize=14
                    )
                    ax.set_xlabel(f"{col} Values", fontsize=12)
                    ax.set_ylabel("Frequency", fontsize=12)
                    ax.legend(loc='best')
                    
                    if current_output_dir_iqr:
                        path = os.path.join(
                            current_output_dir_iqr,
                            f'{col.replace("/", "_").replace(" ", "_")}_iqr_histogram.png',
                        )
                        fig.savefig(path, bbox_inches='tight', dpi=300)
                        all_outlier_info[col]["histogram_plot_path"] = path
                    if current_show_plots:
                        plt.show()
                    plt.close(fig)
        
        self._log(f"IQR outlier detection completed. Analyzed {len(columns_to_analyze)} columns.")
        return all_outlier_info

    @error_handler
    def detect_outliers_with_zscore(
        self,
        dataframe: pd.DataFrame,
        columns: Optional[List[str]] = None,
        threshold: float = 3.0,
        generate_plots: bool = False,
        output_dir: Optional[str] = None,
        show_plots: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Identifies outliers in numerical columns using the Z-score method.

        Parameters:
        -----------
        dataframe : pd.DataFrame
            The DataFrame to analyze.
        columns : list of str, optional
            Specific numerical columns for Z-score analysis (default: all numerical columns).
        threshold : float, default=3.0
            Z-score magnitude above which a value is considered an outlier.
        generate_plots : bool, default=False
            Whether to generate Z-score distribution plots.
        output_dir : str, optional
            Directory to save plots. Uses class default if None.
        show_plots : bool, optional
            Whether to display plots. Uses class default if None.

        Returns:
        --------
        dict
            A dictionary containing:
            - 'outliers_summary': DataFrame detailing outliers per column, their original values, and Z-scores.
            - 'plot_paths' (optional): Dictionary of paths to saved Z-score distribution plots.
            
        Examples:
        ---------
        >>> analyzer = ComprehensiveDataFrameAnalyzer()
        >>> results = analyzer.detect_outliers_with_zscore(df, threshold=2.5)
        >>> outliers_df = results['outliers_summary']
        >>> print(f"Found {len(outliers_df)} outliers across all columns")
        """
        self._log(f"Starting Z-score outlier detection with threshold {threshold}...")
        
        if columns is None:
            columns_to_analyze = dataframe.select_dtypes(
                include=np.number
            ).columns.tolist()
        else:
            columns_to_analyze = [
                col
                for col in columns
                if col in dataframe.columns
                and pd.api.types.is_numeric_dtype(dataframe[col])
            ]

        if not columns_to_analyze:
            return {"error": "No numerical columns to analyze for Z-score outliers."}

        all_outliers_list = []
        plot_paths = {}
        column_stats = {}  # To store per-column statistics

        current_show_plots = (
            show_plots if show_plots is not None else self.show_plots_default
        )
        current_output_dir_zscore = self._prepare_output_dir(
            output_dir, "zscore_outlier_plots"
        )

        for col in columns_to_analyze:
            self._log(f"Analyzing column '{col}' for Z-score outliers...")
            data_col = dataframe[col].dropna()
            if len(data_col) < 2:  # Need mean and std, and at least one non-outlier
                continue

            # Calculate Z-scores more efficiently using vectorized operations
            col_mean = data_col.mean()
            col_std = data_col.std()  # ddof=1 by default for pandas Series.std()

            if col_std == 0:  # Avoid division by zero if all values are the same
                z_scores_values = pd.Series(0, index=data_col.index)
            else:
                z_scores_values = (data_col - col_mean) / col_std

            # Identify outliers based on the threshold
            col_outliers_mask = np.abs(z_scores_values) > threshold
            col_outliers = data_col[col_outliers_mask]
            col_outliers_z = z_scores_values[col_outliers_mask]

            # Store column-level statistics
            outlier_percentage = (len(col_outliers) / len(data_col)) * 100 if len(data_col) > 0 else 0
            column_stats[col] = {
                "total_values": len(data_col),
                "outlier_count": len(col_outliers),
                "outlier_percentage": outlier_percentage,
                "mean": col_mean,
                "std": col_std,
                "min_z_score": z_scores_values.min() if not z_scores_values.empty else None,
                "max_z_score": z_scores_values.max() if not z_scores_values.empty else None,
            }

            # Add outliers to the main list
            for index, value in col_outliers.items():
                all_outliers_list.append(
                    {
                        "column": col,
                        "index": index,  # Original index from the input dataframe
                        "value": value,
                        "z_score": col_outliers_z.loc[index],
                    }
                )

            if generate_plots and len(data_col) > 0:
                # Create enhanced Z-score distribution plot
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Plot 1: Histogram of Z-scores
                sns.histplot(
                    z_scores_values, 
                    kde=True, 
                    ax=ax1,
                    bins='auto',
                    color='skyblue',
                    edgecolor='black',
                    alpha=0.7
                )
                
                # Add threshold lines
                ax1.axvline(
                    threshold,
                    color='red',
                    linestyle='--',
                    linewidth=2,
                    label=f"Threshold (±{threshold})"
                )
                ax1.axvline(
                    -threshold,
                    color='red',
                    linestyle='--',
                    linewidth=2
                )
                
                # Highlight outlier regions
                x_min, x_max = ax1.get_xlim()
                y_min, y_max = ax1.get_ylim()
                
                # Lower outlier region
                if x_min < -threshold:
                    ax1.fill_betweenx(
                        [y_min, y_max], 
                        x_min, 
                        -threshold, 
                        color='red', 
                        alpha=0.2
                    )
                
                # Upper outlier region
                if x_max > threshold:
                    ax1.fill_betweenx(
                        [y_min, y_max], 
                        threshold, 
                        x_max, 
                        color='red', 
                        alpha=0.2
                    )
                
                ax1.set_title(f"Z-score Distribution for {col}", fontsize=14)
                ax1.set_xlabel("Z-score", fontsize=12)
                ax1.set_ylabel("Frequency", fontsize=12)
                ax1.legend(loc='best')
                
                # Plot 2: Scatter plot of values vs Z-scores
                ax2.scatter(
                    data_col, 
                    z_scores_values,
                    alpha=0.7,
                    edgecolor='k',
                    s=50
                )
                
                # Highlight outliers
                if not col_outliers.empty:
                    ax2.scatter(
                        col_outliers,
                        col_outliers_z,
                        color='red',
                        s=80,
                        label='Outliers',
                        zorder=5
                    )
                
                # Add threshold lines
                ax2.axhline(
                    threshold,
                    color='red',
                    linestyle='--',
                    linewidth=2,
                    label=f"Threshold (±{threshold})"
                )
                ax2.axhline(
                    -threshold,
                    color='red',
                    linestyle='--',
                    linewidth=2
                )
                
                ax2.set_title(f"Values vs Z-scores for {col}", fontsize=14)
                ax2.set_xlabel(f"{col} Values", fontsize=12)
                ax2.set_ylabel("Z-score", fontsize=12)
                ax2.legend(loc='best')
                
                # Add text annotations for statistics
                stats_text = (
                    f"Total values: {len(data_col)}\n"
                    f"Outliers: {len(col_outliers)} ({outlier_percentage:.1f}%)\n"
                    f"Mean: {col_mean:.2f}\n"
                    f"Std dev: {col_std:.2f}"
                )
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                ax2.text(
                    0.05, 0.95, stats_text, transform=ax2.transAxes, 
                    fontsize=10, verticalalignment='top', bbox=props
                )
                
                plt.tight_layout()
                
                if current_output_dir_zscore:
                    path = os.path.join(
                        current_output_dir_zscore,
                        f'{col.replace("/", "_").replace(" ", "_")}_zscore_analysis.png',
                    )
                    fig.savefig(path, bbox_inches='tight', dpi=300)
                    plot_paths[col] = path
                if current_show_plots:
                    plt.show()
                plt.close(fig)

        # Create outliers summary DataFrame
        outliers_df = pd.DataFrame(all_outliers_list)
        
        # Sort by absolute Z-score (descending)
        if not outliers_df.empty:
            outliers_df['abs_z_score'] = outliers_df['z_score'].abs()
            outliers_df = outliers_df.sort_values('abs_z_score', ascending=False)
            outliers_df = outliers_df.drop(columns=['abs_z_score'])

        results = {
            "outliers_summary": outliers_df,
            "column_stats": column_stats
        }
        
        if plot_paths:
            results["plot_paths"] = plot_paths
            
        self._log(f"Z-score outlier detection completed. Found {len(outliers_df)} outliers across {len(columns_to_analyze)} columns.")
        return results

    @error_handler
    def apply_principal_component_analysis(
        self,
        dataframe: pd.DataFrame,
        columns: Optional[List[str]] = None,
        n_components: Optional[Union[int, float]] = None,
        variance_threshold_for_auto_n: float = 0.95,
        generate_plots: bool = False,
        output_dir: Optional[str] = None,
        show_plots: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Performs PCA, determines optimal components, and returns transformed data and insights.

        Parameters:
        -----------
        dataframe : pd.DataFrame
            The DataFrame to analyze.
        columns : list of str, optional
            Specific numerical columns for PCA (default: all numerical columns).
        n_components : int or float, optional
            Number of components to keep. If int, it's the number.
            If float (0 to 1), it's the variance to be explained.
            If None, uses variance_threshold_for_auto_n.
        variance_threshold_for_auto_n : float, default=0.95
            If n_components is None, this variance threshold is used to select #components.
        generate_plots : bool, default=False
            Whether to generate scree plots, loading plots, and PC scatter plots.
        output_dir : str, optional
            Directory to save plots. Uses class default if None.
        show_plots : bool, optional
            Whether to display plots. Uses class default if None.

        Returns:
        --------
        dict
            A dictionary containing:
            - 'pca_transformed_data': DataFrame of principal components.
            - 'explained_variance_ratio': List of variance explained by each component.
            - 'cumulative_explained_variance': List of cumulative variance.
            - 'loadings': DataFrame of component loadings.
            - 'n_components_selected': The number of components used.
            - 'plot_paths' (optional): Dictionary of paths to saved PCA plots.
            
        Examples:
        ---------
        >>> analyzer = ComprehensiveDataFrameAnalyzer()
        >>> pca_results = analyzer.apply_principal_component_analysis(
        ...     df, columns=['height', 'weight', 'age'], n_components=2
        ... )
        >>> transformed_data = pca_results['pca_transformed_data']
        """
        self._log("Starting Principal Component Analysis...")
        
        if columns is None:
            num_df = dataframe.select_dtypes(include=np.number)
            columns_to_analyze = num_df.columns.tolist()
        else:
            columns_to_analyze = [
                col
                for col in columns
                if col in dataframe.columns
                and pd.api.types.is_numeric_dtype(dataframe[col])
            ]

        if (
            not columns_to_analyze or len(columns_to_analyze) < 1
        ):  # PCA can run on 1 column, but it's trivial
            return {"error": "PCA requires at least one numerical column for analysis."}

        self._log(f"Preparing data for PCA with {len(columns_to_analyze)} columns...")
        data_for_pca = dataframe[columns_to_analyze].dropna()
        if data_for_pca.shape[0] < 2:  # Need at least 2 samples for PCA
            return {
                "error": "Insufficient data rows after dropping NaNs for PCA (need at least 2)."
            }
        if data_for_pca.shape[1] == 0:  # No columns left after selection/dropna
            return {
                "error": "No valid columns remaining for PCA after selection and NaN handling."
            }

        # Standardize data
        self._log("Standardizing data...")
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_for_pca)

        # Determine n_components for PCA
        actual_max_components = min(
            data_for_pca.shape[0], data_for_pca.shape[1]
        )  # Max possible components

        self._log(f"Determining optimal number of components (max: {actual_max_components})...")
        if n_components is None:
            pca_temp = PCA(
                n_components=actual_max_components
            )  # Fit with max possible components first
            pca_temp.fit(scaled_data)
            cumulative_variance = np.cumsum(pca_temp.explained_variance_ratio_)
            n_components_selected_auto = (
                np.argmax(cumulative_variance >= variance_threshold_for_auto_n) + 1
            )
            n_components_selected = n_components_selected_auto
        elif isinstance(n_components, float) and 0 < n_components <= 1.0:
            pca_temp = PCA(
                n_components=n_components
            )  # n_components is variance ratio here
            pca_temp.fit(scaled_data)
            n_components_selected = pca_temp.n_components_
        elif isinstance(n_components, int) and n_components > 0:
            n_components_selected = min(n_components, actual_max_components)
        else:
            return {
                "error": "Invalid n_components value. Must be None, int > 0, or float between 0 and 1."
            }

        if (
            n_components_selected == 0
        ):  # If variance threshold is too low or data is weird
            n_components_selected = 1  # Default to at least 1 component

        self._log(f"Running PCA with {n_components_selected} components...")
        pca = PCA(n_components=n_components_selected)
        pca_transformed_data = pca.fit_transform(scaled_data)
        pca_df = pd.DataFrame(
            data=pca_transformed_data,
            columns=[f"PC{i+1}" for i in range(n_components_selected)],
            index=data_for_pca.index,
        )

        explained_variance_ratio = pca.explained_variance_ratio_.tolist()
        cumulative_explained_variance = np.cumsum(explained_variance_ratio).tolist()
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f"PC{i+1}" for i in range(n_components_selected)],
            index=columns_to_analyze,
        )

        # Calculate feature importance based on loadings
        feature_importance = pd.DataFrame(index=columns_to_analyze)
        for i in range(n_components_selected):
            pc_name = f"PC{i+1}"
            # Square the loadings and multiply by explained variance ratio
            feature_importance[pc_name] = loadings[pc_name]**2 * explained_variance_ratio[i]
        
        # Sum across all components to get overall importance
        feature_importance['Total_Importance'] = feature_importance.sum(axis=1)
        feature_importance = feature_importance.sort_values('Total_Importance', ascending=False)

        results = {
            "pca_transformed_data": pca_df,
            "explained_variance_ratio": explained_variance_ratio,
            "cumulative_explained_variance": cumulative_explained_variance,
            "loadings": loadings,
            "feature_importance": feature_importance,
            "n_components_selected": n_components_selected,
            "scaler": scaler,  # Include the scaler for future transformations
            "pca_model": pca,  # Include the PCA model for future use
        }

        if generate_plots:
            self._log("Generating PCA visualization plots...")
            current_show_plots = (
                show_plots if show_plots is not None else self.show_plots_default
            )
            current_output_dir_pca = self._prepare_output_dir(output_dir, "pca_plots")
            plot_paths = {}

            # Scree Plot (using all components initially to show full curve if auto-selecting)
            pca_full_for_scree = PCA(n_components=actual_max_components)
            pca_full_for_scree.fit(scaled_data)
            full_evr = pca_full_for_scree.explained_variance_ratio_
            full_cumulative_evr = np.cumsum(full_evr)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot 1: Individual and cumulative explained variance
            ax1.plot(
                range(1, len(full_evr) + 1), 
                full_evr, 
                'o-', 
                color='blue',
                linewidth=2,
                markersize=8,
                label="Individual"
            )
            ax1.plot(
                range(1, len(full_cumulative_evr) + 1),
                full_cumulative_evr,
                's-',
                color='red',
                linewidth=2,
                markersize=8,
                label="Cumulative"
            )
            ax1.set_xlabel("Principal Component", fontsize=12)
            ax1.set_ylabel("Explained Variance Ratio", fontsize=12)
            ax1.set_title("Scree Plot", fontsize=14)
            ax1.axhline(
                y=variance_threshold_for_auto_n,
                color='green',
                linestyle='--',
                linewidth=2,
                label=f"{variance_threshold_for_auto_n*100:.0f}% Threshold"
            )
            ax1.axvline(
                x=n_components_selected,
                color='purple',
                linestyle=':',
                linewidth=2,
                label=f"Selected: {n_components_selected}"
            )
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='best')
            
            # Plot 2: Bar chart of explained variance
            bars = ax2.bar(
                range(1, len(full_evr) + 1),
                full_evr,
                color='skyblue',
                edgecolor='black',
                alpha=0.7
            )
            
            # Highlight selected components
            for i in range(n_components_selected):
                bars[i].set_color('orange')
                bars[i].set_alpha(0.9)
            
            # Add cumulative line
            ax2_twin = ax2.twinx()
            ax2_twin.plot(
                range(1, len(full_cumulative_evr) + 1),
                full_cumulative_evr * 100,  # Convert to percentage
                's-',
                color='red',
                linewidth=2,
                markersize=6
            )
            ax2_twin.set_ylabel("Cumulative Explained Variance (%)", fontsize=12)
            ax2_twin.axhline(
                y=variance_threshold_for_auto_n * 100,
                color='green',
                linestyle='--',
                linewidth=2
            )
            
            ax2.set_xlabel("Principal Component", fontsize=12)
            ax2.set_ylabel("Explained Variance Ratio", fontsize=12)
            ax2.set_title("Explained Variance by Component", fontsize=14)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if current_output_dir_pca:
                path = os.path.join(current_output_dir_pca, "pca_scree_plot.png")
                fig.savefig(path, bbox_inches='tight', dpi=300)
                plot_paths["scree_plot"] = path
            if current_show_plots:
                plt.show()
            plt.close(fig)

            # Loadings Heatmap with enhanced visualization
            if not loadings.empty:
                fig, ax = plt.subplots(
                    figsize=(
                        max(8, loadings.shape[1] * 0.8),
                        max(6, loadings.shape[0] * 0.4),
                    )
                )
                
                # Create a diverging colormap centered at 0
                cmap = sns.diverging_palette(240, 10, as_cmap=True)
                
                # Plot heatmap with improved aesthetics
                sns.heatmap(
                    loadings, 
                    annot=True, 
                    cmap=cmap, 
                    fmt=".2f", 
                    ax=ax,
                    center=0,
                    linewidths=0.5,
                    cbar_kws={"shrink": .8, "label": "Loading Value"}
                )
                
                ax.set_title("PCA Component Loadings", fontsize=14, pad=20)
                plt.tight_layout()
                
                if current_output_dir_pca:
                    path = os.path.join(
                        current_output_dir_pca, "pca_loadings_heatmap.png"
                    )
                    fig.savefig(path, bbox_inches='tight', dpi=300)
                    plot_paths["loadings_heatmap"] = path
                if current_show_plots:
                    plt.show()
                plt.close(fig)
                
                # Feature importance plot
                fig, ax = plt.subplots(figsize=(10, max(6, len(columns_to_analyze) * 0.4)))
                feature_importance.sort_values('Total_Importance').plot(
                    kind='barh', 
                    y='Total_Importance', 
                    ax=ax,
                    color='skyblue',
                    edgecolor='black'
                )
                ax.set_title("Feature Importance in PCA", fontsize=14)
                ax.set_xlabel("Importance Score", fontsize=12)
                ax.grid(True, alpha=0.3)
                
                if current_output_dir_pca:
                    path = os.path.join(
                        current_output_dir_pca, "pca_feature_importance.png"
                    )
                    fig.savefig(path, bbox_inches='tight', dpi=300)
                    plot_paths["feature_importance"] = path
                if current_show_plots:
                    plt.show()
                plt.close(fig)

            # Scatter plot of PC1 vs PC2 (if available)
            if n_components_selected >= 2:
                fig, ax = plt.subplots(figsize=(10, 8))
                scatter = ax.scatter(
                    pca_df["PC1"], 
                    pca_df["PC2"],
                    alpha=0.7,
                    s=70,
                    c=range(len(pca_df)),  # Color by index for visual variety
                    cmap='viridis',
                    edgecolor='k'
                )
                
                # Add a colorbar
                plt.colorbar(scatter, label='Data Point Index')
                
                # Add axis labels with explained variance
                ax.set_xlabel(
                    f"PC1 ({explained_variance_ratio[0]*100:.2f}% variance)",
                    fontsize=12
                )
                ax.set_ylabel(
                    f"PC2 ({explained_variance_ratio[1]*100:.2f}% variance)",
                    fontsize=12
                )
                ax.set_title("PC1 vs PC2 Scatter Plot", fontsize=14)
                
                # Add grid
                ax.grid(True, alpha=0.3)
                
                # Add origin lines
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
                ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
                
                if current_output_dir_pca:
                    path = os.path.join(
                        current_output_dir_pca, "pca_pc1_vs_pc2_scatter.png"
                    )
                    fig.savefig(path, bbox_inches='tight', dpi=300)
                    plot_paths["pc1_vs_pc2_scatter"] = path
                if current_show_plots:
                    plt.show()
                plt.close(fig)
                
                # Biplot (PC1 vs PC2 with feature vectors)
                fig, ax = plt.subplots(figsize=(12, 10))
                
                # Scale features for the plot
                n = loadings.shape[0]  # Number of features
                scale = 1.0  # Adjust this to scale the arrows
                
                # Plot data points
                scatter = ax.scatter(
                    pca_df["PC1"], 
                    pca_df["PC2"],
                    alpha=0.5,
                    s=50,
                    c='skyblue',
                    edgecolor='k'
                )
                
                # Plot feature vectors
                for i, feature in enumerate(loadings.index):
                    ax.arrow(
                        0, 0,  # Start at origin
                        loadings.iloc[i, 0] * scale,  # PC1 loading
                        loadings.iloc[i, 1] * scale,  # PC2 loading
                        head_width=0.05,
                        head_length=0.1,
                        fc='red',
                        ec='red'
                    )
                    
                    # Add feature name at the end of the arrow
                    ax.text(
                        loadings.iloc[i, 0] * scale * 1.15,
                        loadings.iloc[i, 1] * scale * 1.15,
                        feature,
                        color='red',
                        ha='center',
                        va='center',
                        fontweight='bold'
                    )
                
                # Set plot limits
                max_val = max(
                    abs(pca_df["PC1"].max()), 
                    abs(pca_df["PC1"].min()),
                    abs(pca_df["PC2"].max()), 
                    abs(pca_df["PC2"].min()),
                    scale + 0.5  # Add some space for arrows
                )
                ax.set_xlim(-max_val, max_val)
                ax.set_ylim(-max_val, max_val)
                
                # Add axis labels with explained variance
                ax.set_xlabel(
                    f"PC1 ({explained_variance_ratio[0]*100:.2f}% variance)",
                    fontsize=12
                )
                ax.set_ylabel(
                    f"PC2 ({explained_variance_ratio[1]*100:.2f}% variance)",
                    fontsize=12
                )
                ax.set_title("PCA Biplot: PC1 vs PC2 with Feature Vectors", fontsize=14)
                
                # Add grid and origin lines
                ax.grid(True, alpha=0.3)
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
                ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
                
                if current_output_dir_pca:
                    path = os.path.join(
                        current_output_dir_pca, "pca_biplot.png"
                    )
                    fig.savefig(path, bbox_inches='tight', dpi=300)
                    plot_paths["biplot"] = path
                if current_show_plots:
                    plt.show()
                plt.close(fig)

            if plot_paths:
                results["plot_paths"] = plot_paths
                
        self._log(f"PCA completed with {n_components_selected} components explaining {cumulative_explained_variance[-1]*100:.2f}% of variance.")
        return results

    @error_handler
    def filter_column_by_threshold(
        self,
        dataframe: pd.DataFrame,
        column_name: str,
        filter_type: str,
        thresholds: Union[float, Tuple[float, float]],
        inclusive: bool = True,
        generate_plot: bool = False,
        output_dir: Optional[str] = None,
        show_plots: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Applies a threshold-based filter (low-pass, high-pass, band-pass, band-stop)
        to a specified column.

        Parameters:
        -----------
        dataframe : pd.DataFrame
            The DataFrame to filter.
        column_name : str
            The name of the column to apply the filter to.
        filter_type : str
            Type of filter: 'low_pass', 'high_pass', 'band_pass', 'band_stop'.
        thresholds : float or tuple
            Threshold value(s). Single float for low/high_pass.
            Tuple (low, high) for band_pass/band_stop.
        inclusive : bool, default=True
            Whether the threshold(s) are inclusive.
        generate_plot : bool, default=False
            Whether to generate a plot comparing original and filtered data.
        output_dir : str, optional
            Directory to save the plot. Uses class default if None.
        show_plots : bool, optional
            Whether to display the plot. Uses class default if None.

        Returns:
        --------
        dict
            A dictionary containing:
            - 'filtered_dataframe': The DataFrame after filtering.
            - 'filter_summary': Dict with original, filtered, and removed counts.
            - 'plot_path' (optional): Path to the saved comparison plot.
            
        Examples:
        ---------
        >>> analyzer = ComprehensiveDataFrameAnalyzer()
        >>> filtered_results = analyzer.filter_column_by_threshold(
        ...     df, 'income', 'high_pass', 50000, inclusive=True
        ... )
        >>> filtered_df = filtered_results['filtered_dataframe']
        >>> print(f"Kept {filtered_results['filter_summary']['filtered_count']} rows")
        """
        self._log(f"Starting threshold-based filtering for column '{column_name}'...")
        
        if column_name not in dataframe.columns:
            return {"error": f"Column '{column_name}' not found in DataFrame."}
        if not pd.api.types.is_numeric_dtype(dataframe[column_name]):
            return {
                "error": f"Column '{column_name}' is not numerical and cannot be filtered by threshold."
            }

        original_df = dataframe.copy()  # Work on a copy
        data_col = original_df[column_name]
        
        # Initialize mask that keeps NaNs by default
        # This ensures NaNs are not inadvertently dropped by boolean logic
        final_mask = pd.Series(True, index=data_col.index)
        
        self._log(f"Applying {filter_type} filter with threshold(s): {thresholds}")
        
        # Apply the appropriate filter based on filter_type
        if filter_type == "low_pass":
            if not isinstance(thresholds, (int, float)):
                return {
                    "error": "Low-pass filter requires a single numerical threshold."
                }
            condition = data_col <= thresholds if inclusive else data_col < thresholds
            filter_description = f"≤ {thresholds}" if inclusive else f"< {thresholds}"
            
        elif filter_type == "high_pass":
            if not isinstance(thresholds, (int, float)):
                return {
                    "error": "High-pass filter requires a single numerical threshold."
                }
            condition = data_col >= thresholds if inclusive else data_col > thresholds
            filter_description = f"≥ {thresholds}" if inclusive else f"> {thresholds}"
            
        elif filter_type == "band_pass":
            if not (
                isinstance(thresholds, tuple)
                and len(thresholds) == 2
                and all(isinstance(t, (int, float)) for t in thresholds)
            ):
                return {
                    "error": "Band-pass filter requires a tuple of two numerical thresholds (low, high)."
                }
            low, high = thresholds
            if low >= high:
                return {
                    "error": "For band-pass, low threshold must be less than high threshold."
                }
            if inclusive:
                condition = (data_col >= low) & (data_col <= high)
                filter_description = f"[{low}, {high}]"
            else:
                condition = (data_col > low) & (data_col < high)
                filter_description = f"({low}, {high})"
                
        elif filter_type == "band_stop":  # Values to REMOVE are between thresholds
            if not (
                isinstance(thresholds, tuple)
                and len(thresholds) == 2
                and all(isinstance(t, (int, float)) for t in thresholds)
            ):
                return {
                    "error": "Band-stop filter requires a tuple of two numerical thresholds (low, high)."
                }
            low, high = thresholds
            if low >= high:
                return {
                    "error": "For band-stop, low threshold must be less than high threshold."
                }
            if inclusive:  # Inclusive removal: remove if low <= value <= high
                condition = (data_col < low) | (
                    data_col > high
                )  # Keep if outside this band
                filter_description = f"outside [{low}, {high}]"
            else:  # Exclusive removal: remove if low < value < high
                condition = (data_col <= low) | (
                    data_col >= high
                )  # Keep if outside or equal to bounds
                filter_description = f"outside ({low}, {high})"
        else:
            return {
                "error": "Invalid filter_type. Choose from 'low_pass', 'high_pass', 'band_pass', 'band_stop'."
            }

        # Apply the condition while preserving NaN values
        non_nan_mask = ~data_col.isna()
        final_mask[non_nan_mask] = condition[non_nan_mask]
        
        filtered_dataframe = original_df[final_mask]

        # Calculate detailed summary statistics
        total_rows = len(original_df)
        filtered_rows = len(filtered_dataframe)
        removed_rows = total_rows - filtered_rows
        
        # Count NaN values
        nan_count = data_col.isna().sum()
        non_nan_total = total_rows - nan_count
        
        # Calculate percentages
        if total_rows > 0:
            filtered_percentage = (filtered_rows / total_rows) * 100
            removed_percentage = (removed_rows / total_rows) * 100
        else:
            filtered_percentage = 0
            removed_percentage = 0
            
        if non_nan_total > 0:
            # Calculate percentage of non-NaN values that passed the filter
            non_nan_filtered = filtered_dataframe[column_name].notna().sum()
            non_nan_filtered_percentage = (non_nan_filtered / non_nan_total) * 100
        else:
            non_nan_filtered_percentage = 0

        summary = {
            "original_count": total_rows,
            "filtered_count": filtered_rows,
            "removed_count": removed_rows,
            "filtered_percentage": filtered_percentage,
            "removed_percentage": removed_percentage,
            "nan_count": nan_count,
            "non_nan_filtered_percentage": non_nan_filtered_percentage,
            "filter_type": filter_type,
            "filter_description": filter_description,
            "thresholds": thresholds,
            "inclusive": inclusive
        }
        
        results = {"filtered_dataframe": filtered_dataframe, "filter_summary": summary}

        if generate_plot:
            self._log("Generating filter comparison plots...")
            current_show_plots = (
                show_plots if show_plots is not None else self.show_plots_default
            )
            current_output_dir_filter = self._prepare_output_dir(
                output_dir,
                f"filter_plots/{column_name.replace('/', '_').replace(' ', '_')}",
            )

            # Create enhanced visualization with multiple plots
            fig = plt.figure(figsize=(15, 10))
            
            # Define a grid layout
            gs = plt.GridSpec(2, 2, figure=fig)
            
            # Plot 1: Histogram comparison (original vs filtered)
            ax1 = fig.add_subplot(gs[0, :])
            
            # Get non-NaN data for plotting
            original_data = original_df[column_name].dropna()
            filtered_data = filtered_dataframe[column_name].dropna()
            
            # Determine common bin edges for both histograms
            if not original_data.empty:
                # Calculate optimal bin count based on data size
                n_bins = min(50, max(10, int(np.sqrt(len(original_data)))))
                
                # Plot original data histogram
                sns.histplot(
                    original_data,
                    bins=n_bins,
                    color='blue',
                    alpha=0.5,
                    label=f"Original ({len(original_data)} points)",
                    ax=ax1,
                    kde=True
                )
                
                # Plot filtered data histogram
                if not filtered_data.empty:
                    sns.histplot(
                        filtered_data,
                        bins=n_bins,
                        color='green',
                        alpha=0.5,
                        label=f"Filtered ({len(filtered_data)} points)",
                        ax=ax1,
                        kde=True
                    )
                
                # Add threshold markers
                if filter_type in ["low_pass", "high_pass"]:
                    ax1.axvline(
                        thresholds, 
                        color='red', 
                        linestyle='--',
                        linewidth=2,
                        label=f"Threshold: {thresholds}"
                    )
                elif filter_type in ["band_pass", "band_stop"]:
                    ax1.axvline(
                        thresholds[0], 
                        color='red', 
                        linestyle='--',
                        linewidth=2,
                        label=f"Lower: {thresholds[0]}"
                    )
                    ax1.axvline(
                        thresholds[1], 
                        color='red', 
                        linestyle='--',
                        linewidth=2,
                        label=f"Upper: {thresholds[1]}"
                    )
                
                # Add shaded regions to show filtered areas
                x_min, x_max = ax1.get_xlim()
                y_min, y_max = ax1.get_ylim()
                
                if filter_type == "low_pass":
                    # Shade the region that's kept (below threshold)
                    if inclusive:
                        ax1.fill_betweenx(
                            [y_min, y_max], 
                            x_min, 
                            min(thresholds, x_max), 
                            color='green', 
                            alpha=0.1
                        )
                    else:
                        ax1.fill_betweenx(
                            [y_min, y_max], 
                            x_min, 
                            min(thresholds, x_max), 
                            color='green', 
                            alpha=0.1
                        )
                        
                elif filter_type == "high_pass":
                    # Shade the region that's kept (above threshold)
                    if inclusive:
                        ax1.fill_betweenx(
                            [y_min, y_max], 
                            max(thresholds, x_min), 
                            x_max, 
                            color='green', 
                            alpha=0.1
                        )
                    else:
                        ax1.fill_betweenx(
                            [y_min, y_max], 
                            max(thresholds, x_min), 
                            x_max, 
                            color='green', 
                            alpha=0.1
                        )
                        
                elif filter_type == "band_pass":
                    # Shade the region that's kept (between thresholds)
                    low, high = thresholds
                    ax1.fill_betweenx(
                        [y_min, y_max], 
                        max(low, x_min), 
                        min(high, x_max), 
                        color='green', 
                        alpha=0.1
                    )
                    
                elif filter_type == "band_stop":
                    # Shade the regions that are kept (outside thresholds)
                    low, high = thresholds
                    ax1.fill_betweenx(
                        [y_min, y_max], 
                        x_min, 
                        min(low, x_max), 
                        color='green', 
                        alpha=0.1
                    )
                    ax1.fill_betweenx(
                        [y_min, y_max], 
                        max(high, x_min), 
                        x_max, 
                        color='green', 
                        alpha=0.1
                    )
                
                ax1.set_title(
                    f"Distribution Comparison for {column_name}\n"
                    f"Filter: {filter_type.replace('_', ' ').title()} {filter_description}",
                    fontsize=14
                )
                ax1.set_xlabel(column_name, fontsize=12)
                ax1.set_ylabel("Frequency", fontsize=12)
                ax1.legend(loc='best')
                ax1.grid(True, alpha=0.3)
            
            # Plot 2: Box plots comparison
            ax2 = fig.add_subplot(gs[1, 0])
            
            box_data = [
                original_data,
                filtered_data
            ]
            box_labels = [
                f"Original\n({len(original_data)} points)",
                f"Filtered\n({len(filtered_data)} points)"
            ]
            
            sns.boxplot(
                data=box_data,
                ax=ax2,
                palette=['blue', 'green'],
                width=0.5
            )
            
            # Add threshold markers to boxplot
            if filter_type in ["low_pass", "high_pass"]:
                ax2.axhline(
                    thresholds, 
                    color='red', 
                    linestyle='--',
                    linewidth=2
                )
            elif filter_type in ["band_pass", "band_stop"]:
                ax2.axhline(
                    thresholds[0], 
                    color='red', 
                    linestyle='--',
                    linewidth=2
                )
                ax2.axhline(
                    thresholds[1], 
                    color='red', 
                    linestyle='--',
                    linewidth=2
                )
            
            ax2.set_xticklabels(box_labels)
            ax2.set_title("Box Plot Comparison", fontsize=14)
            ax2.set_ylabel(column_name, fontsize=12)
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Summary statistics
            ax3 = fig.add_subplot(gs[1, 1])
            ax3.axis('off')  # Turn off axis
            
            # Create a text summary
            summary_text = (
                f"Filter Summary for '{column_name}':\n\n"
                f"Filter Type: {filter_type.replace('_', ' ').title()}\n"
                f"Threshold: {filter_description}\n"
                f"Inclusive: {inclusive}\n\n"
                f"Original Rows: {total_rows}\n"
                f"Filtered Rows: {filtered_rows} ({filtered_percentage:.1f}%)\n"
                f"Removed Rows: {removed_rows} ({removed_percentage:.1f}%)\n"
                f"NaN Values: {nan_count}\n\n"
                f"Statistics Before/After:\n"
            )
            
            # Add statistics comparison if we have data
            if not original_data.empty or not filtered_data.empty:
                stats_comparison = pd.DataFrame(index=['Original', 'Filtered'])
                
                if not original_data.empty:
                    stats_comparison.loc['Original', 'Count'] = len(original_data)
                    stats_comparison.loc['Original', 'Mean'] = original_data.mean()
                    stats_comparison.loc['Original', 'Median'] = original_data.median()
                    stats_comparison.loc['Original', 'Std Dev'] = original_data.std()
                    stats_comparison.loc['Original', 'Min'] = original_data.min()
                    stats_comparison.loc['Original', 'Max'] = original_data.max()
                
                if not filtered_data.empty:
                    stats_comparison.loc['Filtered', 'Count'] = len(filtered_data)
                    stats_comparison.loc['Filtered', 'Mean'] = filtered_data.mean()
                    stats_comparison.loc['Filtered', 'Median'] = filtered_data.median()
                    stats_comparison.loc['Filtered', 'Std Dev'] = filtered_data.std()
                    stats_comparison.loc['Filtered', 'Min'] = filtered_data.min()
                    stats_comparison.loc['Filtered', 'Max'] = filtered_data.max()
                
                # Format the stats table
                stats_table = stats_comparison.round(2).to_string()
                
                # Add the stats table to the summary text
                summary_text += stats_table
            
            # Display the summary text
            ax3.text(
                0.05, 0.95, summary_text,
                transform=ax3.transAxes,
                fontsize=12,
                verticalalignment='top',
                horizontalalignment='left',
                family='monospace'
            )
            
            plt.tight_layout()
            
            if current_output_dir_filter:
                path = os.path.join(
                    current_output_dir_filter,
                    f'{column_name.replace("/", "_").replace(" ", "_")}_{filter_type}_comparison.png',
                )
                fig.savefig(path, bbox_inches='tight', dpi=300)
                results["plot_path"] = path
            if current_show_plots:
                plt.show()
            plt.close(fig)
            
        self._log(f"Filtering completed. Kept {filtered_rows}/{total_rows} rows ({filtered_percentage:.1f}%).")
        return results

    def identify_skewed_columns(
        self, 
        dataframe: pd.DataFrame, 
        skewness_threshold: float = 1.0
    ) -> Dict[str, float]:
        """
        Identifies columns with significant skewness.
        
        Parameters:
        -----------
        dataframe : pd.DataFrame
            The DataFrame to analyze.
        skewness_threshold : float, default=1.0
            Absolute skewness value above which a column is considered significantly skewed.
            
        Returns:
        --------
        Dict[str, float]
            Dictionary of column names and their skewness values for columns exceeding the threshold.
            
        Examples:
        ---------
        >>> analyzer = ComprehensiveDataFrameAnalyzer()
        >>> skewed_cols = analyzer.identify_skewed_columns(df, skewness_threshold=0.8)
        >>> print(f"Found {len(skewed_cols)} skewed columns")
        """
        self._log(f"Identifying columns with skewness > {skewness_threshold}...")
        
        # Get numerical columns
        num_df = dataframe.select_dtypes(include=np.number)
        
        if num_df.empty:
            return {}
        
        # Calculate skewness for each column
        skewness_values = num_df.skew()
        
        # Filter columns with absolute skewness above threshold
        skewed_columns = skewness_values[abs(skewness_values) > skewness_threshold]
        
        # Sort by absolute skewness (most skewed first)
        skewed_columns = skewed_columns.reindex(
            skewed_columns.abs().sort_values(ascending=False).index
        )
        
        self._log(f"Found {len(skewed_columns)} columns with significant skewness.")
        return skewed_columns.to_dict()
    
    def suggest_transformations(
        self, 
        dataframe: pd.DataFrame, 
        column_name: str
    ) -> Dict[str, Any]:
        """
        Suggests and applies common transformations to improve normality of a column.
        
        Parameters:
        -----------
        dataframe : pd.DataFrame
            The DataFrame containing the column.
        column_name : str
            The name of the column to transform.
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing:
            - 'original_skewness': Skewness of the original data
            - 'transformations': Dict of transformation names and their resulting skewness
            - 'best_transformation': Name of the transformation that minimized absolute skewness
            - 'transformed_data': Series with the best transformation applied
            
        Examples:
        ---------
        >>> analyzer = ComprehensiveDataFrameAnalyzer()
        >>> transform_results = analyzer.suggest_transformations(df, 'income')
        >>> print(f"Best transformation: {transform_results['best_transformation']}")
        """
        self._log(f"Suggesting transformations for column '{column_name}'...")
        
        if column_name not in dataframe.columns:
            return {"error": f"Column '{column_name}' not found in DataFrame."}
        
        if not pd.api.types.is_numeric_dtype(dataframe[column_name]):
            return {"error": f"Column '{column_name}' is not numerical."}
        
        # Get the data and remove NaN values
        data = dataframe[column_name].dropna()
        
        if len(data) < 5:
            return {"error": f"Insufficient data in column '{column_name}' after removing NaN values."}
        
        # Calculate original skewness
        original_skewness = data.skew()
        
        transformations = {}
        transformed_data = {}
        
        # Log transformation (for positive data)
        if data.min() > 0:
            log_data = np.log(data)
            transformations['log'] = log_data.skew()
            transformed_data['log'] = log_data
        
        # Square root transformation (for non-negative data)
        if data.min() >= 0:
            sqrt_data = np.sqrt(data)
            transformations['sqrt'] = sqrt_data.skew()
            transformed_data['sqrt'] = sqrt_data
        
        # Reciprocal transformation (for non-zero data)
        if not (data == 0).any():
            recip_data = 1 / data
            transformations['reciprocal'] = recip_data.skew()
            transformed_data['reciprocal'] = recip_data
        
        # Box-Cox transformation (for positive data)
        if data.min() > 0:
            try:
                boxcox_data, lambda_param = scipy_stats.boxcox(data)
                boxcox_data = pd.Series(boxcox_data, index=data.index)
                transformations['boxcox'] = boxcox_data.skew()
                transformed_data['boxcox'] = boxcox_data
                transformations['boxcox_lambda'] = lambda_param
            except Exception as e:
                self._log(f"Box-Cox transformation failed: {str(e)}")
        
        # Yeo-Johnson transformation (works with negative values too)
        try:
            yeojohnson_data, lambda_param = scipy_stats.yeojohnson(data)
            yeojohnson_data = pd.Series(yeojohnson_data, index=data.index)
            transformations['yeojohnson'] = yeojohnson_data.skew()
            transformed_data['yeojohnson'] = yeojohnson_data
            transformations['yeojohnson_lambda'] = lambda_param
        except Exception as e:
            self._log(f"Yeo-Johnson transformation failed: {str(e)}")
        
        # Find the best transformation (closest to zero skewness)
        if transformations:
            best_transform = min(
                [t for t in transformations.keys() if not t.endswith('_lambda')],
                key=lambda t: abs(transformations[t])
            )
            
            results = {
                'original_skewness': original_skewness,
                'transformations': {k: v for k, v in transformations.items() if not k.endswith('_lambda')},
                'best_transformation': best_transform,
                'transformed_data': transformed_data[best_transform],
                'transformation_parameters': {
                    k: transformations[k] for k in transformations if k.endswith('_lambda')
                }
            }
        else:
            results = {
                'original_skewness': original_skewness,
                'transformations': {},
                'best_transformation': None,
                'transformed_data': data,  # Return original data if no transformations were possible
                'message': "No suitable transformations found for this data."
            }
        
        self._log(f"Transformation analysis completed for '{column_name}'.")
        return results
    
    def analyze_missing_values(
        self, 
        dataframe: pd.DataFrame,
        generate_plot: bool = False,
        output_dir: Optional[str] = None,
        show_plots: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Performs comprehensive analysis of missing values in the DataFrame.
        
        Parameters:
        -----------
        dataframe : pd.DataFrame
            The DataFrame to analyze.
        generate_plot : bool, default=False
            Whether to generate visualizations of missing value patterns.
        output_dir : str, optional
            Directory to save plots. Uses class default if None.
        show_plots : bool, optional
            Whether to display plots. Uses class default if None.
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing:
            - 'missing_by_column': Series with count of missing values per column
            - 'missing_by_row': Series with count of missing values per row
            - 'columns_percent_missing': Series with percentage of missing values per column
            - 'rows_with_any_missing': Count of rows with at least one missing value
            - 'completely_missing_rows': Count of rows with all values missing
            - 'plot_paths' (optional): Paths to saved plots
            
        Examples:
        ---------
        >>> analyzer = ComprehensiveDataFrameAnalyzer()
        >>> missing_analysis = analyzer.analyze_missing_values(df, generate_plot=True)
        >>> print(f"Columns with highest missing %: {missing_analysis['columns_percent_missing'].nlargest(3)}")
        """
        self._log("Starting missing values analysis...")
        
        total_cells = dataframe.size
        total_missing = dataframe.isna().sum().sum()
        
        if total_cells == 0:
            return {"error": "DataFrame is empty."}
        
        # Calculate missing values by column
        missing_by_column = dataframe.isna().sum()
        columns_percent_missing = (missing_by_column / len(dataframe) * 100).round(2)
        
        # Calculate missing values by row
        missing_by_row = dataframe.isna().sum(axis=1)
        rows_with_any_missing = (missing_by_row > 0).sum()
        completely_missing_rows = (missing_by_row == dataframe.shape[1]).sum()
        
        # Identify columns with no missing values
        columns_no_missing = missing_by_column[missing_by_column == 0].index.tolist()
        
        results = {
            'total_missing': total_missing,
            'total_missing_percent': (total_missing / total_cells * 100).round(2),
            'missing_by_column': missing_by_column[missing_by_column > 0].sort_values(ascending=False),
            'missing_by_row': missing_by_row,
            'columns_percent_missing': columns_percent_missing[columns_percent_missing > 0].sort_values(ascending=False),
            'rows_with_any_missing': rows_with_any_missing,
            'rows_with_any_missing_percent': (rows_with_any_missing / len(dataframe) * 100).round(2),
            'completely_missing_rows': completely_missing_rows,
            'columns_no_missing': columns_no_missing
        }
        
        if generate_plot:
            self._log("Generating missing values visualizations...")
            current_show_plots = (
                show_plots if show_plots is not None else self.show_plots_default
            )
            current_output_dir = self._prepare_output_dir(
                output_dir, "missing_values_plots"
            )
            
            plot_paths = {}
            
            # Create a heatmap of missing values
            if total_missing > 0:
                # For large datasets, sample rows to keep the plot manageable
                max_rows_to_plot = 100
                if len(dataframe) > max_rows_to_plot:
                    # Sample rows with missing values to ensure they're represented
                    rows_with_missing = dataframe[missing_by_row > 0].index
                    if len(rows_with_missing) > max_rows_to_plot // 2:
                        sampled_missing_rows = np.random.choice(
                            rows_with_missing, 
                            size=max_rows_to_plot // 2, 
                            replace=False
                        )
                    else:
                        sampled_missing_rows = rows_with_missing
                    
                    # Sample from rows without missing values for the remainder
                    rows_without_missing = dataframe[missing_by_row == 0].index
                    remaining_slots = max_rows_to_plot - len(sampled_missing_rows)
                    
                    if len(rows_without_missing) > remaining_slots and remaining_slots > 0:
                        sampled_complete_rows = np.random.choice(
                            rows_without_missing,
                            size=remaining_slots,
                            replace=False
                        )
                    else:
                        sampled_complete_rows = rows_without_missing
                    
                    # Combine the samples
                    sampled_rows = np.concatenate([sampled_missing_rows, sampled_complete_rows])
                    
                    # Create a sampled DataFrame for plotting
                    plot_df = dataframe.loc[sampled_rows]
                    title_suffix = f" (Sample of {max_rows_to_plot} rows)"
                else:
                    plot_df = dataframe
                    title_suffix = ""
                
                # Create missing values heatmap
                plt.figure(figsize=(12, 8))
                ax = plt.gca()
                
                # Use a mask to show only missing values
                missing_mask = plot_df.isna()
                
                # Create a custom colormap with only one color
                cmap = plt.cm.get_cmap('Reds', 2)
                
                # Plot the heatmap
                sns.heatmap(
                    missing_mask,
                    cbar=False,
                    cmap=cmap,
                    yticklabels=False,
                    ax=ax
                )
                
                plt.title(f"Missing Values Heatmap{title_suffix}", fontsize=14)
                plt.xlabel("Columns", fontsize=12)
                plt.ylabel("Rows", fontsize=12)
                plt.tight_layout()
                
                if current_output_dir:
                    path = os.path.join(current_output_dir, "missing_values_heatmap.png")
                    plt.savefig(path, bbox_inches='tight', dpi=300)
                    plot_paths['heatmap'] = path
                if current_show_plots:
                    plt.show()
                plt.close()
                
                # Create a bar chart of missing values by column
                if not results['columns_percent_missing'].empty:
                    plt.figure(figsize=(12, 6))
                    
                    # Sort by missing percentage
                    missing_percent_sorted = results['columns_percent_missing'].sort_values(ascending=False)
                    
                    ax = missing_percent_sorted.plot(
                        kind='bar',
                        color='coral',
                        edgecolor='black',
                        alpha=0.7
                    )
                    
                    # Add data labels
                    for i, v in enumerate(missing_percent_sorted):
                        ax.text(
                            i, 
                            v + 1, 
                            f"{v:.1f}%", 
                            ha='center',
                            fontweight='bold'
                        )
                    
                    plt.title("Percentage of Missing Values by Column", fontsize=14)
                    plt.xlabel("Column", fontsize=12)
                    plt.ylabel("Missing Values (%)", fontsize=12)
                    plt.xticks(rotation=45, ha='right')
                    plt.grid(axis='y', alpha=0.3)
                    plt.tight_layout()
                    
                    if current_output_dir:
                        path = os.path.join(current_output_dir, "missing_values_by_column.png")
                        plt.savefig(path, bbox_inches='tight', dpi=300)
                        plot_paths['by_column'] = path
                    if current_show_plots:
                        plt.show()
                    plt.close()
                
                # Create a histogram of missing values per row
                if rows_with_any_missing > 0:
                    plt.figure(figsize=(10, 6))
                    
                    # Create bins for the histogram
                    max_missing = missing_by_row.max()
                    
                    # Only include rows with missing values
                    missing_counts = missing_by_row[missing_by_row > 0]
                    
                    # Plot histogram
                    sns.histplot(
                        missing_counts,
                        bins=min(20, int(max_missing)),
                        kde=False,
                        color='skyblue',
                        edgecolor='black',
                        alpha=0.7
                    )
                    
                    plt.title("Distribution of Missing Values per Row", fontsize=14)
                    plt.xlabel("Number of Missing Values", fontsize=12)
                    plt.ylabel("Number of Rows", fontsize=12)
                    plt.grid(axis='y', alpha=0.3)
                    plt.tight_layout()
                    
                    if current_output_dir:
                        path = os.path.join(current_output_dir, "missing_values_per_row.png")
                        plt.savefig(path, bbox_inches='tight', dpi=300)
                        plot_paths['per_row'] = path
                    if current_show_plots:
                        plt.show()
                    plt.close()
                
                # Create a correlation heatmap of missing values
                if len(results['missing_by_column']) > 1:
                    # Create a DataFrame where each cell is True if the value is missing
                    missing_binary = dataframe.isna().astype(int)
                    
                    # Only include columns with missing values
                    cols_with_missing = missing_by_column[missing_by_column > 0].index
                    missing_binary = missing_binary[cols_with_missing]
                    
                    # Calculate correlation of missingness
                    missing_corr = missing_binary.corr()
                    
                    plt.figure(figsize=(10, 8))
                    
                    # Create a mask for the upper triangle
                    mask = np.triu(np.ones_like(missing_corr, dtype=bool))
                    
                    # Plot the heatmap
                    sns.heatmap(
                        missing_corr,
                        mask=mask,
                        cmap='coolwarm',
                        vmin=-1,
                        vmax=1,
                        center=0,
                        square=True,
                        linewidths=.5,
                        annot=True,
                        fmt='.2f'
                    )
                    
                    plt.title("Correlation of Missing Values Between Columns", fontsize=14)
                    plt.tight_layout()
                    
                    if current_output_dir:
                        path = os.path.join(current_output_dir, "missing_values_correlation.png")
                        plt.savefig(path, bbox_inches='tight', dpi=300)
                        plot_paths['correlation'] = path
                    if current_show_plots:
                        plt.show()
                    plt.close()
            
            if plot_paths:
                results['plot_paths'] = plot_paths
        
        self._log("Missing values analysis completed.")
        return results
    
    def compare_groups(
        self,
        dataframe: pd.DataFrame,
        numerical_column: str,
        groupby_column: str,
        test_type: str = 'auto',
        alpha: float = 0.05,
        generate_plots: bool = False,
        output_dir: Optional[str] = None,
        show_plots: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Compares the distribution of a numerical column across different groups.
        
        Parameters:
        -----------
        dataframe : pd.DataFrame
            The DataFrame containing the data.
        numerical_column : str
            The name of the numerical column to analyze.
        groupby_column : str
            The name of the column containing group labels.
        test_type : str, default='auto'
            Statistical test to use: 'ttest' (two groups), 'anova' (multiple groups), 
            'mannwhitney' (two groups, non-parametric), 'kruskal' (multiple groups, non-parametric),
            or 'auto' to automatically select based on data properties.
        alpha : float, default=0.05
            Significance level for the statistical test.
        generate_plots : bool, default=False
            Whether to generate comparison plots.
        output_dir : str, optional
            Directory to save plots. Uses class default if None.
        show_plots : bool, optional
            Whether to display plots. Uses class default if None.
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing:
            - 'group_stats': DataFrame with descriptive statistics for each group
            - 'test_results': Results of the statistical test
            - 'test_type': The test that was used
            - 'p_value': p-value from the test
            - 'significant': Whether the difference is statistically significant
            - 'plot_paths' (optional): Paths to saved plots
            
        Examples:
        ---------
        >>> analyzer = ComprehensiveDataFrameAnalyzer()
        >>> comparison = analyzer.compare_groups(
        ...     df, numerical_column='income', groupby_column='gender'
        ... )
        >>> print(f"P-value: {comparison['p_value']}, Significant: {comparison['significant']}")
        """
        self._log(f"Comparing groups in '{groupby_column}' for '{numerical_column}'...")
        
        if numerical_column not in dataframe.columns:
            return {"error": f"Column '{numerical_column}' not found in DataFrame."}
        
        if groupby_column not in dataframe.columns:
            return {"error": f"Column '{groupby_column}' not found in DataFrame."}
        
        if not pd.api.types.is_numeric_dtype(dataframe[numerical_column]):
            return {"error": f"Column '{numerical_column}' is not numerical."}
        
        # Get the groups
        groups = dataframe.groupby(groupby_column)[numerical_column].apply(lambda x: x.dropna()).reset_index()
        unique_groups = groups[groupby_column].unique()
        num_groups = len(unique_groups)
        
        if num_groups < 2:
            return {"error": f"Need at least 2 groups for comparison, but found {num_groups}."}
        
        # Calculate descriptive statistics for each group
        group_stats = dataframe.groupby(groupby_column)[numerical_column].describe()
        
        # Check sample sizes
        min_sample_size = group_stats['count'].min()
        if min_sample_size < 5:
            self._log(f"Warning: Small sample size detected ({min_sample_size} in smallest group).")
        
        # Determine appropriate statistical test if 'auto' is selected
        if test_type == 'auto':
            # Check normality for each group
            normality_results = {}
            normal_distribution = True
            
            for group_name in unique_groups:
                group_data = dataframe[dataframe[groupby_column] == group_name][numerical_column].dropna()
                
                # Skip if too few samples
                if len(group_data) < 8:  # Shapiro-Wilk works best with n ≥ 8
                    normality_results[group_name] = {
                        'test': 'skipped (too few samples)',
                        'normal': None
                    }
                    normal_distribution = False
                    continue
                
                # Perform Shapiro-Wilk test
                stat, p_val = scipy_stats.shapiro(group_data)
                normality_results[group_name] = {
                    'test': 'shapiro',
                    'statistic': stat,
                    'p_value': p_val,
                    'normal': p_val > alpha
                }
                
                if not normality_results[group_name]['normal']:
                    normal_distribution = False
            
            # Select test based on number of groups and normality
            if num_groups == 2:
                if normal_distribution:
                    # Check for equal variances (Levene's test)
                    group_data = [
                        dataframe[dataframe[groupby_column] == group][numerical_column].dropna()
                        for group in unique_groups
                    ]
                    _, p_val_levene = scipy_stats.levene(*group_data)
                    equal_variances = p_val_levene > alpha
                    
                    test_type = 'ttest'
                    if not equal_variances:
                        self._log("Unequal variances detected, using Welch's t-test.")
                else:
                    test_type = 'mannwhitney'
                    self._log("Non-normal distribution detected, using Mann-Whitney U test.")
            else:
                if normal_distribution:
                    test_type = 'anova'
                else:
                    test_type = 'kruskal'
                    self._log("Non-normal distribution detected, using Kruskal-Wallis test.")
        
        # Perform the selected statistical test
        test_results = {}
        p_value = None
        
        if test_type == 'ttest':
            # Extract data for the two groups
            group1_data = dataframe[dataframe[groupby_column] == unique_groups[0]][numerical_column].dropna()
            group2_data = dataframe[dataframe[groupby_column] == unique_groups[1]][numerical_column].dropna()
            
            # Perform t-test
            t_stat, p_value = scipy_stats.ttest_ind(
                group1_data, 
                group2_data,
                equal_var=True  # Use Student's t-test with equal variance assumption
            )
            
            test_results = {
                'test': 't-test',
                'statistic': t_stat,
                'p_value': p_value,
                'df': len(group1_data) + len(group2_data) - 2
            }
            
        elif test_type == 'mannwhitney':
            # Extract data for the two groups
            group1_data = dataframe[dataframe[groupby_column] == unique_groups[0]][numerical_column].dropna()
            group2_data = dataframe[dataframe[groupby_column] == unique_groups[1]][numerical_column].dropna()
            
            # Perform Mann-Whitney U test
            u_stat, p_value = scipy_stats.mannwhitneyu(
                group1_data, 
                group2_data,
                alternative='two-sided'
            )
            
            test_results = {
                'test': 'Mann-Whitney U test',
                'statistic': u_stat,
                'p_value': p_value
            }
            
        elif test_type == 'anova':
            # Extract data for all groups
            group_data = [
                dataframe[dataframe[groupby_column] == group][numerical_column].dropna()
                for group in unique_groups
            ]
            
            # Perform one-way ANOVA
            f_stat, p_value = scipy_stats.f_oneway(*group_data)
            
            test_results = {
                'test': 'One-way ANOVA',
                'statistic': f_stat,
                'p_value': p_value,
                'df_between': num_groups - 1,
                'df_within': sum(len(group) for group in group_data) - num_groups
            }
            
            # If ANOVA is significant, perform post-hoc tests
            if p_value < alpha:
                # Perform Tukey's HSD test for pairwise comparisons
                try:
                    from statsmodels.stats.multicomp import pairwise_tukeyhsd
                    
                    # Prepare data for Tukey's test
                    values = []
                    groups_labels = []
                    
                    for i, group in enumerate(unique_groups):
                        group_values = dataframe[dataframe[groupby_column] == group][numerical_column].dropna()
                        values.extend(group_values)
                        groups_labels.extend([group] * len(group_values))
                    
                    # Perform Tukey's test
                    tukey_results = pairwise_tukeyhsd(
                        values, 
                        groups_labels,
                        alpha=alpha
                    )
                    
                    # Convert results to a more usable format
                    tukey_summary = pd.DataFrame(
                        data=tukey_results._results_table.data[1:],
                        columns=tukey_results._results_table.data[0]
                    )
                    
                    test_results['post_hoc'] = {
                        'test': "Tukey's HSD",
                        'results': tukey_summary
                    }
                    
                except ImportError:
                    self._log("statsmodels not available for Tukey's HSD test.")
                    
        elif test_type == 'kruskal':
            # Extract data for all groups
            group_data = [
                dataframe[dataframe[groupby_column] == group][numerical_column].dropna()
                for group in unique_groups
            ]
            
            # Perform Kruskal-Wallis H-test
            h_stat, p_value = scipy_stats.kruskal(*group_data)
            
            test_results = {
                'test': 'Kruskal-Wallis H test',
                'statistic': h_stat,
                'p_value': p_value,
                'df': num_groups - 1
            }
            
            # If Kruskal-Wallis is significant, perform post-hoc tests
            if p_value < alpha and num_groups > 2:
                # Perform Dunn's test for pairwise comparisons
                try:
                    from scikit_posthocs import posthoc_dunn
                    
                    # Prepare data for Dunn's test
                    data_for_dunn = pd.DataFrame({
                        'value': np.concatenate([group for group in group_data]),
                        'group': np.concatenate([
                            [i] * len(group) for i, group in enumerate(group_data)
                        ])
                    })
                    
                    # Perform Dunn's test
                    dunn_results = posthoc_dunn(
                        data_for_dunn,
                        val_col='value',
                        group_col='group',
                        p_adjust='bonferroni'
                    )
                    
                    # Rename indices and columns to group names
                    group_mapping = {i: name for i, name in enumerate(unique_groups)}
                    dunn_results.index = [group_mapping[i] for i in dunn_results.index]
                    dunn_results.columns = [group_mapping[i] for i in dunn_results.columns]
                    
                    test_results['post_hoc'] = {
                        'test': "Dunn's test with Bonferroni correction",
                        'results': dunn_results
                    }
                    
                except ImportError:
                    self._log("scikit-posthocs not available for Dunn's test.")
        
        else:
            return {"error": f"Invalid test_type: {test_type}. Choose from 'ttest', 'anova', 'mannwhitney', 'kruskal', or 'auto'."}
        
        # Compile results
        results = {
            'group_stats': group_stats,
            'test_results': test_results,
            'test_type': test_type,
            'p_value': p_value,
            'significant': p_value < alpha if p_value is not None else None,
            'alpha': alpha,
            'num_groups': num_groups
        }
        
        if generate_plots:
            self._log("Generating group comparison plots...")
            current_show_plots = (
                show_plots if show_plots is not None else self.show_plots_default
            )
            current_output_dir = self._prepare_output_dir(
                output_dir, f"group_comparison_plots/{numerical_column}_by_{groupby_column}"
            )
            
            plot_paths = {}
            
            # Box plot comparison
            plt.figure(figsize=(12, 6))
            
            # Create box plot
            ax = sns.boxplot(
                x=groupby_column,
                y=numerical_column,
                data=dataframe,
                palette='Set3'
            )
            
            # Add individual data points
            sns.stripplot(
                x=groupby_column,
                y=numerical_column,
                data=dataframe,
                color='black',
                size=3,
                alpha=0.3,
                jitter=True
            )
            
            # Add mean markers
            for i, group in enumerate(unique_groups):
                group_mean = group_stats.loc[group]['mean']
                ax.plot(i, group_mean, 'ro', markersize=8, label='Mean' if i == 0 else "")
            
            # Add p-value annotation
            if p_value is not None:
                if p_value < 0.001:
                    p_text = "p < 0.001"
                else:
                    p_text = f"p = {p_value:.3f}"
                
                significance_marker = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                
                plt.annotate(
                    f"{p_text} {significance_marker}",
                    xy=(0.5, 0.01),
                    xycoords='axes fraction',
                    xytext=(0.5, 0.05),
                    textcoords='axes fraction',
                    ha='center',
                    va='bottom',
                    fontsize=12,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.1),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2')
                )
            
            plt.title(
                f"Comparison of {numerical_column} by {groupby_column}\n"
                f"Test: {test_results.get('test', 'Unknown')}, "
                f"{'Significant' if results['significant'] else 'Not Significant'} (α={alpha})",
                fontsize=14
            )
            plt.xlabel(groupby_column, fontsize=12)
            plt.ylabel(numerical_column, fontsize=12)
            plt.grid(axis='y', alpha=0.3)
            
            # Add legend for the mean marker
            handles, labels = ax.get_legend_handles_labels()
            if len(handles) < 1:  # If no legend exists yet
                from matplotlib.lines import Line2D
                handles = [Line2D([0], [0], marker='o', color='r', markersize=8, linestyle='')]
                labels = ['Mean']
                plt.legend(handles, labels, loc='best')
            
            plt.tight_layout()
            
            if current_output_dir:
                path = os.path.join(current_output_dir, "boxplot_comparison.png")
                plt.savefig(path, bbox_inches='tight', dpi=300)
                plot_paths['boxplot'] = path
            if current_show_plots:
                plt.show()
            plt.close()
            
            # Violin plot comparison with statistics
            plt.figure(figsize=(14, 8))
            
            # Create a subplot grid
            gs = plt.GridSpec(1, 2, width_ratios=[3, 1])
            
            # Violin plot
            ax1 = plt.subplot(gs[0])
            sns.violinplot(
                x=groupby_column,
                y=numerical_column,
                data=dataframe,
                inner='quartile',
                palette='Set3',
                ax=ax1
            )
            
            # Add individual data points
            sns.stripplot(
                x=groupby_column,
                y=numerical_column,
                data=dataframe,
                color='black',
                size=3,
                alpha=0.3,
                jitter=True,
                ax=ax1
            )
            
            ax1.set_title(
                f"Distribution of {numerical_column} by {groupby_column}",
                fontsize=14
            )
            ax1.set_xlabel(groupby_column, fontsize=12)
            ax1.set_ylabel(numerical_column, fontsize=12)
            ax1.grid(axis='y', alpha=0.3)
            
            # Statistics table
            ax2 = plt.subplot(gs[1])
            ax2.axis('off')
            
            # Create a table of statistics
            cell_text = []
            rows = []
            
            for group in unique_groups:
                rows.append(str(group))
                stats = group_stats.loc[group]
                cell_text.append([
                    f"{stats['count']:.0f}",
                    f"{stats['mean']:.2f}",
                    f"{stats['std']:.2f}",
                    f"{stats['min']:.2f}",
                    f"{stats['25%']:.2f}",
                    f"{stats['50%']:.2f}",
                    f"{stats['75%']:.2f}",
                    f"{stats['max']:.2f}"
                ])
            
            columns = ['Count', 'Mean', 'Std Dev', 'Min', '25%', 'Median', '75%', 'Max']
            
            table = ax2.table(
                cellText=cell_text,
                rowLabels=rows,
                colLabels=columns,
                cellLoc='center',
                loc='center',
                bbox=[0, 0.2, 1, 0.6]  # [left, bottom, width, height]
            )
            
            # Style the table
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)
            
            # Add test results below the table
            test_info = (
                f"Test: {test_results.get('test', 'Unknown')}\n"
                f"p-value: {p_value:.4f}\n"
                f"Significant: {'Yes' if results['significant'] else 'No'} (α={alpha})"
            )
            
            ax2.text(
                0.5, 0.1, 
                test_info,
                ha='center',
                va='center',
                fontsize=11,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.5)
            )
            
            plt.tight_layout()
            
            if current_output_dir:
                path = os.path.join(current_output_dir, "violin_with_stats.png")
                plt.savefig(path, bbox_inches='tight', dpi=300)
                plot_paths['violin_with_stats'] = path
            if current_show_plots:
                plt.show()
            plt.close()
            
            # Histogram comparison
            plt.figure(figsize=(12, 6))
            
            for group in unique_groups:
                group_data = dataframe[dataframe[groupby_column] == group][numerical_column].dropna()
                sns.histplot(
                    group_data,
                    kde=True,
                    label=f"{group} (n={len(group_data)})",
                    alpha=0.5
                )
            
            plt.title(f"Distribution of {numerical_column} by {groupby_column}", fontsize=14)
            plt.xlabel(numerical_column, fontsize=12)
            plt.ylabel("Frequency", fontsize=12)
            plt.grid(alpha=0.3)
            plt.legend(title=groupby_column)
            plt.tight_layout()
            
            if current_output_dir:
                path = os.path.join(current_output_dir, "histogram_comparison.png")
                plt.savefig(path, bbox_inches='tight', dpi=300)
                plot_paths['histogram'] = path
            if current_show_plots:
                plt.show()
            plt.close()
            
            # Add post-hoc test visualization if available
            if 'post_hoc' in test_results and hasattr(test_results['post_hoc']['results'], 'shape'):
                post_hoc_results = test_results['post_hoc']['results']
                
                if hasattr(post_hoc_results, 'columns') and 'p-adj' in post_hoc_results.columns:
                    # For Tukey's HSD results
                    plt.figure(figsize=(10, max(6, len(post_hoc_results) * 0.4)))
                    
                    # Create a table-like visualization
                    ax = plt.gca()
                    ax.axis('off')
                    
                    # Convert to a more friendly format for plotting
                    plot_data = post_hoc_results.copy()
                    plot_data['reject'] = plot_data['reject'].map({True: 'Yes', False: 'No'})
                    plot_data['significant'] = plot_data['p-adj'] < alpha
                    
                    # Sort by p-value
                    plot_data = plot_data.sort_values('p-adj')
                    
                    # Create a table
                    table = ax.table(
                        cellText=plot_data.values,
                        colLabels=plot_data.columns,
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0, 1, 1]
                    )
                    
                    # Style the table
                    table.auto_set_font_size(False)
                    table.set_fontsize(10)
                    table.scale(1, 1.5)
                    
                    # Color significant rows
                    for i, row in enumerate(plot_data.itertuples()):
                        if row.significant:
                            for j in range(len(plot_data.columns)):
                                table[(i+1, j)].set_facecolor('lightgreen')
                    
                    plt.title(f"Post-hoc Test: {test_results['post_hoc']['test']}", fontsize=14, pad=20)
                    plt.tight_layout()
                    
                    if current_output_dir:
                        path = os.path.join(current_output_dir, "post_hoc_test_results.png")
                        plt.savefig(path, bbox_inches='tight', dpi=300)
                        plot_paths['post_hoc'] = path
                    if current_show_plots:
                        plt.show()
                    plt.close()
                
                elif isinstance(post_hoc_results, pd.DataFrame) and post_hoc_results.shape[0] == post_hoc_results.shape[1]:
                    # For Dunn's test results (p-value matrix)
                    plt.figure(figsize=(10, 8))
                    
                    # Create a heatmap of p-values
                    sns.heatmap(
                        post_hoc_results,
                        annot=True,
                        cmap='YlGnBu',
                        fmt='.3f',
                        linewidths=0.5
                    )
                    
                    plt.title(f"Post-hoc Test: {test_results['post_hoc']['test']}", fontsize=14)
                    plt.tight_layout()
                    
                    if current_output_dir:
                        path = os.path.join(current_output_dir, "post_hoc_pvalue_matrix.png")
                        plt.savefig(path, bbox_inches='tight', dpi=300)
                        plot_paths['post_hoc_matrix'] = path
                    if current_show_plots:
                        plt.show()
                    plt.close()
            
            if plot_paths:
                results['plot_paths'] = plot_paths
        
        self._log(f"Group comparison completed. Test: {test_results.get('test', 'Unknown')}, p-value: {p_value:.4f}")
        return results


# --- Unit Test Class ---
class TestComprehensiveDataFrameAnalyzer(unittest.TestCase):
    def setUp(self):
        """Set up a sample DataFrame and analyzer instance for tests."""
        self.data = {
            "Numeric1": [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                100,
                np.nan,
            ],  # Includes an outlier and NaN
            "Numeric2": [10, 20, 30, 40, 50, 55, 45, 35, 25, 15, 20],  # Fairly normal
            "Numeric3": [
                1.1,
                1.2,
                1.15,
                1.22,
                1.18,
                1.25,
                1.13,
                1.21,
                1.16,
                1.19,
                1.2,
            ],  # Low variance
            "Correlated1": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
            "Correlated2": [
                1.9,
                4.1,
                5.8,
                8.2,
                9.9,
                12.1,
                13.8,
                16.2,
                18.1,
                19.8,
                22.2,
            ],  # Highly correlated with Correlated1
            "Category": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B", "A"],
            "ForFilter": [10, 20, 5, 30, 15, 25, 8, 22, 12, 18, 28],
        }
        self.sample_df = pd.DataFrame(self.data)
        self.analyzer = ComprehensiveDataFrameAnalyzer(
            show_plots_default=False
        )  # Disable plots by default for tests
        self.temp_output_dir = (
            tempfile.mkdtemp()
        )  # Create a temporary directory for plot path testing

    def tearDown(self):
        """Remove the temporary directory after tests."""
        shutil.rmtree(self.temp_output_dir)

    def test_perform_descriptive_analysis(self):
        """Test descriptive statistics calculation and optional normality tests."""
        results = self.analyzer.perform_descriptive_analysis(
            self.sample_df, columns=["Numeric1", "Numeric2"]
        )
        self.assertIn("statistics_summary", results)
        self.assertIsInstance(results["statistics_summary"], pd.DataFrame)
        self.assertEqual(
            len(results["statistics_summary"]), 2
        )  # For Numeric1, Numeric2
        self.assertAlmostEqual(
            results["statistics_summary"].loc["Numeric1", "mean"],
            self.sample_df["Numeric1"].mean(),
        )

        # Test with normality test
        results_with_normality = self.analyzer.perform_descriptive_analysis(
            self.sample_df, columns=["Numeric2"], test_normality_alpha=0.05
        )
        self.assertIn("normality_tests", results_with_normality)
        self.assertIsInstance(results_with_normality["normality_tests"], pd.DataFrame)
        self.assertTrue(
            results_with_normality["normality_tests"].loc["Numeric2", "is_normal"]
            in [True, False, np.nan]
        )

        # Test plot path generation
        results_with_plots = self.analyzer.perform_descriptive_analysis(
            self.sample_df,
            columns=["Numeric1"],
            generate_plots=True,
            output_dir=self.temp_output_dir,
            plot_types=["boxplot"],
        )
        self.assertIn("plot_paths", results_with_plots)
        self.assertIn("Numeric1", results_with_plots["plot_paths"])
        self.assertIn("boxplot", results_with_plots["plot_paths"]["Numeric1"])
        self.assertTrue(
            os.path.exists(results_with_plots["plot_paths"]["Numeric1"]["boxplot"])
        )

        # Test with no numerical columns
        df_no_numeric = pd.DataFrame({"Category": ["X", "Y"]})
        results_no_numeric = self.analyzer.perform_descriptive_analysis(df_no_numeric)
        self.assertIn("error", results_no_numeric)

    def test_analyze_column_correlations(self):
        """Test correlation matrix and identification of highly correlated pairs."""
        results = self.analyzer.analyze_column_correlations(
            self.sample_df,
            columns=["Correlated1", "Correlated2", "Numeric1"],
            correlation_threshold=0.9,
        )
        self.assertIn("correlation_matrix", results)
        self.assertIsInstance(results["correlation_matrix"], pd.DataFrame)
        self.assertEqual(results["correlation_matrix"].shape, (3, 3))
        self.assertAlmostEqual(
            results["correlation_matrix"].loc["Correlated1", "Correlated2"],
            self.sample_df[["Correlated1", "Correlated2"]].corr().iloc[0, 1],
        )

        self.assertIn("highly_correlated_pairs", results)
        self.assertIsInstance(results["highly_correlated_pairs"], pd.DataFrame)
        self.assertEqual(
            len(results["highly_correlated_pairs"]), 1
        )  # Correlated1 and Correlated2
        self.assertEqual(
            results["highly_correlated_pairs"].iloc[0]["Variable1"], "Correlated1"
        )
        self.assertEqual(
            results["highly_correlated_pairs"].iloc[0]["Variable2"], "Correlated2"
        )

        # Test heatmap generation path
        results_with_heatmap = self.analyzer.analyze_column_correlations(
            self.sample_df,
            columns=["Correlated1", "Correlated2"],
            generate_heatmap=True,
            output_dir=self.temp_output_dir,
        )
        self.assertIn("heatmap_path", results_with_heatmap)
        self.assertTrue(os.path.exists(results_with_heatmap["heatmap_path"]))

    def test_investigate_value_distribution(self):
        """Test distribution analysis for a single column."""
        results = self.analyzer.investigate_value_distribution(
            self.sample_df, "Numeric2", distributions_to_test=["norm", "expon"]
        )
        self.assertIn("column_name", results)
        self.assertEqual(results["column_name"], "Numeric2")
        self.assertIn("moments", results)
        self.assertAlmostEqual(
            results["moments"]["mean"], self.sample_df["Numeric2"].mean()
        )
        self.assertIn("distribution_fit_tests", results)
        self.assertIn("norm", results["distribution_fit_tests"])
        self.assertTrue(
            results["distribution_fit_tests"]["norm"]["fits"] in [True, False]
        )

        # Test plot path generation
        results_with_plots = self.analyzer.investigate_value_distribution(
            self.sample_df,
            "Numeric2",
            generate_plots=True,
            output_dir=self.temp_output_dir,
        )
        self.assertIn("plot_paths", results_with_plots)
        self.assertIn("qq_plot", results_with_plots["plot_paths"])
        self.assertTrue(os.path.exists(results_with_plots["plot_paths"]["qq_plot"]))
        self.assertIn("distribution_comparison_plot", results_with_plots["plot_paths"])
        self.assertTrue(
            os.path.exists(
                results_with_plots["plot_paths"]["distribution_comparison_plot"]
            )
        )

        # Test with insufficient data
        short_df = pd.DataFrame({"ShortNum": [1, 2, 3]})
        results_short = self.analyzer.investigate_value_distribution(
            short_df, "ShortNum"
        )
        self.assertIn("error", results_short)

    def test_detect_outliers_with_iqr(self):
        """Test IQR outlier detection."""
        results = self.analyzer.detect_outliers_with_iqr(
            self.sample_df, columns=["Numeric1"], k_multiplier=1.0
        )  # Lower k to catch 100
        self.assertIn("Numeric1", results)
        self.assertIn("outlier_indices", results["Numeric1"])
        self.assertIn(9, results["Numeric1"]["outlier_indices"])  # Index of 100
        self.assertIn(100, results["Numeric1"]["outlier_values"])
        # Check if 100 is outside the calculated bounds
        lower_b = results["Numeric1"]["bounds"][0]
        upper_b = results["Numeric1"]["bounds"][1]
        self.assertFalse(lower_b <= 100 <= upper_b)

        # Test plot path generation
        results_with_plot = self.analyzer.detect_outliers_with_iqr(
            self.sample_df,
            columns=["Numeric1"],
            generate_plots=True,
            output_dir=self.temp_output_dir,
        )
        self.assertIn("plot_path", results_with_plot["Numeric1"])
        self.assertTrue(os.path.exists(results_with_plot["Numeric1"]["plot_path"]))

    def test_detect_outliers_with_zscore(self):
        """Test Z-score outlier detection."""
        results = self.analyzer.detect_outliers_with_zscore(
            self.sample_df, columns=["Numeric1"], threshold=2.0
        )  # Lower threshold
        self.assertIn("outliers_summary", results)
        outliers_df = results["outliers_summary"]
        self.assertIsInstance(outliers_df, pd.DataFrame)
        numeric1_outliers = outliers_df[outliers_df["column"] == "Numeric1"]
        self.assertTrue(any(numeric1_outliers["value"] == 100))
        self.assertTrue(
            np.abs(
                numeric1_outliers[numeric1_outliers["value"] == 100]["z_score"].iloc[0]
            )
            > 2.0
        )

        # Test plot path generation
        results_with_plots = self.analyzer.detect_outliers_with_zscore(
            self.sample_df,
            columns=["Numeric1"],
            generate_plots=True,
            output_dir=self.temp_output_dir,
        )
        self.assertIn("plot_paths", results_with_plots)
        self.assertIn("Numeric1", results_with_plots["plot_paths"])
        self.assertTrue(os.path.exists(results_with_plots["plot_paths"]["Numeric1"]))

    def test_apply_principal_component_analysis(self):
        """Test PCA functionality."""
        # Using columns that are somewhat correlated and have variance
        pca_cols = ["Numeric1", "Numeric2", "Correlated1", "Correlated2"]
        # Drop NaN for PCA test data as PCA itself handles it by dropping rows
        test_df_pca = self.sample_df[pca_cols].dropna()

        results = self.analyzer.apply_principal_component_analysis(
            test_df_pca, variance_threshold_for_auto_n=0.90
        )
        self.assertIn("pca_transformed_data", results)
        self.assertIsInstance(results["pca_transformed_data"], pd.DataFrame)
        self.assertIn("loadings", results)
        self.assertIsInstance(results["loadings"], pd.DataFrame)
        self.assertTrue(results["n_components_selected"] > 0)
        self.assertTrue(results["n_components_selected"] <= len(pca_cols))

        # Test with fixed n_components
        results_fixed_n = self.analyzer.apply_principal_component_analysis(
            test_df_pca, n_components=2
        )
        self.assertEqual(results_fixed_n["n_components_selected"], 2)
        self.assertEqual(results_fixed_n["pca_transformed_data"].shape[1], 2)

        # Test plot path generation
        results_with_plots = self.analyzer.apply_principal_component_analysis(
            test_df_pca, generate_plots=True, output_dir=self.temp_output_dir
        )
        self.assertIn("plot_paths", results_with_plots)
        self.assertIn("scree_plot", results_with_plots["plot_paths"])
        self.assertTrue(os.path.exists(results_with_plots["plot_paths"]["scree_plot"]))
        if results_with_plots["n_components_selected"] >= 2:
            self.assertIn(
                "pc1_vs_pc2_scatter", results_with_plots["plot_paths"]
            )  # Only if >=2 PCs
            self.assertTrue(
                os.path.exists(results_with_plots["plot_paths"]["pc1_vs_pc2_scatter"])
            )

    def test_filter_column_by_threshold(self):
        """Test threshold-based filtering for various filter types."""
        col_to_filter = "ForFilter"  # Range [5, 30]

        # Low-pass
        res_low = self.analyzer.filter_column_by_threshold(
            self.sample_df, col_to_filter, "low_pass", 15
        )
        self.assertLess(res_low["filtered_dataframe"][col_to_filter].max(), 16)
        self.assertEqual(
            res_low["filter_summary"]["filtered_count"],
            len(self.sample_df[self.sample_df[col_to_filter] <= 15]),
        )

        # High-pass
        res_high = self.analyzer.filter_column_by_threshold(
            self.sample_df, col_to_filter, "high_pass", 20
        )
        self.assertGreater(res_high["filtered_dataframe"][col_to_filter].min(), 19)
        self.assertEqual(
            res_high["filter_summary"]["filtered_count"],
            len(self.sample_df[self.sample_df[col_to_filter] >= 20]),
        )

        # Band-pass
        res_band = self.analyzer.filter_column_by_threshold(
            self.sample_df, col_to_filter, "band_pass", (10, 25)
        )
        self.assertTrue(
            all(
                (res_band["filtered_dataframe"][col_to_filter] >= 10)
                & (res_band["filtered_dataframe"][col_to_filter] <= 25)
            )
        )
        self.assertEqual(
            res_band["filter_summary"]["filtered_count"],
            len(
                self.sample_df[
                    (self.sample_df[col_to_filter] >= 10)
                    & (self.sample_df[col_to_filter] <= 25)
                ]
            ),
        )

        # Band-stop
        res_stop = self.analyzer.filter_column_by_threshold(
            self.sample_df, col_to_filter, "band_stop", (12, 22)
        )  # Remove 12-22 inclusive
        self.assertTrue(
            all(
                (res_stop["filtered_dataframe"][col_to_filter] < 12)
                | (res_stop["filtered_dataframe"][col_to_filter] > 22)
            )
        )
        self.assertEqual(
            res_stop["filter_summary"]["filtered_count"],
            len(
                self.sample_df[
                    (self.sample_df[col_to_filter] < 12)
                    | (self.sample_df[col_to_filter] > 22)
                ]
            ),
        )

        # Test plot path generation
        res_plot = self.analyzer.filter_column_by_threshold(
            self.sample_df,
            col_to_filter,
            "low_pass",
            15,
            generate_plot=True,
            output_dir=self.temp_output_dir,
        )
        self.assertIn("plot_path", res_plot)
        self.assertTrue(os.path.exists(res_plot["plot_path"]))

        # Test non-numeric column
        res_non_numeric = self.analyzer.filter_column_by_threshold(
            self.sample_df, "Category", "low_pass", 15
        )
        self.assertIn("error", res_non_numeric)
    
    def test_identify_skewed_columns(self):
        """Test identification of skewed columns."""
        # Create a DataFrame with known skewness
        skewed_df = pd.DataFrame({
            'normal': np.random.normal(0, 1, 100),  # Normal distribution
            'right_skewed': np.random.exponential(1, 100),  # Right skewed
            'left_skewed': -np.random.exponential(1, 100)  # Left skewed
        })
        
        # Test with default threshold
        results = self.analyzer.identify_skewed_columns(skewed_df)
        self.assertIsInstance(results, dict)
        
        # Test with custom threshold
        results_custom = self.analyzer.identify_skewed_columns(skewed_df, skewness_threshold=0.5)
        self.assertIsInstance(results_custom, dict)
        
        # The exponential distributions should be detected as skewed
        self.assertIn('right_skewed', results_custom)
        self.assertIn('left_skewed', results_custom)
        
        # Verify skewness values
        self.assertGreater(results_custom['right_skewed'], 0)  # Right skew is positive
        self.assertLess(results_custom['left_skewed'], 0)  # Left skew is negative
    
    def test_suggest_transformations(self):
        """Test suggestion of transformations for skewed data."""
        # Create a right-skewed column
        skewed_df = pd.DataFrame({
            'right_skewed': np.random.exponential(1, 100) + 1  # Add 1 to ensure all positive
        })
        
        results = self.analyzer.suggest_transformations(skewed_df, 'right_skewed')
        
        self.assertIn('original_skewness', results)
        self.assertIn('transformations', results)
        self.assertIn('best_transformation', results)
        self.assertIn('transformed_data', results)
        
        # Verify that the best transformation reduced skewness
        best_transform = results['best_transformation']
        if best_transform:
            original_skew = abs(results['original_skewness'])
            transformed_skew = abs(results['transformations'][best_transform])
            self.assertLess(transformed_skew, original_skew)
    
    def test_analyze_missing_values(self):
        """Test analysis of missing values."""
        # Create a DataFrame with missing values
        missing_df = pd.DataFrame({
            'complete': [1, 2, 3, 4, 5],
            'missing_20pct': [1, 2, np.nan, 4, 5],
            'missing_40pct': [1, np.nan, 3, np.nan, 5],
            'all_missing': [np.nan, np.nan, np.nan, np.nan, np.nan]
        })
        
        results = self.analyzer.analyze_missing_values(missing_df)
        
        self.assertIn('missing_by_column', results)
        self.assertIn('columns_percent_missing', results)
        self.assertIn('rows_with_any_missing', results)
        
        # Check specific counts
        self.assertEqual(results['missing_by_column']['missing_20pct'], 1)
        self.assertEqual(results['missing_by_column']['missing_40pct'], 2)
        self.assertEqual(results['missing_by_column']['all_missing'], 5)
        
        # Test with plot generation
        results_with_plots = self.analyzer.analyze_missing_values(
            missing_df,
            generate_plot=True,
            output_dir=self.temp_output_dir
        )
        
        if 'plot_paths' in results_with_plots:
            for path_key, path in results_with_plots['plot_paths'].items():
                self.assertTrue(os.path.exists(path))
    
    def test_compare_groups(self):
        """Test comparison of groups."""
        # Create a DataFrame with groups
        groups_df = pd.DataFrame({
            'value': [10, 12, 8, 15, 20, 25, 22, 28, 30, 35],
            'group': ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B']
        })
        
        # Test with t-test
        results_ttest = self.analyzer.compare_groups(
            groups_df, 
            numerical_column='value',
            groupby_column='group',
            test_type='ttest'
        )
        
        self.assertIn('group_stats', results_ttest)
        self.assertIn('test_results', results_ttest)
        self.assertIn('p_value', results_ttest)
        self.assertIn('significant', results_ttest)
        
        # Test with Mann-Whitney
        results_mw = self.analyzer.compare_groups(
            groups_df, 
            numerical_column='value',
            groupby_column='group',
            test_type='mannwhitney'
        )
        
        self.assertEqual(results_mw['test_type'], 'mannwhitney')
        
        # Test with auto test selection
        results_auto = self.analyzer.compare_groups(
            groups_df, 
            numerical_column='value',
            groupby_column='group',
            test_type='auto'
        )
        
        self.assertIn(results_auto['test_type'], ['ttest', 'mannwhitney', 'anova', 'kruskal'])
        
        # Test with plot generation
        results_with_plots = self.analyzer.compare_groups(
            groups_df, 
            numerical_column='value',
            groupby_column='group',
            generate_plots=True,
            output_dir=self.temp_output_dir
        )
        
        if 'plot_paths' in results_with_plots:
            for path_key, path in results_with_plots['plot_paths'].items():
                self.assertTrue(os.path.exists(path))


if __name__ == "__main__":
    # This setup allows running tests in environments like Jupyter notebooks
    # where unittest.main() might not work as expected by default.
    # It also ensures that plots are not displayed during automated test runs.

    # Create a dummy ComprehensiveDataFrameAnalyzer to ensure plt.show() is managed if tests are run directly
    # This is more of a safeguard if tests were to call plt.show() directly, though the class methods handle it.
    original_show = plt.show
    plt.show = lambda: None  # Suppress plt.show() during tests

    try:
        unittest.main(argv=["first-arg-is-ignored"], exit=False)
    finally:
        plt.show = original_show  # Restore original plt.show()

