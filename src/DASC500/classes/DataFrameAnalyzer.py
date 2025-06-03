import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as scipy_stats  # Renamed to avoid conflict with DataFrame.stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
import itertools  # For correlation pair identification
import unittest  # Added for unit testing
import tempfile  # For creating temporary directories for plot path testing
import shutil  # For removing temporary directories

from DASC500.stats import *


class ComprehensiveDataFrameAnalyzer:
    """
    A class for performing various comprehensive analyses on pandas DataFrames.
    It provides methods for descriptive statistics, correlation analysis,
    distribution investigation, IQR and Z-score outlier detection,
    Principal Component Analysis (PCA), and threshold-based filtering.
    """

    def __init__(self, show_plots_default: bool = True, output_dir_default: str = None):
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

    def _prepare_output_dir(
        self, method_output_dir: str = None, specific_subdir: str = None
    ) -> str:
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

    # Method 1: Descriptive Analysis
    def perform_descriptive_analysis(
        self,
        dataframe: pd.DataFrame,
        columns: list = None,
        test_normality_alpha: float = 0.05,
        generate_plots: bool = False,
        plot_types: list = None,  # ['boxplot', 'histogram', 'violin']
        output_dir: str = None,
        show_plots: bool = None,
    ) -> dict:
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
        """
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

        # Compute advanced statistics
        desc_stats = num_df_selected.describe().T
        desc_stats["skewness"] = num_df_selected.skew()
        desc_stats["kurtosis"] = num_df_selected.kurtosis()
        desc_stats["median"] = num_df_selected.median()
        # Pandas Series.mad is deprecated since version 1.5.0 and will be removed in a future version.
        # Replace with Series.abs().mean() if needed, or remove if not critical.
        # For now, let's use a robust MAD calculation if scipy is available.
        try:
            desc_stats["mad"] = num_df_selected.apply(
                lambda x: scipy_stats.median_abs_deviation(x.dropna(), scale="normal"),
                axis=0,
            )
        except (
            AttributeError
        ):  # Fallback for older scipy or if median_abs_deviation is not preferred
            desc_stats["mad"] = (
                num_df_selected - num_df_selected.median()
            ).abs().median() * 1.4826

        desc_stats["sum"] = num_df_selected.sum()
        desc_stats["variance"] = num_df_selected.var()
        for q_val in [0.01, 0.05, 0.10, 0.90, 0.95, 0.99]:
            desc_stats[f"{q_val*100:.0f}%"] = num_df_selected.quantile(q_val)
        desc_stats["iqr"] = desc_stats["75%"] - desc_stats["25%"]
        desc_stats["range"] = desc_stats["max"] - desc_stats["min"]
        # Handle potential division by zero for cv if mean is zero
        desc_stats["cv"] = np.where(
            desc_stats["mean"] == 0, np.nan, desc_stats["std"] / desc_stats["mean"]
        )
        desc_stats["missing_count"] = num_df_selected.isnull().sum()
        desc_stats["missing_percent"] = (
            num_df_selected.isnull().sum() / len(num_df_selected)
        ) * 100
        results["statistics_summary"] = desc_stats

        # Test for normality
        if test_normality_alpha is not None:
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
                        fig.savefig(path)
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
                        fig.savefig(path)
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
                        fig.savefig(path)
                        col_plot_paths["violin"] = path
                    if current_show_plots:
                        plt.show()
                    plt.close(fig)
                if col_plot_paths:
                    plot_paths[col] = col_plot_paths
            if plot_paths:
                results["plot_paths"] = plot_paths
        return results

    # Method 2: Correlation Analysis
    def analyze_column_correlations(
        self,
        dataframe: pd.DataFrame,
        columns: list = None,
        method: str = "pearson",
        correlation_threshold: float = 0.7,
        generate_heatmap: bool = False,
        output_dir: str = None,
        show_plots: bool = None,
    ) -> dict:
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
        """
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

        corr_matrix = num_df.corr(method=method)
        results = {"correlation_matrix": corr_matrix}

        # Identify highly correlated pairs
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
        )  # Ensure boolean mask
        highly_correlated = []
        for col_idx, col_name in enumerate(upper_tri.columns):
            for row_idx, row_name in enumerate(upper_tri.index):
                if row_idx < col_idx:  # Ensure we are in the upper triangle
                    value = upper_tri.iloc[row_idx, col_idx]
                    if pd.notna(value) and abs(value) >= correlation_threshold:
                        highly_correlated.append(
                            {
                                "Variable1": row_name,
                                "Variable2": col_name,
                                "Correlation": value,
                            }
                        )

        results["highly_correlated_pairs"] = pd.DataFrame(highly_correlated)

        if generate_heatmap:
            current_show_plots = (
                show_plots if show_plots is not None else self.show_plots_default
            )
            current_output_dir = self._prepare_output_dir(
                output_dir, "correlation_plots"
            )

            fig, ax = plt.subplots(
                figsize=(
                    max(8, corr_matrix.shape[1] * 0.6),
                    max(6, corr_matrix.shape[0] * 0.6),
                )
            )
            sns.heatmap(
                corr_matrix,
                annot=True,
                cmap="coolwarm",
                fmt=".2f",
                linewidths=0.5,
                ax=ax,
            )
            ax.set_title(f"{method.capitalize()} Correlation Matrix")
            if current_output_dir:
                path = os.path.join(
                    current_output_dir, f"{method}_correlation_heatmap.png"
                )
                fig.savefig(path)
                results["heatmap_path"] = path
            if current_show_plots:
                plt.show()
            plt.close(fig)
        return results

    # Method 3: Distribution Investigation
    def investigate_value_distribution(
        self,
        dataframe: pd.DataFrame,
        column_name: str,
        distributions_to_test: list = None,  # ['norm', 'expon', 'uniform', 'lognorm']
        alpha: float = 0.05,
        generate_plots: bool = False,
        output_dir: str = None,
        show_plots: bool = None,
    ) -> dict:
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
        """
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
        results["moments"] = {
            "mean": np.mean(data),
            "variance": np.var(data),
            "skewness": scipy_stats.skew(data),
            "kurtosis": scipy_stats.kurtosis(data),  # Fisher's definition (normal=0)
        }

        # Test distribution fits
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
            # scipy.probplot doesn't directly support 'uniform' in the same way as others for dist argument
            # It's better to use 'norm' as a general reference or a specific implemented distribution

            fig, ax = plt.subplots()
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
                fig.savefig(path)
                plot_paths["qq_plot"] = path
            if current_show_plots:
                plt.show()
            plt.close(fig)

            # Histogram with fitted distributions
            fig, ax = plt.subplots()
            sns.histplot(
                data,
                kde=False,
                stat="density",
                label="Data Histogram",
                ax=ax,
                bins="auto",
            )
            x_plot = np.linspace(data.min(), data.max(), 200)
            for (
                dist_name_plot
            ) in distributions_to_test:  # Iterate through original list for plotting
                if "error" not in fit_tests.get(
                    dist_name_plot, {}
                ):  # Check if fit was successful
                    try:
                        dist_plot = getattr(scipy_stats, dist_name_plot)
                        if dist_name_plot == "uniform":
                            params_plot = (data.min(), data.max() - data.min())
                        else:
                            params_plot = dist_plot.fit(data)
                        pdf = dist_plot.pdf(x_plot, *params_plot)
                        ax.plot(x_plot, pdf, label=f"{dist_name_plot.capitalize()} fit")
                    except (
                        Exception
                    ):  # Catch any error during plotting this specific dist
                        pass
            ax.set_title(f"Distribution Fits for {column_name}")
            ax.legend()
            if current_output_dir:
                path = os.path.join(
                    current_output_dir,
                    f'{column_name.replace("/", "_").replace(" ", "_")}_distribution_comparison.png',
                )
                fig.savefig(path)
                plot_paths["distribution_comparison_plot"] = path
            if current_show_plots:
                plt.show()
            plt.close(fig)

            if plot_paths:
                results["plot_paths"] = plot_paths
        return results

    # Method 4: IQR Outlier Detection
    def detect_outliers_with_iqr(
        self,
        dataframe: pd.DataFrame,
        columns: list = None,
        k_multiplier: float = 1.5,
        generate_plots: bool = False,
        output_dir: str = None,
        show_plots: bool = None,
    ) -> dict:
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
        """
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

            Q1 = data_col.quantile(0.25)
            Q3 = data_col.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - k_multiplier * IQR
            upper_bound = Q3 + k_multiplier * IQR

            outliers = data_col[(data_col < lower_bound) | (data_col > upper_bound)]
            all_outlier_info[col] = {
                "outlier_indices": outliers.index.tolist(),
                "outlier_values": outliers.tolist(),
                "bounds": (lower_bound, upper_bound),
                "Q1": Q1,
                "Q3": Q3,
                "IQR": IQR,
            }

            if generate_plots:
                fig, ax = plt.subplots()
                sns.boxplot(
                    x=data_col, ax=ax, whis=k_multiplier, orientation="vertical"
                )  # whis controls IQR multiplier for whiskers
                ax.set_title(f"IQR Outlier Detection for {col} (k={k_multiplier})")
                # Highlight bounds
                ax.axvline(
                    lower_bound,
                    color="r",
                    linestyle="--",
                    label=f"Lower: {lower_bound:.2f}",
                )
                ax.axvline(
                    upper_bound,
                    color="r",
                    linestyle="--",
                    label=f"Upper: {upper_bound:.2f}",
                )
                ax.legend()

                if current_output_dir_iqr:
                    path = os.path.join(
                        current_output_dir_iqr,
                        f'{col.replace("/", "_").replace(" ", "_")}_iqr_boxplot.png',
                    )
                    fig.savefig(path)
                    all_outlier_info[col]["plot_path"] = path
                if current_show_plots:
                    plt.show()
                plt.close(fig)
        return all_outlier_info

    # Method 5: Z-score Outlier Detection
    def detect_outliers_with_zscore(
        self,
        dataframe: pd.DataFrame,
        columns: list = None,
        threshold: float = 3.0,
        generate_plots: bool = False,
        output_dir: str = None,
        show_plots: bool = None,
    ) -> dict:
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
        """
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

        current_show_plots = (
            show_plots if show_plots is not None else self.show_plots_default
        )
        current_output_dir_zscore = self._prepare_output_dir(
            output_dir, "zscore_outlier_plots"
        )

        for col in columns_to_analyze:
            data_col = dataframe[
                col
            ].dropna()  # Ensure we work with non-NaN data for Z-score calculation
            if len(data_col) < 2:  # Need mean and std, and at least one non-outlier
                continue

            # Calculate Z-scores for the non-NaN data
            # Note: scipy_stats.zscore handles ddof=0 by default, like pandas .std()
            # If ddof=1 is desired (sample std), it needs to be calculated manually or ensure data_col.std(ddof=1)
            col_mean = data_col.mean()
            col_std = data_col.std()  # ddof=1 by default for pandas Series.std()

            if col_std == 0:  # Avoid division by zero if all values are the same
                z_scores_values = pd.Series(0, index=data_col.index)
            else:
                z_scores_values = (data_col - col_mean) / col_std

            # Identify outliers based on the threshold
            col_outliers_mask = np.abs(z_scores_values) > threshold
            col_outliers = data_col[col_outliers_mask]

            for index, value in col_outliers.items():
                all_outliers_list.append(
                    {
                        "column": col,
                        "index": index,  # Original index from the input dataframe
                        "value": value,
                        "z_score": z_scores_values.loc[
                            index
                        ],  # Z-score for this specific value
                    }
                )

            if generate_plots and len(data_col) > 0:
                fig, ax = plt.subplots()
                sns.histplot(
                    z_scores_values, kde=True, ax=ax
                )  # Plot Z-scores of non-NaN values
                ax.axvline(
                    threshold,
                    color="r",
                    linestyle="--",
                    label=f"Threshold ({threshold})",
                )
                ax.axvline(-threshold, color="r", linestyle="--")
                ax.set_title(f"Z-score Distribution for {col}")
                ax.set_xlabel("Z-score")
                ax.legend()
                if current_output_dir_zscore:
                    path = os.path.join(
                        current_output_dir_zscore,
                        f'{col.replace("/", "_").replace(" ", "_")}_zscore_distribution.png',
                    )
                    fig.savefig(path)
                    plot_paths[col] = path
                if current_show_plots:
                    plt.show()
                plt.close(fig)

        results = {"outliers_summary": pd.DataFrame(all_outliers_list)}
        if plot_paths:
            results["plot_paths"] = plot_paths
        return results

    # Method 6: Principal Component Analysis (PCA)
    def apply_principal_component_analysis(
        self,
        dataframe: pd.DataFrame,
        columns: list = None,
        n_components=None,
        variance_threshold_for_auto_n: float = 0.95,
        generate_plots: bool = False,
        output_dir: str = None,
        show_plots: bool = None,
    ) -> dict:
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
        """
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
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_for_pca)

        # Determine n_components for PCA
        actual_max_components = min(
            data_for_pca.shape[0], data_for_pca.shape[1]
        )  # Max possible components

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

        results = {
            "pca_transformed_data": pca_df,
            "explained_variance_ratio": explained_variance_ratio,
            "cumulative_explained_variance": cumulative_explained_variance,
            "loadings": loadings,
            "n_components_selected": n_components_selected,
        }

        if generate_plots:
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

            fig, ax = plt.subplots()
            ax.plot(range(1, len(full_evr) + 1), full_evr, "o-", label="Individual EVR")
            ax.plot(
                range(1, len(full_cumulative_evr) + 1),
                full_cumulative_evr,
                "s-",
                label="Cumulative EVR",
            )
            ax.set_xlabel("Principal Component")
            ax.set_ylabel("Explained Variance Ratio")
            ax.set_title("Scree Plot")
            ax.axhline(
                y=variance_threshold_for_auto_n,
                color="r",
                linestyle="--",
                label=f"{variance_threshold_for_auto_n*100:.0f}% Threshold",
            )
            ax.axvline(
                x=n_components_selected,
                color="g",
                linestyle=":",
                label=f"Selected: {n_components_selected} Components",
            )
            ax.legend()
            if current_output_dir_pca:
                path = os.path.join(current_output_dir_pca, "pca_scree_plot.png")
                fig.savefig(path)
                plot_paths["scree_plot"] = path
            if current_show_plots:
                plt.show()
            plt.close(fig)

            # Loadings Heatmap
            if not loadings.empty:
                fig_load, ax_load = plt.subplots(
                    figsize=(
                        max(8, loadings.shape[1] * 0.8),
                        max(6, loadings.shape[0] * 0.4),
                    )
                )
                sns.heatmap(loadings, annot=True, cmap="viridis", fmt=".2f", ax=ax_load)
                ax_load.set_title("PCA Component Loadings")
                plt.tight_layout()  # Adjust layout to prevent labels from being cut off
                if current_output_dir_pca:
                    path = os.path.join(
                        current_output_dir_pca, "pca_loadings_heatmap.png"
                    )
                    fig_load.savefig(path, bbox_inches="tight")
                    plot_paths["loadings_heatmap"] = path
                if current_show_plots:
                    plt.show()
                plt.close(fig_load)

            # Scatter plot of PC1 vs PC2
            if n_components_selected >= 2:
                fig, ax = plt.subplots()
                ax.scatter(pca_df["PC1"], pca_df["PC2"], alpha=0.7)
                ax.set_xlabel(f"PC1 ({explained_variance_ratio[0]*100:.2f}%)")
                ax.set_ylabel(f"PC2 ({explained_variance_ratio[1]*100:.2f}%)")
                ax.set_title("PC1 vs PC2")
                if current_output_dir_pca:
                    path = os.path.join(
                        current_output_dir_pca, "pca_pc1_vs_pc2_scatter.png"
                    )
                    fig.savefig(path)
                    plot_paths["pc1_vs_pc2_scatter"] = path
                if current_show_plots:
                    plt.show()
                plt.close(fig)

            if plot_paths:
                results["plot_paths"] = plot_paths
        return results

    # Method 7: Threshold-based Filtering
    def filter_column_by_threshold(
        self,
        dataframe: pd.DataFrame,
        column_name: str,
        filter_type: str,
        thresholds,  # float for low/high, tuple for band
        inclusive: bool = True,
        generate_plot: bool = False,
        output_dir: str = None,
        show_plots: bool = None,
    ) -> dict:
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
        """
        if column_name not in dataframe.columns:
            return {"error": f"Column '{column_name}' not found in DataFrame."}
        if not pd.api.types.is_numeric_dtype(dataframe[column_name]):
            return {
                "error": f"Column '{column_name}' is not numerical and cannot be filtered by threshold."
            }

        original_df = dataframe.copy()  # Work on a copy
        data_col = original_df[column_name]
        # Initialize mask that keeps NaNs by default, then update based on filter
        # This ensures NaNs are not inadvertently dropped by boolean logic unless they fail the condition
        mask = pd.Series(True, index=data_col.index)

        if filter_type == "low_pass":
            if not isinstance(thresholds, (int, float)):
                return {
                    "error": "Low-pass filter requires a single numerical threshold."
                }
            condition = data_col <= thresholds if inclusive else data_col < thresholds
        elif filter_type == "high_pass":
            if not isinstance(thresholds, (int, float)):
                return {
                    "error": "High-pass filter requires a single numerical threshold."
                }
            condition = data_col >= thresholds if inclusive else data_col > thresholds
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
            else:
                condition = (data_col > low) & (data_col < high)
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
            else:  # Exclusive removal: remove if low < value < high
                condition = (data_col <= low) | (
                    data_col >= high
                )  # Keep if outside or equal to bounds
        else:
            return {
                "error": "Invalid filter_type. Choose from 'low_pass', 'high_pass', 'band_pass', 'band_stop'."
            }

        # Apply the condition. Non-matching rows (and rows where data_col is NaN, making condition NaN) will be False.
        # We only want to filter non-NaN values. NaNs in the original column should be preserved unless filtered out by a condition they satisfy.
        # However, boolean indexing with NaNs in the mask will convert to False.
        # A more explicit way:
        final_mask = condition.copy()
        final_mask[data_col.isnull()] = True  # Keep rows where the target column is NaN

        filtered_dataframe = original_df[final_mask]

        summary = {
            "original_count": len(original_df),
            "filtered_count": len(filtered_dataframe),
            "removed_count": len(original_df) - len(filtered_dataframe),
        }
        results = {"filtered_dataframe": filtered_dataframe, "filter_summary": summary}

        if generate_plot:
            current_show_plots = (
                show_plots if show_plots is not None else self.show_plots_default
            )
            current_output_dir_filter = self._prepare_output_dir(
                output_dir,
                f"filter_plots/{column_name.replace('/', '_').replace(' ', '_')}",
            )

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            sns.histplot(
                original_df[column_name].dropna(),
                kde=False,
                ax=ax1,
                label="Original (non-NaN)",
                color="blue",
                alpha=0.6,
                bins=50,
            )
            ax1.set_title(f"Original Distribution of {column_name}")
            ax1.legend()
            sns.histplot(
                filtered_dataframe[column_name].dropna(),
                kde=False,
                ax=ax2,
                label="Filtered (non-NaN)",
                color="green",
                alpha=0.6,
                bins=50,
            )
            ax2.set_title(f"Filtered Distribution ({filter_type})")
            ax2.legend()

            # Add threshold lines
            if filter_type in ["low_pass", "high_pass"]:
                ax1.axvline(thresholds, color="r", linestyle="--")
                ax2.axvline(thresholds, color="r", linestyle="--")
            elif filter_type in ["band_pass", "band_stop"]:
                ax1.axvline(thresholds[0], color="r", linestyle="--")
                ax1.axvline(thresholds[1], color="r", linestyle="--")
                ax2.axvline(thresholds[0], color="r", linestyle="--")
                ax2.axvline(thresholds[1], color="r", linestyle="--")

            plt.tight_layout()
            if current_output_dir_filter:
                path = os.path.join(
                    current_output_dir_filter,
                    f'{column_name.replace("/", "_").replace(" ", "_")}_{filter_type}_comparison.png',
                )
                fig.savefig(path)
                results["plot_path"] = path
            if current_show_plots:
                plt.show()
            plt.close(fig)
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
