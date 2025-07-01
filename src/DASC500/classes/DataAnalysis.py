import os
from copy import deepcopy
import warnings
from typing import Dict, Union, List, Tuple, Optional

import pandas as pd
import numpy as np
import scipy.stats as stats


from DASC500.utilities.data_type.distinguish_data_types import distinguish_data_types
from DASC500.utilities.print.print_series_mode import print_series_mode
from DASC500.formulas.statistics.confidence_interval import (
    calculate_confidence_interval,
)
from DASC500.formulas.statistics.hypothesis_test import hypothesis_test

from DASC500.models.build_linear_regression_model import build_linear_regression_model
from DASC500.models.build_mult_linear_regression_model import build_multiple_linear_regression_model
from DASC500.models.build_parsimonious_regression_model import (
    build_stepwise_parsimonious_regression_model,
)

from DASC500.plotting.plot_histogram import plot_histogram
from DASC500.plotting.plot_stacked_bar_chart import (
    plot_stacked_bar_chart_horizontal,
    plot_stacked_bar_chart_vertical,
)
from DASC500.plotting.plot_clustered_bar_chart import (
    plot_clustered_bar_chart_horizontal,
    plot_clustered_bar_chart_vertical,
)
from DASC500.plotting.plot_individual_bar_charts import plot_individual_bar_charts
from DASC500.plotting.plot_line_chart import plot_line_chart
from DASC500.plotting.plot_heatmap import plot_heatmap
from DASC500.plotting.plot_radar_chart import plot_radar_chart
from DASC500.plotting.visualize_regression_models import visualize_regression_models
from DASC500.plotting.bar_graph import plot_bar_chart
from DASC500.plotting.box_plot import plot_box
from DASC500.plotting.scatter_plot import plot_scatter


# -------------------------------------
# DataAnalysis Class Definition
# -------------------------------------
class DataAnalysis:
    def __init__(self, file=None, dataframe=None):
        """
        Initialize the DataAnalysis object by loading a CSV file
        or using an existing DataFrame. Calculates initial stats
        and identifies column data types.
        
        Args:
            file (str, optional): Path to the CSV file.
            dataframe (pd.DataFrame, optional): Existing DataFrame.
        """
        self.file = file
        if file is not None:
            self.df_original = pd.read_csv(file)
        elif file is None and dataframe is not None:
            self.df_original = dataframe
        else:
            raise ValueError("Need a valid file or dataframe.")
        
        self.df_test = None
        self.df = self.df_original.copy()
        
        # Analyze data types and store them
        self.analyze_data_types()
        
        # Calculate statistics for numeric columns
        self.calculate_stats()
    
    def analyze_data_types(self):
        """
        Analyze and categorize the data types of all columns in the DataFrame.
        This enhances the previous determine_numeric_col functionality.
        """
        # Get detailed data types
        self.column_types = distinguish_data_types(self.df)
        
        # Store columns by type for easy access
        self.numeric_columns = [col for col, type in self.column_types.items() if type == 'Numeric']
        self.categorical_columns = [col for col, type in self.column_types.items() if type == 'Categorical']
        self.binary_columns = [col for col, type in self.column_types.items() if type == 'Binary']
        self.datetime_columns = [col for col, type in self.column_types.items() if type == 'DateTime']
        self.text_columns = [col for col, type in self.column_types.items() if type == 'Text']
        
        # For backward compatibility with existing code
        self.num_headers = {header: {} for header in self.numeric_columns}
        
        # Automatic type conversion for better analysis
        self._convert_column_types()
    
    def _convert_column_types(self):
        """
        Convert DataFrame columns to appropriate types based on the identified data types.
        This improves data processing and visualization capabilities.
        """
        # Convert numeric columns to float or int when possible
        for col in self.numeric_columns:
            try:
                # Check if all values are integers
                if all(pd.notna(val) and float(val).is_integer() for val in self.df[col].dropna()):
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce').astype('Int64')  # Int64 handles NaN
                else:
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            except:
                # If conversion fails, keep as is
                pass
        
        # Convert datetime columns
        for col in self.datetime_columns:
            try:
                self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
            except:
                pass
        
        # Convert binary columns to boolean where appropriate
        for col in self.binary_columns:
            try:
                # For text-based boolean values
                if self.df[col].dtype == 'object':
                    # Map common boolean representations to True/False
                    bool_map = {'true': True, 'false': False, 'yes': True, 'no': False, 
                                't': True, 'f': False, 'y': True, 'n': False,
                                '1': True, '0': False, 1: True, 0: False}
                    
                    self.df[col] = self.df[col].apply(
                        lambda x: bool_map.get(str(x).lower(), None) if pd.notna(x) else None
                    )
                # For numeric 0/1 values
                elif pd.api.types.is_numeric_dtype(self.df[col]):
                    self.df[col] = self.df[col].map({1: True, 0: False})
            except:
                pass
    
    def determine_numeric_col(self):
        """
        Legacy method for backward compatibility.
        Now just calls analyze_data_types.
        """
        self.analyze_data_types()
    
    def calculate_stats(self):
        """
        Calculate and store statistics (mean, median, variance, etc.)
        for numeric columns in the DataFrame.
        """
        for key in self.numeric_columns:
            self.num_headers[key] = {
                "mean": self.df[key].mean(),
                "median": self.df[key].median(),
                "mode": self.df[key].mode(dropna=True),
                "pop_variance": self.df[key].var(ddof=0),
                "pop_std": self.df[key].std(ddof=0),
                "sample_variance": self.df[key].var(),
                "sample_std": self.df[key].std(),
                "first_quartile": self.df[key].quantile(0.25),
                "third_quartile": self.df[key].quantile(0.75),
                "min": self.df[key].min(),
                "max": self.df[key].max(),
                "skewness": self.df[key].skew(),
                "kurtosis": self.df[key].kurt()
            }
    
    def get_columns_by_type(self, data_type: str) -> List[str]:
        """
        Get a list of column names that match the specified data type.
        
        Args:
            data_type: Type of columns to retrieve ('Numeric', 'Categorical', 
                       'Binary', 'DateTime', 'Text', or 'All')
                       
        Returns:
            List of column names matching the specified type
        """
        if data_type.lower() == 'all':
            return list(self.df.columns)
        
        return [col for col, type in self.column_types.items() 
                if type.lower() == data_type.lower()]
    
    def summarize_data(self) -> Dict[str, Dict]:
        """
        Generate a comprehensive summary of the DataFrame.
        
        Returns:
            Dictionary with summary statistics for each column type
        """
        summary = {
            'dataset_info': {
                'rows': len(self.df),
                'columns': len(self.df.columns),
                'memory_usage': self.df.memory_usage(deep=True).sum() / (1024 * 1024),  # in MB
                'missing_cells': self.df.isna().sum().sum(),
                'missing_percentage': (self.df.isna().sum().sum() / (len(self.df) * len(self.df.columns))) * 100
            },
            'column_types': {
                'numeric': len(self.numeric_columns),
                'categorical': len(self.categorical_columns),
                'binary': len(self.binary_columns),
                'datetime': len(self.datetime_columns),
                'text': len(self.text_columns)
            },
            'columns': {}
        }
        
        # Add column-specific summaries
        for col in self.df.columns:
            col_type = self.column_types[col]
            missing = self.df[col].isna().sum()
            missing_pct = (missing / len(self.df)) * 100
            
            col_summary = {
                'type': col_type,
                'missing': missing,
                'missing_percentage': missing_pct
            }
            
            # Add type-specific statistics
            if col_type == 'Numeric':
                col_summary.update({
                    'mean': self.df[col].mean(),
                    'median': self.df[col].median(),
                    'std': self.df[col].std(),
                    'min': self.df[col].min(),
                    'max': self.df[col].max(),
                    'unique_values': len(self.df[col].unique())
                })
            elif col_type in ['Categorical', 'Binary']:
                value_counts = self.df[col].value_counts().to_dict()
                col_summary.update({
                    'unique_values': len(value_counts),
                    'most_common': self.df[col].value_counts().index[0] if not self.df[col].value_counts().empty else None,
                    'distribution': value_counts
                })
            elif col_type == 'DateTime':
                col_summary.update({
                    'min_date': self.df[col].min(),
                    'max_date': self.df[col].max(),
                    'range_days': (self.df[col].max() - self.df[col].min()).days 
                        if not pd.isna(self.df[col].max()) and not pd.isna(self.df[col].min()) else None
                })
            
            summary['columns'][col] = col_summary
        
        return summary

    def print_stats(self, file=None):
        """!
        @brief Print or save statistics of numeric columns.

        Args:
        - file (str): File path to save stats. If None, prints to console.
        """
        for key, value in self.num_headers.items():
            # Build the string with all metrics
            stats_string = (
                f"Calculated metrics for {key}\n"
                f"Mean: {value['mean']}\n"
                f"Median: {value['median']}\n"
                f"Mode:\n{print_series_mode(value['mode'])}\n"
                f"Population Variance: {value['pop_variance']}\n"
                f"Population Standard Deviation: {value['pop_std']}\n"
                f"Sample Variance: {value['sample_variance']}\n"
                f"Sample Standard Deviation: {value['sample_std']}\n"
                f"First quartile: {value['first_quartile']}\n"
                f"Third quartile: {value['third_quartile']}\n"
            )

            # Print or write the string based on the `file` argument
            if file is None:
                print(stats_string)
            else:
                with open(file, "a+") as f:
                    f.write(stats_string)

    def calculate_pearson_corr_coeff(self, col1_name, col2_name):
        """!
        @brief Calculate the Pearson correlation coefficient between two columns.
        Args:
        - col1_name (str): First column name.
        - col2_name (str): Second column name.
        """
        return self.df[col1_name].corr(self.df[col2_name])

    def confidence_intervals(self, confidence=0.95):
        """
        Compute confidence intervals for the mean and variance of each numerical column in a Pandas DataFrame.

        Parameters:
            confidence (float): Confidence level (default 0.95 for 95%).

        Returns:
            dict: Confidence intervals for mean and variance of each column.
        """
        conf_interval = {}

        for column in self.df.select_dtypes(
            include=[np.number]
        ):  # Process only numerical columns
            result = calculate_confidence_interval(
                self.df[column], confidence=confidence
            )

            # Store results
            conf_interval[column] = result

        self.conf_interval = conf_interval

    def print_confidence_intervals(self, file=None, col_names=None):
        """!
        @brief Print or save confidence intervals of numeric columns.

        Args:
        - file (str): File path to save confidence intervals. If None, prints to console.
        """
        if col_names is None:
            col_names = self.df.select_dtypes(include=[np.number])

        for col, res in self.conf_interval.items():
            if col not in col_names:
                continue
            # Build the string with all metrics
            conf_string = (
                f"{col}:\n"
                f"Mean CI: {res['mean_CI']}\n"
                f"Variance CI: {res['variance_CI']}\n"
            )

            # Print or write the string based on the `file` argument
            if file is None:
                print(conf_string)
            else:
                with open(file, "a+") as f:
                    f.write(conf_string)

    def hypothesis_test(self, data_col_name, **kwargs):
        """
        Perform a hypothesis test on a specified column.

        Args:
            data_col_name (str): Name of the column to test.
            **kwargs: Additional arguments for the hypothesis test function.

        Returns:
            dict: Results of the hypothesis test.
        """
        return hypothesis_test(self.df[data_col_name], **kwargs)

    def build_linear_regression_model(self, *args):
        """
        Build a simple linear regression model using the DataFrame.

        Args:
            *args: Additional arguments for the regression model.
        """
        self.lin_reg_model = linear_regression_model(self.df, *args)

    def build_mult_linear_regression_model(self, *args, **kwargs):
        """
        Build a multiple linear regression model using the DataFrame.

        Args:
            *args: Positional arguments for the regression model.
            **kwargs: Keyword arguments for the regression model.
        """
        self.mult_lin_reg_model = multiple_linear_regression(self.df, *args, **kwargs)

    def build_stepwise_parsimonious_regression_model(self, *args, **kwargs):
        """
        Build a stepwise parsimonious regression model.

        Args:
            *args: Positional arguments for the regression model.
            **kwargs: Keyword arguments for the regression model.

        Returns:
            dict: Final model, variables used, and VIF values.
        """
        model, vars, vif = stepwise_parsimonious_regression(self.df, *args, **kwargs)
        self.parsimonious_model = {"final_model": model, "used_vars": vars, "vif": vif}

    def calculate_relative_frequency(self, category_col):
        """
        Calculates the relative frequency of each unique value in a column.

        Args:
            category_col (str): The name of the column with categorical data.

        Returns:
            pd.Series: A Series containing the relative frequencies, indexed by category.
        """
        if category_col not in self.df.columns:
            raise ValueError(f"Column '{category_col}' not found in the DataFrame.")
        
        return self.df[category_col].value_counts(normalize=True)

    def categorize_by_bin(self, source_col, new_col_name, bins, labels, right_inclusive=True, inplace=True):
        """
        Creates a new categorical column by binning a numerical column.

        Args:
            source_col (str): The name of the numerical column to bin.
            new_col_name (str): The name for the new categorical column.
            bins (list): A list of bin edges.
            labels (list): A list of labels for the bins. Must be one less than the number of bin edges.
            right_inclusive (bool, optional): Whether the bins should be right-inclusive. Defaults to True.
            inplace (bool, optional): Whether to modify the current DataFrame or return a new one. Defaults to True.

        Returns:
            DataAnalysis or pd.DataFrame: If inplace is True, returns self with modified df; 
                                        otherwise returns the modified DataFrame.
        """
        binned_data = pd.cut(
            self.df[source_col], 
            bins=bins, 
            labels=labels, 
            right=right_inclusive, 
            include_lowest=True
        )
        
        if inplace:
            self.df[new_col_name] = binned_data
            return self
        else:
            result_df = self.df.copy()
            result_df[new_col_name] = binned_data
            return result_df

    def filter_by_threshold(self, value_col, threshold, comparison='absolute', inplace=False):
        """
        Filters the DataFrame based on a value in a column exceeding a threshold.

        Args:
            value_col (str): The name of the column to check.
            threshold (int or float): The threshold value.
            comparison (str, optional): How to compare. Options: 'absolute', 'greater', 'less'. 
                                    Defaults to 'absolute'.
            inplace (bool, optional): Whether to modify the current DataFrame or return a new one.
                                    Defaults to False.

        Returns:
            DataAnalysis or pd.DataFrame: If inplace is True, returns self with modified df; 
                                        otherwise returns the filtered DataFrame.
        """
        if comparison == 'absolute':
            filtered_df = self.df[abs(self.df[value_col]) > threshold]
        elif comparison == 'greater':
            filtered_df = self.df[self.df[value_col] > threshold]
        elif comparison == 'less':
            filtered_df = self.df[self.df[value_col] < threshold]
        else:
            raise ValueError("Comparison must be one of 'absolute', 'greater', or 'less'.")
        
        if inplace:
            self.df = filtered_df
            return self
        else:
            return filtered_df


    def plot_histograms_per_col(self, key_in=None, **kwargs):
        """
        Create and save histograms for numeric columns.

        Args:
            key_in (list or str, optional): Column names to plot. If None, plots all numeric columns.
            **kwargs: Additional arguments for the histogram plotting function.
        """
        if key_in is None:
            key_in = self.num_headers.keys()
        if isinstance(key_in, str):
            key_in = [key_in]

        for key in key_in:
            data = self.df[key].dropna()
            plot_histogram(data, **kwargs)

    def plot_stacked_bar_chart_horizontal(self, **kwargs):
        """
        Plot a horizontal stacked bar chart.

        Args:
            **kwargs: Additional arguments for the plotting function.
        """
        plot_stacked_bar_chart_horizontal(self.df, **kwargs)

    def plot_stacked_bar_chart_vertical(self, **kwargs):
        """
        Plot a vertical stacked bar chart.

        Args:
            **kwargs: Additional arguments for the plotting function.
        """
        plot_stacked_bar_chart_vertical(self.df, **kwargs)

    def plot_clustered_bar_chart_horizontal(self, **kwargs):
        """
        Plot a horizontal clustered bar chart.

        Args:
            **kwargs: Additional arguments for the plotting function.
        """
        plot_clustered_bar_chart_horizontal(self.df, **kwargs)

    def plot_clustered_bar_chart_vertical(self, **kwargs):
        """
        Plot a vertical clustered bar chart.

        Args:
            **kwargs: Additional arguments for the plotting function.
        """
        plot_clustered_bar_chart_vertical(self.df, **kwargs)

    def plot_individual_bar_charts(self, **kwargs):
        """
        Plot individual bar charts for each column.

        Args:
            **kwargs: Additional arguments for the plotting function.
        """
        plot_individual_bar_charts(self.df, **kwargs)

    def plot_line_chart(self, **kwargs):
        """
        Plot a line chart using the DataFrame.

        Args:
            **kwargs: Additional arguments for the plotting function.
        """
        plot_line_chart(self.df, **kwargs)

    def plot_heatmap(self, **kwargs):
        """
        Plot a heatmap using the DataFrame.

        Args:
            **kwargs: Additional arguments for the plotting function.
        """
        plot_heatmap(self.df, **kwargs)

    def plot_radar_chart(self, **kwargs):
        """
        Plot a radar chart using the DataFrame.

        Args:
            **kwargs: Additional arguments for the plotting function.
        """
        plot_radar_chart(self.df, **kwargs)
    
    def plot_bar_chart(self, **kwargs):
        """
        Plot a bar chart using the DataFrame.

        Args:
            **kwargs: Additional arguments for the bar chart plotting function.
            
        Key Arguments:
            x_column (str): Column name for the x-axis (categories).
            y_column (str): Column name for the y-axis (values).
            output_dir (str, optional): Directory to save the plot as a PNG file.
            output_name (str, optional): Custom file name for the saved plot.
            orientation (str, optional): Bar orientation: 'v' for vertical, 'h' for horizontal.
            color (str, optional): Color for the bars.
            bar_width (float, optional): Width of the bars.
            title_name (str, optional): Custom title for the chart.
            show_values (bool, optional): Whether to display values on top of bars.
        """
        plot_bar_chart(self.df, **kwargs)

    def plot_box(self, **kwargs):
        """
        Plot a box plot of the specified data, optionally grouped by a categorical variable.

        Args:
            **kwargs: Additional arguments for the box plot plotting function.
            
        Key Arguments:
            value_column (str): Column name for the values to be represented in the box plot.
            group_column (str, optional): Optional column name for grouping the data.
            output_dir (str, optional): Directory to save the plot as a PNG file.
            output_name (str, optional): Custom file name for the saved plot.
            box_color (str, optional): Color for the box plot when not grouped.
            box_colors (list, optional): List of colors for the box plots when grouped.
            show_points (bool, optional): Whether to display individual data points.
            notched (bool, optional): Whether to display notched box plots.
        """
        plot_box(self.df, **kwargs)

    def plot_scatter(self, **kwargs):
        """
        Plot a scatter plot of the specified data columns.

        Args:
            **kwargs: Additional arguments for the scatter plot plotting function.
            
        Key Arguments:
            x_column (str): Column name for the x-axis.
            y_column (str): Column name for the y-axis.
            color_column (str, optional): Optional column name for coloring points.
            size_column (str, optional): Optional column name for sizing points.
            output_dir (str, optional): Directory to save the plot as a PNG file.
            output_name (str, optional): Custom file name for the saved plot.
            marker_color (str, optional): Color for the markers when not using color_column.
            marker_size (int, optional): Size of the markers when not using size_column.
            add_trendline (bool, optional): Whether to add a linear trendline.
            show_r2 (bool, optional): Whether to display R² value with the trendline.
        """
        plot_scatter(self.df, **kwargs)


    def vis_reg_models(self, *args, **kwargs):
        """
        Visualize regression models using the DataFrame.

        Args:
            *args: Positional arguments for the visualization function.
            **kwargs: Keyword arguments for the visualization function.
        """
        visualize_regression_models(self.df, *args, **kwargs)
