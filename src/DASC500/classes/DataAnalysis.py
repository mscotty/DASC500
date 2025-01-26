import os

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from DASC500.utilities.data_type.distinguish_data_types import distinguish_data_types
from DASC500.utilities.print.print_series_mode import print_series_mode
from DASC500.plotting.plot_histogram import plot_histogram
from DASC500.plotting.plot_stacked_bar_chart import plot_stacked_bar_chart_horizontal, plot_stacked_bar_chart_vertical
from DASC500.plotting.plot_clustered_bar_chart import plot_clustered_bar_chart_horizontal, plot_clustered_bar_chart_vertical
from DASC500.plotting.plot_individual_bar_charts import plot_individual_bar_charts
from DASC500.plotting.plot_line_chart import plot_line_chart
from DASC500.plotting.plot_heatmap import plot_heatmap
from DASC500.plotting.plot_radar_chart import plot_radar_chart


# -------------------------------------
# DataAnalysis Class Definition
# -------------------------------------
class DataAnalysis:
    def __init__(self, file):
        """
        Initialize the DataAnalysis object by loading a CSV file 
        and calculating initial stats and numeric columns.
        """
        self.file = file
        self.df = pd.read_csv(file)
        self.determine_numeric_col()
        self.calculate_stats()
    
    def determine_numeric_col(self):
        """
        Identify numeric columns in the DataFrame and store them.
        """
        self.col_types = distinguish_data_types(self.df)
        col_types = np.array(list(self.col_types.values()))
        headers = np.array(self.df.columns)
        num_headers = headers[col_types == 'Numeric']
        self.num_headers = {header: {} for header in num_headers}
    
    def calculate_stats(self):
        """
        Calculate and store statistics (mean, median, variance, etc.)
        for numeric columns in the DataFrame.
        """
        for key, value in self.num_headers.items():
            value['mean'] = self.df[key].mean()
            value['median'] = self.df[key].median()
            value['mode'] = self.df[key].mode(dropna=True)
            value['pop_variance'] = self.df[key].var(ddof=0)
            value['pop_std'] = self.df[key].std(ddof=0)
            value['sample_variance'] = self.df[key].var()
            value['sample_std'] = self.df[key].std()
            value['first_quartile'] = self.df[key].quantile(0.25)
            value['third_quartile'] = self.df[key].quantile(0.75)
    
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
                with open(file, 'a+') as f:
                    f.write(stats_string)
            
    
    def calculate_pearson_corr_coeff(self, 
                                     col1_name, 
                                     col2_name):
        """!
        @brief Calculate the Pearson correlation coefficient between two columns.
        Args:
        - col1_name (str): First column name.
        - col2_name (str): Second column name.
        """
        return self.df[col1_name].corr(self.df[col2_name])

    def plot_histograms(self, 
                        **kwargs):
        """!
        @brief Create and save histograms for numeric columns using the specified binning method.
        Args:
        - kwargs: Optional arguments for binning method, output directory, or bin width/count.
        """
        for key in self.num_headers.keys():
            data = self.df[key].dropna()
            plot_histogram(data,
                           **kwargs)
    
    def plot_stacked_bar_chart_horizontal(self, 
                                          **kwargs):
        plot_stacked_bar_chart_horizontal(self.df, 
                                          **kwargs)
    
    def plot_stacked_bar_chart_vertical(self, 
                                        **kwargs):
        plot_stacked_bar_chart_vertical(self.df, 
                                        **kwargs)
    
    def plot_clustered_bar_chart_horizontal(self, 
                                            **kwargs):
        plot_clustered_bar_chart_horizontal(self.df, 
                                            **kwargs)
    
    def plot_clustered_bar_chart_vertical(self, 
                                          **kwargs):
        plot_clustered_bar_chart_vertical(self.df, 
                                          **kwargs)
    
    def plot_individual_bar_charts(self, 
                               **kwargs):
        plot_individual_bar_charts(self.df, 
                                   **kwargs)
    
    def plot_line_chart(self, 
                        **kwargs):
        plot_line_chart(self.df, 
                        **kwargs)
    
    def plot_heatmap(self, 
                     **kwargs):
        plot_heatmap(self.df,
                     **kwargs)
    
    def plot_radar_chart(self,
                         **kwargs):
        plot_radar_chart(self.df,
                         **kwargs)
