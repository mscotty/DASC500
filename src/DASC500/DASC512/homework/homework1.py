"""
This script provides a complete solution to DASC 512 Homework 1.

It uses the DataAnalysis class to perform data loading, cleaning, analysis, 
and visualization for all three problems in the assignment.

To run this script:
1. Make sure you have the DASC500 package installed with the DataAnalysis class.
2. Place 'bgg2022.csv' and 'hofbatting.csv' in the same directory.
3. Run the script from your terminal: python solve_homework.py
"""
import os

import pandas as pd
import numpy as np

from DASC500.classes.DataAnalysis import DataAnalysis
from DASC500.plotting.scatter_plot import plot_scatter
from DASC500.utilities.get_top_level_module import get_project_root
from DASC500.utilities.print.redirect_print import redirect_print

#OUTPUT_FOLDER = r'D:\Mitchell\School\2025_Winter\DASC500\github\DASC500\outputs\DASC512\Homework1'
FOLDER = get_project_root()
OUTPUT_FOLDER = os.path.join(FOLDER, 'outputs/DASC512/Homework1')
DATA_FOLDER = os.path.join(FOLDER, 'data/DASC512/Homework/1')

redirect_print(os.path.join(OUTPUT_FOLDER, 'homework1_logger.txt'), also_to_stdout=True)

def solve_problem_1():
    """Solves all parts of Problem 1."""
    print("\n" + "="*50)
    print("Solving Problem 1: The Meaning of Life")
    print("="*50)

    # Manually create the DataFrame from the data provided in the PDF
    data = {
        "Topic": ["Family", "Friends", "Material Well-being", "Career", 
                  "Challenges", "Spirituality", "Society", "Health", "Hobbies"],
        "Frequency": [716, 300, 265, 262, 285, 250, 210, 172, 136]
    }
    df = pd.DataFrame(data)
    analysis = DataAnalysis(dataframe=df)
    
    # The survey was of 2,596 adults
    total_respondents = analysis.df['Frequency'].sum() 

    # (a) Compute the Relative Frequencies for each response category
    analysis.df['Relative Frequency'] = analysis.df['Frequency'] / total_respondents
    print("\n(a) Calculated Relative Frequencies:")
    print(analysis.df[['Topic', 'Relative Frequency']].round(4))

    # (b) Construct a bar graph of the Relative Frequencies
    print("\n(b) Generating bar graph of Relative Frequencies...")
    analysis.plot_bar_chart(
        x_column='Topic',
        y_column='Relative Frequency',
        title_name='What Aspects of Life are Meaningful?',
        x_axis_name='Response Category',
        y_axis_name='Relative Frequency',
        output_dir=OUTPUT_FOLDER,
        output_name='bar_plot_meaningful_aspects.png'
    )

    # (c) Does the data support the conclusion that over half of respondents chose one of the top three topics?
    print("\n(c) Checking if the top three topics account for over half of responses...")
    top_three_freq = analysis.df.nlargest(3, 'Frequency')['Frequency'].sum()
    proportion_top_three = top_three_freq / total_respondents
    
    is_over_half = proportion_top_three > 0.5
    
    print(f"The top three topics are: {list(analysis.df.nlargest(3, 'Frequency')['Topic'])}")
    print(f"Combined frequency of top three: {top_three_freq}")
    print(f"Total respondents: {total_respondents}")
    print(f"Proportion from top three: {proportion_top_three:.4f}")
    print(f"Conclusion: The data {'supports' if is_over_half else 'does not support'} the conclusion.")


def solve_problem_2():
    """Solves all parts of Problem 2."""
    
    print("\n" + "="*50)
    print("Solving Problem 2: Board Game Weights")
    print("="*50)

    # Load the dataset
    try:
        data_file = os.path.join(DATA_FOLDER, 'bgg2022.csv')
        bgg_df = pd.read_csv(data_file)
        # Per instructions, remove games where AveWeight is 0
        bgg_df = bgg_df[bgg_df['AveWeight'] > 0].copy()
        analysis = DataAnalysis(dataframe=bgg_df)
    except FileNotFoundError:
        print(f"Error: 'bgg2022.csv' not found at {data_file}. Please place it in the same directory.")
        return

    # (a) Create a single box plot for the review scores (AveRating)
    print("\n(a) Generating box plot for all game review scores...")
    analysis.plot_box(
        value_column='AveRating',
        title_name='Distribution of Board Game Review Scores',
        y_axis_name='Average User Rating (1-10)',
        orientation='horizontal',
        show_outliers=True,
        output_dir=OUTPUT_FOLDER,
        output_name='box_plot_board_game_scores.png'
    )

    # (b) Based on the box plot, is the distribution of all game review scores skewed?
    print("\n(b) Analysis of Review Score Skewness:")
    print("Based on the box plot from (a), the median is closer to the third quartile (Q3),")
    print("and the lower whisker is longer than the upper whisker. This indicates that the")
    print("distribution of game review scores is left-skewed (negatively skewed).")

    # (c) Create Box Plots for the average weight of games by Family Game rank
    # Add a new column to classify games
    analysis.df['IsFamilyGame'] = np.where(analysis.df['Family Game Rank'].notna(), 'Family Game', 'Non-Family Game')
    
    print("\n(c) Generating grouped box plots for game weights...")
    analysis.plot_box(
        value_column='AveWeight',
        group_column='IsFamilyGame',
        title_name='Game Weight by Family Game Classification',
        y_axis_name='Average Game Weight (1-5)',
        orientation='horizontal',
        show_outliers=True,
        output_dir=OUTPUT_FOLDER,
        output_name='box_plot_game_weight.png'
    )

    # (d) Approx. what percentage of Non-Family games are heavier than 75% of Family games?
    print("\n(d) Calculating percentage of Non-Family games heavier than 75% of Family games...")
    family_games = analysis.df[analysis.df['IsFamilyGame'] == 'Family Game']
    non_family_games = analysis.df[analysis.df['IsFamilyGame'] == 'Non-Family Game']
    
    # Find the 75th percentile (Q3) for Family game weights
    q3_family_weight = family_games['AveWeight'].quantile(0.75)
    
    # Find how many Non-Family games are heavier than this value
    heavier_non_family_count = (non_family_games['AveWeight'] > q3_family_weight).sum()
    
    # Calculate the percentage
    percentage = (heavier_non_family_count / len(non_family_games)) * 100
    
    print(f"The 75th percentile (Q3) weight for Family games is: {q3_family_weight:.2f}")
    print(f"Number of Non-Family games heavier than {q3_family_weight:.2f}: {heavier_non_family_count}")
    print(f"Total number of Non-Family games: {len(non_family_games)}")
    print(f"Result: Approximately {percentage:.2f}% of Non-Family games are heavier than 75% of Family games.")


def solve_problem_3():
    """Solves all parts of Problem 3."""
    def detect_custom_outliers_obp_slg(data, threshold_percentile=90):
        """
        Custom outlier detection for OBP vs SLG analysis.
        Identifies players with high OBP but unexpectedly low SLG.
        
        @param[in] data DataFrame containing OBP and SLG columns.
        @param[in] threshold_percentile Percentile threshold for OBP-SLG difference.
        @return Boolean mask indicating outlier rows.
        """
        # Calculate OBP - SLG difference
        obp_minus_slg = data['OBP'] - data['SLG']
        
        # Find outliers: high OBP-SLG difference (top percentile)
        threshold = obp_minus_slg.quantile(threshold_percentile / 100)
        outlier_mask = obp_minus_slg >= threshold
        
        return outlier_mask
    
    def detect_outliers_iqr(data, columns, threshold=1.5):
        """
        Detect outliers using the Interquartile Range (IQR) method.
        
        @param[in] data DataFrame containing the data.
        @param[in] columns List of column names to check for outliers.
        @param[in] threshold Multiplier for IQR (default 1.5).
        @return Boolean mask indicating outlier rows.
        """
        outlier_mask = np.zeros(len(data), dtype=bool)
        
        for col in columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outlier_mask |= (data[col] < lower_bound) | (data[col] > upper_bound)
        
        return outlier_mask


    def detect_outliers_zscore(data, columns, threshold=2.0):
        """
        Detect outliers using the Z-score method.
        
        @param[in] data DataFrame containing the data.
        @param[in] columns List of column names to check for outliers.
        @param[in] threshold Z-score threshold (default 2.0).
        @return Boolean mask indicating outlier rows.
        """
        from scipy.stats import zscore
        outlier_mask = np.zeros(len(data), dtype=bool)
        
        for col in columns:
            z_scores = np.abs(zscore(data[col].dropna()))
            # Handle case where column has NaN values
            col_outliers = np.zeros(len(data), dtype=bool)
            col_outliers[data[col].notna()] = z_scores > threshold
            outlier_mask |= col_outliers
        
        return outlier_mask
    
    def combine_outliers_union(*outlier_masks):
        """
        Combine multiple outlier detection results using UNION (OR operation).
        A point is considered an outlier if ANY method identifies it as an outlier.
        
        @param[in] *outlier_masks Variable number of boolean masks from outlier detection methods.
        @return Combined boolean mask.
        """
        combined_mask = np.zeros(len(outlier_masks[0]), dtype=bool)
        for mask in outlier_masks:
            combined_mask |= mask
        return combined_mask
    
    print("\n" + "="*50)
    print("Solving Problem 3: Baseball Hall-of-Famers")
    print("="*50)

    # Load the dataset
    try:
        data_file = os.path.join(DATA_FOLDER, 'hofbatting.csv')
        hof_df = pd.read_csv(data_file)
        analysis = DataAnalysis(dataframe=hof_df)
    except FileNotFoundError:
        print(f"Error: 'hofbatting.csv' not found at {data_file}. Please place it in the same directory.")
        return

    # Define eras based on the homework
    era_bins = [0, 1900, 1919, 1941, 1960, 1976, 1993, 2005, 2100]
    era_labels = ["19th Century", "Dead Ball", "Live Ball", "Integration", 
                  "Expansion", "Free Agency", "Steroid Era", "Post-Steroid Era"]

    # Calculate Mid-Career Year and classify into eras
    analysis.df['midCareer'] = np.ceil((analysis.df['From'] + analysis.df['To']) / 2)
    analysis.categorize_by_bin(
        source_col='midCareer', 
        new_col_name='Era', 
        bins=era_bins, 
        labels=era_labels
    )

    # (a) Create a Bar Graph for the number of HoFs from each era
    print("\n(a) Generating bar graph of Hall of Famers by era...")
    era_counts = analysis.df['Era'].value_counts().reset_index()
    era_counts.columns = ['Era', 'Count']
    
    # Create a temporary DataAnalysis object for the counts
    counts_analysis = DataAnalysis(dataframe=era_counts)
    counts_analysis.plot_bar_chart(
        x_column='Era',
        y_column='Count',
        title_name='Baseball Hall of Famers by Era',
        x_axis_name='Era',
        y_axis_name='Number of Hall of Famers',
        output_dir=OUTPUT_FOLDER,
        output_name='bar_chart_baseball_hall_of_famers.png'
    )

    # (b) Which era has produced far more Hall-of-Famers than the rest?
    most_prolific_era = era_counts.loc[era_counts['Count'].idxmax(), 'Era']
    print(f"\n(b) The era that has produced the most Hall of Famers is the '{most_prolific_era}' era.")

    # (c) Create a histogram showing the distribution of Mid-Career year
    print("\n(c) Generating histogram of Mid-Career year by decade...")
    decade_bins = list(range(1880, 2021, 10))
    analysis.plot_histograms_per_col(
        key_in='midCareer',
        bin_edges=decade_bins,
        title_name='Distribution of Hall of Famer Mid-Career Year',
        x_axis_name='Mid-Career Year (by Decade)',
        output_dir=OUTPUT_FOLDER,
        output_name='histogram_distro_hall_of_famer_mid.png'
    )

    # (d) Create a scatterplot of OBP vs. SLG
    print("\n(d) Generating scatterplot of OBP vs. SLG...")
    analysis.plot_scatter(
        x_column='OBP',
        y_column='SLG',
        title_name='OBP vs. SLG for Baseball Hall of Famers',
        x_axis_name='On-Base Percentage (OBP)',
        y_axis_name='Slugging Percentage (SLG)',
        output_dir=OUTPUT_FOLDER,
        output_name='scatter_plot_obp_vs_slg.png'
    )

    # (e) In the scatterplot above, are there any outliers?
    print("\n(e) Analysis of Outliers in OBP vs. SLG plot:")
    # Find the player with high OBP and unexpectedly low SLG
    analysis.df['OBP_minus_SLG'] = analysis.df['OBP'] - analysis.df['SLG']
    bottom_right_outlier = analysis.df.loc[analysis.df['OBP_minus_SLG'].idxmax()]
    print(f"Yes, there are outliers. For example, the player in the bottom-right, with a much higher OBP")
    print(f"than expected for his SLG, is: {bottom_right_outlier['Name']}")
    
    print("\nDetecting outliers based on OBP-SLG difference with a 95% threshold...")
    custom_outliers = detect_custom_outliers_obp_slg(analysis.df, threshold_percentile=98)
    
    print("\nDetecting outliers based on IQR with a 1.5 threshold...")
    iqr_outliers = detect_outliers_iqr(analysis.df, ['OBP', 'SLG'], threshold=1.5)
    
    # Combine using union (most outliers)
    combined_outliers = combine_outliers_union(custom_outliers, iqr_outliers)
    print("\nGenerating scatterplot of OBP vs. SLG with outliers labeled...")
    analysis.plot_scatter(
        x_column='OBP',
        y_column='SLG',
        label_column='Name',
        label_points=combined_outliers,
        labeled_points_name='Outliers',
        title_name='OBP vs SLG Analysis',
        x_axis_name='On-Base Percentage (OBP)',
        y_axis_name='Slugging Percentage (SLG)',
        output_dir=OUTPUT_FOLDER,
        output_name='scatter_plot_obp_vs_slg_w_outliers.png'
    )

    # (f) Create a scatterplot with standardized OPS on the y-axis
    print("\n(f) Generating scatterplot of Standardized OPS vs. Mid-Career Year...")
    analysis.df['OPS'] = analysis.df['OBP'] + analysis.df['SLG']
    # Standardize the data using a z-score calculation
    analysis.df['OPS_zscore'] = (analysis.df['OPS'] - analysis.df['OPS'].mean()) / analysis.df['OPS'].std()
    analysis.plot_scatter(
        x_column='midCareer',
        y_column='OPS_zscore',
        title_name='Standardized OPS vs. Mid-Career Year',
        x_axis_name='Mid-Career Year',
        y_axis_name='On-base Plus Slugging (OPS) Z-Score',
        output_dir=OUTPUT_FOLDER,
        output_name='scatter_plot_ops_standard_vs_mid.png'
    )

    # (g) Which Hall-of-Famers have an absolute standardized value greater than 3?
    print("\n(g) Identifying Hall of Famers with standardized OPS greater than 3...")
    ops_outliers = analysis.filter_by_threshold('OPS_zscore', 3, comparison='absolute')
    print("The following players have an absolute standardized OPS value greater than 3:")
    for name in ops_outliers['Name']:
        print(f"- {name}")
    analysis.plot_scatter(
        x_column='midCareer',
        y_column='OPS_zscore',
        label_column='Name',
        label_points=ops_outliers,
        labeled_points_name='Outliers',
        title_name='Standardized OPS vs. Mid-Career Year',
        x_axis_name='Mid-Career Year',
        y_axis_name='On-base Plus Slugging (OPS) Z-Score',
        output_dir=OUTPUT_FOLDER,
        output_name='scatter_plot_ops_standard_vs_mid_w_outliers.png'
    )


if __name__ == '__main__':
    solve_problem_1()
    solve_problem_2()
    solve_problem_3()
