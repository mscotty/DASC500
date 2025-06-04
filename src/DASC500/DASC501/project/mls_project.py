# Import necessary libraries
import pandas as pd
import sqlite3 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Import custom classes
# Ensure these .py files are in the same directory as this notebook or in your PYTHONPATH
from DASC500.classes.DatabaseManager import DatabaseManager, FileType
from DASC500.classes.DataFrameAnalyzer import ComprehensiveDataFrameAnalyzer
# from DataAnalysis import DataAnalysis # We will primarily use ComprehensiveDataFrameAnalyzer and seaborn/matplotlib

from DASC500.utilities.get_top_level_module import get_top_level_module_path

FOLDER = os.path.join(get_top_level_module_path(), '../..')
INPUT_FOLDER = os.path.join(FOLDER, 'data/DASC501/project')
OUTPUT_FOLDER = os.path.join(FOLDER, 'outputs/DASC501/project')

# Configure plotting style (optional)
plt.style.use('ggplot')
sns.set_palette("viridis")
print("Libraries and custom classes imported successfully.")

# Create a directory for outputs if it doesn't exist
output_dir = OUTPUT_FOLDER
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")
else:
    print(f"Directory already exists: {output_dir}")

# Define file paths and table name
csv_file_path = os.path.join(INPUT_FOLDER, "global_health.csv")
db_file_path = os.path.join(OUTPUT_FOLDER, "global_health.db")
table_name = "health_data"

# Initialize DatabaseManager
db_manager = DatabaseManager(db_path=db_file_path)

try:
    db_manager.connect()
    print(f"Successfully connected to database: {db_file_path}")

    # Load data from CSV into the SQLite database using DatabaseManager
    # This method should handle reading the CSV, standardizing column names, 
    # creating the table, and inserting data.
    print(f"Loading data from {csv_file_path} into table '{table_name}'...")
    load_status = db_manager.load_data(
        file_path=csv_file_path,
        table_name=table_name,
        file_type=FileType.CSV, # Using the imported FileType enum
        clear_table_first=True, # Ensures a fresh load if the table already exists
        insert_strategy="INSERT OR REPLACE" # Strategy for inserting data
    )

    if load_status:
        print(f"Data from '{csv_file_path}' successfully processed for loading into table '{table_name}'.")
        db_manager.commit() # Explicitly commit the transaction
        print("Data committed to the database.")
        
        # Verify standardized column names by fetching the schema or a sample row
        df_check_schema = db_manager.execute_select_query(f"PRAGMA table_info({table_name});", return_dataframe=True)
        if df_check_schema is not None and not df_check_schema.empty:
            print("\nColumns in the database table (standardized names expected):")
            print(df_check_schema[['name', 'type']])
        else:
            print(f"Could not verify columns for table '{table_name}'.")
            
    else:
        print(f"Failed to load data into table '{table_name}'. Check DatabaseManager logs.")

except Exception as e:
    print(f"An error occurred during data loading: {e}")
    if db_manager and db_manager.conn: # Check if connection was established and an error occurred
        try:
            db_manager.rollback() # Rollback on error
            print("Transaction rolled back due to error during data loading.")
        except Exception as rb_e:
            print(f"Error during rollback: {rb_e}")
finally:
    if db_manager and db_manager.conn: # Check if connection was established
        db_manager.close()
        print("Database connection closed after initial load attempt.")

# Re-establish connection for SQL queries if needed, or ensure db_manager is still connected from a previous cell if run sequentially.
# For safety, we'll reconnect within this cell's scope.
db_manager_sql = DatabaseManager(db_path=db_file_path)

try:
    db_manager_sql.connect()
    print("\n--- SQL Exploration Results ---")

    # Query 1: Count distinct countries
    # Assuming 'Country' is the standardized column name (likely unchanged as it has no spaces)
    query1 = f"SELECT COUNT(DISTINCT Country) as DistinctCountries FROM {table_name};"
    print("\nQuery 1: Count of distinct countries")
    result1 = db_manager_sql.execute_select_query(query1, return_dataframe=True)
    print(result1)

    # Query 2: Find the range of years
    query2 = f"SELECT MIN(Year) as MinYear, MAX(Year) as MaxYear FROM {table_name};"
    print("\nQuery 2: Range of years in the dataset")
    result2 = db_manager_sql.execute_select_query(query2, return_dataframe=True)
    print(result2)

    # Query 3: Retrieve data for a specific country (e.g., 'Canada') and recent years (e.g., after 2018)
    country_to_check = 'Canada'
    year_threshold = 2018
    query3 = f"SELECT * FROM {table_name} WHERE Country = '{country_to_check}' AND Year > {year_threshold};"
    print(f"\nQuery 3: Data for {country_to_check} after {year_threshold}")
    result3 = db_manager_sql.execute_select_query(query3, return_dataframe=True)
    if result3 is not None and not result3.empty:
        print(result3.head())
    else:
        print(f"No data found for {country_to_check} after {year_threshold}, or query failed.")

    # Query 4: Top 10 countries with the highest average life expectancy
    # Standardized column name for 'Life Expectancy' is 'Life_Expectancy'
    life_exp_col = 'Life_Expectancy' 
    query4 = f"""
    SELECT Country, AVG("{life_exp_col}") as AvgLifeExpectancy
    FROM {table_name}
    WHERE "{life_exp_col}" IS NOT NULL AND TRIM(CAST("{life_exp_col}" AS TEXT)) != '' -- Ensure it's not NULL or empty string
    GROUP BY Country
    ORDER BY AvgLifeExpectancy DESC
    LIMIT 10;
    """
    print("\nQuery 4: Top 10 countries by average life expectancy")
    result4 = db_manager_sql.execute_select_query(query4, return_dataframe=True)
    print(result4)

    # Query 5: Get table schema (column names and types) - already shown in previous cell, but good for reference
    query5 = f"PRAGMA table_info({table_name});"
    print(f"\nQuery 5: Schema for table '{table_name}'")
    result5 = db_manager_sql.execute_select_query(query5, return_dataframe=True)
    print(result5[['name', 'type']])

except Exception as e:
    print(f"An error occurred during SQL exploration: {e}")
finally:
    if db_manager_sql and db_manager_sql.conn:
        db_manager_sql.close()
        print("\nDatabase connection closed after SQL exploration.")

# Load the entire table into a pandas DataFrame for EDA
db_manager_eda = DatabaseManager(db_path=db_file_path)
health_df = None # Initialize health_df
try:
    db_manager_eda.connect()
    health_df = db_manager_eda.execute_select_query(f"SELECT * FROM {table_name}", return_dataframe=True)
except Exception as e:
    print(f"Error connecting to database or fetching data: {e}")
finally:
    if db_manager_eda and db_manager_eda.conn:
        db_manager_eda.close()

if health_df is not None:
    print(f"\nData loaded into DataFrame for EDA. Shape: {health_df.shape}")
    print("\n--- DataFrame Head ---")
    print(health_df.head())
    
    print("\n--- Initial DataFrame Info (before type conversion) ---")
    health_df.info()

    # Data Cleaning and Type Conversion
    # Convert columns to numeric where appropriate. Errors will be coerced to NaN.
    # This is crucial because CSV data might be loaded as objects if there are mixed types or non-standard NaNs.
    print("\n--- Converting columns to numeric (errors coerced to NaN) ---")
    # Exclude known non-numeric columns like 'Country', 'Country_Code'
    # 'Year' should be integer, others float or integer.
    cols_to_convert = health_df.columns.drop(['Country', 'Country_Code'], errors='ignore')
    
    for col in cols_to_convert:
        if health_df[col].dtype == 'object':
            # Attempt to convert to numeric. If it fails for all values, it might remain object.
            health_df[col] = pd.to_numeric(health_df[col], errors='coerce')
            print(f"Attempted numeric conversion for column '{col}'. New dtype: {health_df[col].dtype}")
    
    # Specifically ensure 'Year' is integer if it's numeric
    if 'Year' in health_df.columns and pd.api.types.is_numeric_dtype(health_df['Year']):
        health_df['Year'] = health_df['Year'].astype('Int64') # Use Int64 to handle potential NaNs
        
    print("\n--- DataFrame Info After Type Conversion Attempts ---")
    health_df.info()

    print("\n--- Missing Values Count (after potential coercions) ---")
    print(health_df.isnull().sum().sort_values(ascending=False))

    print("\n--- Basic Descriptive Statistics (for numeric columns) ---")
    # .describe() will automatically select numeric columns
    if not health_df.empty:
        print(health_df.describe())
    else:
        print("DataFrame is empty, cannot generate descriptive statistics.")
else:
    print("Failed to load data into DataFrame for EDA.")

# Initialize ComprehensiveDataFrameAnalyzer
# Output plots and reports to 'eda_outputs/' directory
analyzer = ComprehensiveDataFrameAnalyzer(output_dir_default=output_dir, show_plots_default=False)

if health_df is not None and not health_df.empty:
    print("\n--- Comprehensive Descriptive Analysis (using DataFrameAnalyzer) ---")
    
    # Identify numeric columns after conversion for the analyzer
    numeric_cols_for_analyzer = health_df.select_dtypes(include=np.number).columns.tolist()
    
    if numeric_cols_for_analyzer:
        # The analyzer's method might generate plots and save them to its output_dir_default
        descriptive_results = analyzer.perform_descriptive_analysis(
            health_df, 
            columns=numeric_cols_for_analyzer, 
            generate_plots=True, # This should trigger plot saving
            plot_types=['boxplot', 'histogram']
        )
        print("Descriptive Statistics Summary (sample from DataFrameAnalyzer):")
        if 'statistics_summary' in descriptive_results:
            print(descriptive_results['statistics_summary'].head())
        else:
            print("Statistics summary not found in results.")
            
        if 'normality_tests' in descriptive_results:
            print("\nNormality Test Results (sample from DataFrameAnalyzer):")
            print(descriptive_results['normality_tests'].head())
            
        if 'plot_paths' in descriptive_results and descriptive_results['plot_paths']:
            print(f"\nDescriptive plots saved in '{os.path.join(output_dir, 'descriptive_plots')}/'. Example paths:")
            # Display a few example paths
            count = 0
            for col, paths in descriptive_results['plot_paths'].items():
                if count < 2:
                    print(f"  Column '{col}':")
                    for plot_type, path in paths.items():
                        print(f"    {plot_type}: {path}")
                    count += 1
                else:
                    break
        else:
            print("No plot paths found in descriptive analysis results or plots were not generated.")
            
    else:
        print("No numeric columns found for descriptive analysis after type conversion.")
else:
    print("health_df is None or empty, skipping comprehensive descriptive analysis.")

if health_df is not None and not health_df.empty:
    key_indicators_for_dist = ['Life_Expectancy', 'GDP_Per_Capita', 'Fertility_Rate', 'Obesity_Rate_Percent']
    print(f"\n--- Distribution Investigation for {key_indicators_for_dist} (using DataFrameAnalyzer) ---")
    
    for indicator in key_indicators_for_dist:
        if indicator in health_df.columns and health_df[indicator].notna().sum() > 5 : # Check if column exists and has enough data
            print(f"\nInvestigating distribution for: {indicator}")
            dist_results = analyzer.investigate_value_distribution(
                health_df.dropna(subset=[indicator]), # Pass df with NaNs removed for this column
                column_name=indicator,
                generate_plots=True # This should trigger plot saving
            )
            if 'error' in dist_results:
                print(f"  Error for {indicator}: {dist_results['error']}")
            else:
                print(f"  Moments for {indicator}: {dist_results.get('moments')}")
                print(f"  Best Fit for {indicator}: {dist_results.get('best_fit_distribution')}")
                if 'plot_paths' in dist_results and dist_results['plot_paths']:
                     print(f"  Plots saved for {indicator}. Paths: {dist_results['plot_paths']}")
                else:
                    print(f"  No plot paths found for {indicator} or plots were not generated.")
        else:
            print(f"Skipping distribution analysis for '{indicator}' due to missing column or insufficient non-NaN data.")
else:
    print("health_df is None or empty, skipping distribution investigation.")

correlation_results_global = {} # To store results for later use
if health_df is not None and not health_df.empty:
    print("\n--- Correlation Analysis (using DataFrameAnalyzer) ---")
    # Select relevant numeric columns for correlation
    correlation_cols = [
        'Life_Expectancy', 'GDP_Per_Capita', 'Fertility_Rate', 
        'Immunization_Rate', 'Unemployment_Rate', 'Obesity_Rate_Percent', 
        'Hospital_Beds_Per_1000', 'Water_Access_Percent', 'Urban_Population_Percent',
        'Alcohol_Consumption_Per_Capita', 'Suicide_Rate_Percent'
    ]
    # Filter to only include columns that actually exist in health_df and are numeric
    valid_correlation_cols = [col for col in correlation_cols if col in health_df.columns and pd.api.types.is_numeric_dtype(health_df[col])]

    if len(valid_correlation_cols) > 1:
        # DataFrameAnalyzer's correlation method will save the heatmap
        correlation_results_global = analyzer.analyze_column_correlations(
            health_df, 
            columns=valid_correlation_cols, 
            method='pearson',
            correlation_threshold=0.7, # Identify pairs with abs(correlation) > 0.7
            generate_heatmap=True 
        )
        print("Correlation Matrix (sample from DataFrameAnalyzer):")
        if 'correlation_matrix' in correlation_results_global:
            print(correlation_results_global['correlation_matrix'].head())
        else:
            print("Correlation matrix not found in results.")
            
        print("\nHighly Correlated Pairs (from DataFrameAnalyzer):")
        if 'highly_correlated_pairs' in correlation_results_global:
            print(correlation_results_global['highly_correlated_pairs'])
        else:
            print("Highly correlated pairs not found in results.")
            
        if 'heatmap_path' in correlation_results_global and correlation_results_global['heatmap_path']:
            print(f"\nCorrelation heatmap saved to: {correlation_results_global['heatmap_path']}")
        else:
            print("Heatmap path not found or heatmap was not generated.")
    else:
        print("Not enough valid numeric columns for correlation analysis.")
else:
    print("health_df is None or empty, skipping correlation analysis.")

if health_df is not None and not health_df.empty:
    print("--- Missing Values Summary ---")
    missing_summary = health_df.isnull().sum()
    missing_percent = (health_df.isnull().sum() / len(health_df)) * 100
    missing_df = pd.DataFrame({'Missing Count': missing_summary, 'Missing Percent': missing_percent})
    print(missing_df.sort_values(by='Missing Percent', ascending=False))
    
    # No global imputation will be applied here to keep the original data context for this EDA stage.
    # Visualizations will handle NaNs locally if necessary.
else:
    print("health_df is None or empty, skipping missing values summary.")

if health_df is not None and not health_df.empty:
    print("--- Visualization 1: Life Expectancy Over Time ---")
    # Select a few diverse countries for trend analysis
    countries_for_trend = ['United States', 'China', 'India', 'Nigeria', 'Germany', 'Brazil']
    
    # Filter DataFrame for selected countries. 'Year' and 'Life_Expectancy' should be numeric.
    trend_df_vis1 = health_df[health_df['Country'].isin(countries_for_trend)].copy()
    # Ensure 'Life_Expectancy' is numeric for plotting, errors='coerce' will turn non-numerics into NaN
    trend_df_vis1['Life_Expectancy'] = pd.to_numeric(trend_df_vis1['Life_Expectancy'], errors='coerce')
    trend_df_vis1.dropna(subset=['Year', 'Life_Expectancy'], inplace=True)

    if not trend_df_vis1.empty:
        plt.figure(figsize=(14, 7))
        sns.lineplot(data=trend_df_vis1, x='Year', y='Life_Expectancy', hue='Country', marker='o', linewidth=2)
        plt.title('Life Expectancy Over Time for Selected Countries (2012-2021)')
        plt.xlabel('Year')
        plt.ylabel('Life Expectancy (Years)')
        plt.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save the plot
        plot_path_1 = os.path.join(output_dir, "life_expectancy_trend.png")
        plt.savefig(plot_path_1, bbox_inches='tight')
        print(f"Plot 1 saved to {plot_path_1}")
        plt.show()
    else:
        print("No data available for the selected countries and Life Expectancy after cleaning.")
else:
    print("health_df is None or empty, skipping Visualization 1.")

if health_df is not None and not health_df.empty:
    print("\n--- Visualization 2: GDP per Capita vs. Life Expectancy ---")
    if 'Year' in health_df.columns and health_df['Year'].notna().any():
        latest_year = int(health_df['Year'].max()) # Ensure latest_year is an integer for the title
        latest_year_df_vis2 = health_df[health_df['Year'] == latest_year].copy()
        
        # Ensure relevant columns are numeric and drop NaNs for this plot
        cols_for_scatter = ['GDP_Per_Capita', 'Life_Expectancy', 'Total_Population']
        for col in cols_for_scatter:
            latest_year_df_vis2[col] = pd.to_numeric(latest_year_df_vis2[col], errors='coerce')
        latest_year_df_vis2.dropna(subset=cols_for_scatter, inplace=True)

        if not latest_year_df_vis2.empty:
            plt.figure(figsize=(12, 7))
            scatter_plot = sns.scatterplot(
                data=latest_year_df_vis2,
                x='GDP_Per_Capita',
                y='Life_Expectancy',
                size='Total_Population',
                hue='Total_Population', # Using population for hue as well, can be changed to a region if available
                sizes=(30, 600), # Adjusted size range for better visibility
                alpha=0.7,
                palette="viridis_r" # Using a reverse viridis palette
            )
            plt.title(f'GDP per Capita vs. Life Expectancy ({latest_year})')
            plt.xlabel('GDP per Capita (USD, Log Scale)')
            plt.ylabel('Life Expectancy (Years)')
            plt.xscale('log') # GDP per capita often has a wide, skewed distribution
            
            # Improve legend
            handles, labels = scatter_plot.get_legend_handles_labels()
            # Create a more manageable legend for size
            # This example simplifies by just showing the hue legend if it's based on a categorical variable
            # If hue is also numeric (like Total_Population), it might be better to remove one of the legends or use a color bar.
            # For now, we keep it as is, but this might need adjustment based on the specific data/grouping.
            plt.legend(title='Population', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)

            plt.grid(True, which="both", ls="--", linewidth=0.5)
            plt.tight_layout()
            
            plot_path_2 = os.path.join(output_dir, "gdp_vs_life_expectancy.png")
            plt.savefig(plot_path_2, bbox_inches='tight') # bbox_inches='tight' helps with legend
            print(f"Plot 2 saved to {plot_path_2}")
            plt.show()
        else:
            print("No data available for GDP vs Life Expectancy plot after cleaning.")
    else:
        print("Year column not found or is all NaN, cannot determine latest year for Visualization 2.")
else:
    print("health_df is None or empty, skipping Visualization 2.")

if health_df is not None and not health_df.empty:
    print("\n--- Visualization 3: Distribution of Obesity Rate ---")
    if 'Year' in health_df.columns and health_df['Year'].notna().any() and 'Obesity_Rate_Percent' in health_df.columns:
        latest_year = int(health_df['Year'].max())
        latest_year_df_vis3 = health_df[health_df['Year'] == latest_year].copy()
        
        latest_year_df_vis3['Obesity_Rate_Percent'] = pd.to_numeric(latest_year_df_vis3['Obesity_Rate_Percent'], errors='coerce')
        latest_year_df_vis3.dropna(subset=['Obesity_Rate_Percent'], inplace=True)

        if not latest_year_df_vis3.empty:
            plt.figure(figsize=(10, 6))
            sns.histplot(latest_year_df_vis3['Obesity_Rate_Percent'], kde=True, bins=25, color='teal')
            plt.title(f'Distribution of Obesity Rate ({latest_year})')
            plt.xlabel('Obesity Rate (%)')
            plt.ylabel('Number of Countries')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            plot_path_3 = os.path.join(output_dir, "obesity_rate_distribution.png")
            plt.savefig(plot_path_3)
            print(f"Plot 3 saved to {plot_path_3}")
            plt.show()
        else:
            print("No data available for Obesity Rate distribution plot after cleaning.")
    else:
        print("Required columns ('Year', 'Obesity_Rate_Percent') not found or are all NaN for Visualization 3.")
else:
    print("health_df is None or empty, skipping Visualization 3.")

if health_df is not None and not health_df.empty and 'correlation_results_global' in locals() and isinstance(correlation_results_global, dict) and 'heatmap_path' in correlation_results_global:
    print("\n--- Visualization 4: Correlation Heatmap ---")
    heatmap_path = correlation_results_global['heatmap_path']
    if heatmap_path and os.path.exists(heatmap_path):
        print(f"The correlation heatmap was generated during EDA and saved to: {heatmap_path}")
        print("Displaying the heatmap image below:")
        from IPython.display import Image, display
        display(Image(filename=heatmap_path))
    else:
        print(f"Heatmap image not found at the expected path: {heatmap_path}. It should have been saved by DataFrameAnalyzer. You may need to re-run the correlation analysis cell.")
        # As a fallback, if the image isn't found but matrix exists, try to replot a basic version
        if 'correlation_matrix' in correlation_results_global:
            print("Attempting to replot a basic heatmap...")
            plt.figure(figsize=(12, 10))
            sns.heatmap(correlation_results_global['correlation_matrix'], annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
            plt.title('Correlation Matrix of Selected Health Indicators (Fallback Plot)')
            plt.tight_layout()
            fallback_path = os.path.join(output_dir, "correlation_heatmap_fallback.png")
            plt.savefig(fallback_path)
            print(f"Fallback heatmap saved to {fallback_path}")
            plt.show()
elif health_df is not None and not health_df.empty:
    print("Correlation results or heatmap path not available. Please ensure the EDA correlation analysis cell was run successfully.")
    print("Attempting to generate correlation matrix and heatmap now...")
    correlation_cols_fallback = [
        'Life_Expectancy', 'GDP_Per_Capita', 'Fertility_Rate', 
        'Immunization_Rate', 'Unemployment_Rate', 'Obesity_Rate_Percent', 
        'Hospital_Beds_Per_1000', 'Water_Access_Percent'
    ]
    valid_correlation_cols_fallback = [col for col in correlation_cols_fallback if col in health_df.columns and pd.api.types.is_numeric_dtype(health_df[col])]
    if len(valid_correlation_cols_fallback) > 1:
        correlation_matrix_fallback = health_df[valid_correlation_cols_fallback].corr(method='pearson')
        plt.figure(figsize=(12,10))
        sns.heatmap(correlation_matrix_fallback, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title('Correlation Matrix of Selected Health Indicators (Generated On-the-fly)')
        plt.tight_layout()
        plot_path_4 = os.path.join(output_dir, "correlation_heatmap_generated.png")
        plt.savefig(plot_path_4, bbox_inches='tight')
        print(f"Plot 4 (heatmap) saved to {plot_path_4}")
        plt.show()
    else:
        print("Not enough valid numeric columns to generate a fallback correlation heatmap.")
else:
    print("health_df is None or empty, skipping Visualization 4.")

if health_df is not None and not health_df.empty:
    print("\n--- Visualization 5: Average Hospital Beds per 1000 by Income Group ---")
    if 'Year' in health_df.columns and health_df['Year'].notna().any():
        latest_year = int(health_df['Year'].max())
        latest_year_df_vis5 = health_df[health_df['Year'] == latest_year].copy()
        
        # Ensure relevant columns are numeric and drop NaNs
        cols_for_grouping = ['GDP_Per_Capita', 'Hospital_Beds_Per_1000']
        for col in cols_for_grouping:
            latest_year_df_vis5[col] = pd.to_numeric(latest_year_df_vis5[col], errors='coerce')
        latest_year_df_vis5.dropna(subset=cols_for_grouping, inplace=True)

        if not latest_year_df_vis5.empty and len(latest_year_df_vis5['GDP_Per_Capita'].unique()) >= 4:
            # Create income groups based on GDP_Per_Capita quantiles (quartiles)
            try:
                latest_year_df_vis5['Income_Group'] = pd.qcut(
                    latest_year_df_vis5['GDP_Per_Capita'], 
                    q=4, 
                    labels=['Low', 'Lower-Middle', 'Upper-Middle', 'High'],
                    duplicates='drop' # Important for handling non-unique quantile edges
                )
            except ValueError as e:
                print(f"Could not create 4 income groups due to non-unique quantiles: {e}. Assigning to 'General' group.")
                latest_year_df_vis5['Income_Group'] = 'General' # Fallback if qcut fails

            avg_beds_by_income = latest_year_df_vis5.groupby('Income_Group')['Hospital_Beds_Per_1000'].mean().reset_index()

            if not avg_beds_by_income.empty:
                plt.figure(figsize=(10, 6))
                sns.barplot(data=avg_beds_by_income, x='Income_Group', y='Hospital_Beds_Per_1000', palette='coolwarm', order=['Low', 'Lower-Middle', 'Upper-Middle', 'High'] if 'Low' in avg_beds_by_income['Income_Group'].values else None)
                plt.title(f'Average Hospital Beds per 1000 by GDP-based Income Group ({latest_year})')
                plt.xlabel('Income Group (based on GDP per Capita)')
                plt.ylabel('Average Hospital Beds per 1000 Population')
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                
                plot_path_5 = os.path.join(output_dir, "hospital_beds_by_income_group.png")
                plt.savefig(plot_path_5)
                print(f"Plot 5 saved to {plot_path_5}")
                plt.show()
            else:
                print("No data to plot for average hospital beds by income group.")
        elif latest_year_df_vis5.empty:
            print("No data available after filtering for hospital beds and GDP for Visualization 5.")
        else:
            print("Not enough unique GDP per Capita values to form 4 distinct income groups. Skipping Visualization 5.")
    else:
        print("Year column not found or is all NaN, cannot determine latest year for Visualization 5.")
else:
    print("health_df is None or empty, skipping Visualization 5.")

if health_df is not None and not health_df.empty:
    print("\n--- Visualization 6: Top 10 Countries by Immunization Rate ---")
    # Assuming 'Immunization_Rate' refers to a key childhood vaccine like DPT or Measles.
    immunization_col_name = 'Immunization_Rate' 
    
    if 'Year' in health_df.columns and health_df['Year'].notna().any() and immunization_col_name in health_df.columns:
        latest_year = int(health_df['Year'].max())
        latest_year_df_vis6 = health_df[health_df['Year'] == latest_year].copy()
        
        latest_year_df_vis6[immunization_col_name] = pd.to_numeric(latest_year_df_vis6[immunization_col_name], errors='coerce')
        latest_year_df_vis6.dropna(subset=[immunization_col_name, 'Country'], inplace=True)
        
        top_10_immunization = latest_year_df_vis6.sort_values(by=immunization_col_name, ascending=False).head(10)

        if not top_10_immunization.empty:
            plt.figure(figsize=(12, 7))
            sns.barplot(data=top_10_immunization, y='Country', x=immunization_col_name, palette='summer_r')
            plt.title(f'Top 10 Countries by Immunization Rate ({latest_year})')
            plt.xlabel('Immunization Rate (%)')
            plt.ylabel('Country')
            plt.xlim(0, 105) # Rates are percentages, allow for slight over 100 if data has it
            plt.grid(axis='x', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            plot_path_6 = os.path.join(output_dir, "top_10_immunization_rate.png")
            plt.savefig(plot_path_6)
            print(f"Plot 6 saved to {plot_path_6}")
            plt.show()
        else:
            print("No data available for immunization rate ranking after cleaning.")
            
    else:
        print(f"Required columns ('Year', '{immunization_col_name}') not found or are all NaN for Visualization 6.")
else:
    print("health_df is None or empty, skipping Visualization 6.")
