import os
import sqlite3
import pandas as pd
import numpy as np
import logging # Import logging for the main script

# Import from your provided files
from DASC500.classes.DatabaseManager import DatabaseManager # Assuming DatabaseManager.py is in the same directory
from DASC500.stats.outlier_detection import run_outlier_detection # Assuming (modified) outlier_detection.py is here
from DASC500.utilities.get_top_level_module import get_top_level_module_path

# --- Configuration for the main script ---
FOLDER = os.path.join(get_top_level_module_path(), '../..')
DB_PATH = os.path.join(FOLDER, "data/DASC501/homework6/DataViz501.db")
TABLE_NAME = "military-bases" # Standardized table name
COLUMNS_TO_ANALYZE = ['PERIMETER', 'AREA', 'Shape_Leng', 'Shape_Area']
OUTLIER_METHODS = ['iqr', 'zscore', 'pca']
# Output directory for all reports and plots
# The run_outlier_detection script will create a timestamped sub-folder within this
MAIN_OUTPUT_DIR = os.path.join(FOLDER, "outputs/DASC501/homework8")


# --- Set up basic logging for this script ---
script_logger = logging.getLogger(__name__)
script_logger.setLevel(logging.INFO)
if not script_logger.hasHandlers():
    # Console Handler for script's own logs
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    script_logger.addHandler(console_handler)

    # File Handler for script's own logs (optional, but good practice)
    # This log will be outside the timestamped folder from run_outlier_detection
    # os.makedirs(MAIN_OUTPUT_DIR, exist_ok=True) # Ensure main output dir exists
    # script_file_handler = logging.FileHandler(os.path.join(MAIN_OUTPUT_DIR, "main_script.log"))
    # script_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    # script_logger.addHandler(script_file_handler)


def part1():
    script_logger.info("Starting Part 1A: Outlier Detection an.d Reporting.")

    # --- 1. Create Dummy Database (or ensure your DB is available) ---
    # For this example, we create a dummy DB. In a real scenario, DataViz501.db would exist.

    # --- 2. Load Data using DatabaseManager ---
    db_manager = DatabaseManager(db_path=DB_PATH)
    df_military_bases = None
    try:
        with db_manager: # Connects and closes automatically
            script_logger.info(f"Attempting to load table '{TABLE_NAME}' from '{DB_PATH}'.")
            # Ensure the table name in SQL query matches exactly how it is in the DB
            # If table/column names have spaces or special chars, they might need quoting in SQL
            # Here, create_dummy_database creates it as 'military_bases'
            df_military_bases = db_manager.execute_select_query(f'SELECT * FROM `{TABLE_NAME}`')

        if df_military_bases is None or df_military_bases.empty:
            script_logger.error(f"Failed to load data from table '{TABLE_NAME}' or table is empty.")
            return
        script_logger.info(f"Successfully loaded {len(df_military_bases)} rows from table '{TABLE_NAME}'.")
        script_logger.info(f"DataFrame columns: {df_military_bases.columns.tolist()}")

        # Drop ID column if it was loaded, as it's not for outlier analysis
        if 'ID' in df_military_bases.columns:
            df_military_bases = df_military_bases.drop(columns=['ID'])
            script_logger.info("Dropped 'ID' column from DataFrame.")

        # Verify that the necessary columns for analysis are present
        missing_cols = [col for col in COLUMNS_TO_ANALYZE if col not in df_military_bases.columns]
        if missing_cols:
            script_logger.error(f"The following columns required for analysis are missing from the loaded table: {missing_cols}")
            script_logger.error(f"Available columns: {df_military_bases.columns.tolist()}")
            return
        
        script_logger.info(f"Proceeding with columns for analysis: {COLUMNS_TO_ANALYZE}")


    except Exception as e:
        script_logger.error(f"An error occurred during database interaction: {e}", exc_info=True)
        return

    # --- 3. Run Outlier Detection ---
    if df_military_bases is not None and not df_military_bases.empty:
        script_logger.info("Running comprehensive outlier detection...")
        try:
            # The run_outlier_detection function will create its own timestamped subdirectory
            # within MAIN_OUTPUT_DIR.
            results = run_outlier_detection(
                df=df_military_bases,
                columns=COLUMNS_TO_ANALYZE,
                methods=OUTLIER_METHODS,
                output_dir=MAIN_OUTPUT_DIR, # Pass the main output directory
                create_plots=True,       # Will save plots to files
                show_plots=False,        # No interactive plot showing
                save_csv=True,
                include_descriptive_stats=True,
                replot_without_outliers=True
            )
            script_logger.info("Outlier detection process complete.")
            # The HTML report and other files will be in a timestamped subfolder of MAIN_OUTPUT_DIR
            # Example: ./homework_part1a_output/outlier_detection_20250530_163000/outlier_detection_report.html

            # You can inspect the 'results' dictionary if needed for further programmatic steps
            # For Part 1A, the generation of the report and plots by run_outlier_detection is the main goal.

        except Exception as e:
            script_logger.error(f"An error occurred during outlier detection: {e}", exc_info=True)
    else:
        script_logger.warning("DataFrame is empty. Skipping outlier detection.")

    script_logger.info("Homework Part 1A script finished.")

# ------------ PART 2 ------------
import os
import sqlite3
import pandas as pd
import numpy as np
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

def initialize_spark():
    """Initialize Spark session"""
    spark = SparkSession.builder \
        .appName("MilitaryBasesAnalysis") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()
    
    # Set log level to reduce verbose output
    spark.sparkContext.setLogLevel("WARN")
    script_logger.info("Spark session initialized successfully")
    return spark

def load_data_from_database():
    """Load military bases data from SQLite database"""
    db_manager = DatabaseManager(db_path=DB_PATH)
    df_pandas = None
    
    try:
        with db_manager:
            script_logger.info(f"Loading table '{TABLE_NAME}' from '{DB_PATH}'")
            df_pandas = db_manager.execute_select_query(f'SELECT * FROM `{TABLE_NAME}`')
        
        if df_pandas is None or df_pandas.empty:
            script_logger.error(f"Failed to load data from table '{TABLE_NAME}' or table is empty")
            return None
            
        script_logger.info(f"Successfully loaded {len(df_pandas)} rows from database")
        script_logger.info(f"Columns: {df_pandas.columns.tolist()}")
        return df_pandas
        
    except Exception as e:
        script_logger.error(f"Error loading data from database: {e}")
        return None

def create_sample_data():
    """Create sample military bases data for demonstration"""
    script_logger.info("Creating sample military bases data")
    
    # Sample data based on your table structure
    sample_data = [
        ("Base Alpha", "Army", "Active", "California", "USA", 15.5, 120.3, 2500.0, 2480.5),
        ("Naval Station Beta", "Navy", "Active", "Florida", "USA", 22.1, 185.7, 3200.1, 3180.2),
        ("Air Force Gamma", "Air Force", "Active", "Texas", "USA", 18.3, 145.2, 2800.5, 2775.8),
        ("Marine Base Delta", "Marines", "Active", "North Carolina", "USA", 12.8, 98.4, 1950.2, 1925.1),
        ("Joint Base Echo", "Joint", "Active", "Virginia", "USA", 25.6, 210.8, 4100.3, 4075.6),
        ("Reserve Station Foxtrot", "Army", "Reserve", "Ohio", "USA", 8.2, 65.1, 1200.0, 1185.5),
        ("Training Base Golf", "Air Force", "Training", "Nevada", "USA", 35.2, 285.6, 5500.2, 5475.8),
        ("Depot Hotel", "Navy", "Maintenance", "Washington", "USA", 14.1, 108.7, 2100.1, 2085.3),
        ("Forward Base India", "Army", "Active", "Alaska", "USA", 28.9, 245.3, 4750.5, 4725.2),
        ("Air Station Juliet", "Air Force", "Active", "Hawaii", "USA", 19.7, 158.4, 3050.8, 3025.1),
        ("Naval Yard Kilo", "Navy", "Active", "Maine", "USA", 11.5, 88.2, 1750.3, 1730.6),
        ("Training Center Lima", "Marines", "Training", "South Carolina", "USA", 16.8, 132.5, 2650.7, 2625.4),
        ("Joint Facility Mike", "Joint", "Active", "Colorado", "USA", 21.3, 172.9, 3350.2, 3325.8),
        ("Reserve Base November", "Army", "Reserve", "Georgia", "USA", 9.6, 75.8, 1450.5, 1430.2),
        ("Air Defense Oscar", "Air Force", "Active", "New Mexico", "USA", 13.2, 102.1, 2000.8, 1985.5)
    ]
    
    columns = ["Site_Name", "COMPONENT", "Oper_Stat", "State_Terr", "COUNTRY", 
               "PERIMETER", "AREA", "Shape_Leng", "Shape_Area"]
    
    df_pandas = pd.DataFrame(sample_data, columns=columns)
    
    # Add some additional columns for more interesting analysis
    df_pandas['Base_Size_Category'] = pd.cut(df_pandas['AREA'], 
                                           bins=[0, 100, 200, 300, float('inf')], 
                                           labels=['Small', 'Medium', 'Large', 'Very Large'])
    
    df_pandas['Region'] = df_pandas['State_Terr'].map({
        'California': 'West', 'Nevada': 'West', 'Washington': 'West', 'Alaska': 'West', 'Hawaii': 'West',
        'Texas': 'South', 'Florida': 'South', 'North Carolina': 'South', 'Virginia': 'South', 
        'South Carolina': 'South', 'Georgia': 'South', 'New Mexico': 'South',
        'Ohio': 'Midwest', 'Colorado': 'West', 'Maine': 'Northeast'
    })
    
    return df_pandas

def task_a_spark_sql_queries(spark, df_spark):
    """Task A: Perform Spark SQL queries"""
    script_logger.info("=== TASK A: SPARK SQL QUERIES ===")
    
    # Register DataFrame as temporary view
    df_spark.createOrReplaceTempView("military_bases")
    script_logger.info("Registered DataFrame as temporary view 'military_bases'")
    
    # Query 1: Grouping and aggregation - Average area by component and operational status
    script_logger.info("\n--- Query 1: Average area by component and operational status ---")
    query1 = """
    SELECT 
        COMPONENT,
        Oper_Stat,
        COUNT(*) as base_count,
        ROUND(AVG(AREA), 2) as avg_area,
        ROUND(AVG(PERIMETER), 2) as avg_perimeter,
        ROUND(MAX(AREA), 2) as max_area,
        ROUND(MIN(AREA), 2) as min_area
    FROM military_bases 
    GROUP BY COMPONENT, Oper_Stat
    ORDER BY avg_area DESC
    """
    
    result1 = spark.sql(query1)
    result1.show(truncate=False)
    
    # Query 2: Complex filtering with CASE WHEN and multiple conditions
    script_logger.info("\n--- Query 2: Complex filtering with CASE WHEN ---")
    query2 = """
    SELECT 
        Site_Name,
        COMPONENT,
        State_Terr,
        Region,
        AREA,
        CASE 
            WHEN AREA > 250 THEN 'Very Large'
            WHEN AREA > 150 THEN 'Large'
            WHEN AREA > 100 THEN 'Medium'
            ELSE 'Small'
        END as size_category,
        CASE 
            WHEN COMPONENT = 'Joint' THEN 'Multi-Service'
            WHEN COMPONENT IN ('Army', 'Navy', 'Air Force', 'Marines') THEN 'Single-Service'
            ELSE 'Other'
        END as service_type,
        ROUND(AREA / PERIMETER, 2) as area_to_perimeter_ratio
    FROM military_bases 
    WHERE AREA > 100 
        AND Oper_Stat = 'Active'
        AND Region IS NOT NULL
    ORDER BY AREA DESC, COMPONENT
    """
    
    result2 = spark.sql(query2)
    result2.show(truncate=False)
    
    return result1, result2

def task_b_dataframe_operations(spark, df_spark):
    """Task B: Use PySpark DataFrame functions"""
    script_logger.info("\n=== TASK B: PYSPARK DATAFRAME OPERATIONS ===")
    
    # B1: Filtering and transformation on columns
    script_logger.info("\n--- B1: Filtering and Transformation ---")
    
    # Filter active bases with area > 100 and add transformed columns
    filtered_df = df_spark.filter(
        (col("Oper_Stat") == "Active") & 
        (col("AREA") > 100)
    ).withColumn(
        "area_category", 
        when(col("AREA") > 250, "Very Large")
        .when(col("AREA") > 150, "Large") 
        .when(col("AREA") > 100, "Medium")
        .otherwise("Small")
    ).withColumn(
        "area_efficiency",
        round(col("AREA") / col("PERIMETER"), 3)
    ).withColumn(
        "is_joint_base",
        col("COMPONENT") == "Joint"
    ).withColumn(
        "base_priority",
        when(col("COMPONENT") == "Joint", 1)
        .when(col("area_category") == "Very Large", 2)
        .when(col("COMPONENT").isin(["Army", "Navy", "Air Force"]), 3)
        .otherwise(4)
    )
    
    script_logger.info("Filtered and transformed data:")
    filtered_df.select("Site_Name", "COMPONENT", "AREA", "area_category", 
                      "area_efficiency", "is_joint_base", "base_priority").show(truncate=False)
    
    # B2: Grouping and aggregation
    script_logger.info("\n--- B2: Grouping and Aggregation ---")
    
    aggregated_df = df_spark.groupBy("COMPONENT", "Region").agg(
        count("*").alias("base_count"),
        round(avg("AREA"), 2).alias("avg_area"),
        round(avg("PERIMETER"), 2).alias("avg_perimeter"),
        max("AREA").alias("max_area"),
        min("AREA").alias("min_area"),
        round(stddev("AREA"), 2).alias("area_std_dev"),
        round(sum("AREA"), 2).alias("total_area")
    ).orderBy(desc("total_area"))
    
    script_logger.info("Grouped and aggregated data:")
    aggregated_df.show(truncate=False)
    
    # B3: Join two datasets
    script_logger.info("\n--- B3: Join Operations ---")
    
    # Create a second dataset with regional information
    regional_data = [
        ("West", "Pacific", 5, 850000),
        ("South", "Atlantic/Gulf", 8, 1200000), 
        ("Midwest", "Great Lakes", 2, 180000),
        ("Northeast", "Atlantic", 1, 95000)
    ]
    
    regional_schema = StructType([
        StructField("Region", StringType(), True),
        StructField("Coast_Type", StringType(), True),
        StructField("Region_Base_Count", IntegerType(), True),
        StructField("Region_Population", IntegerType(), True)
    ])
    
    regional_df = spark.createDataFrame(regional_data, regional_schema)
    
    script_logger.info("Regional information dataset:")
    regional_df.show()
    
    # Perform inner join
    joined_df = df_spark.join(regional_df, "Region", "inner")
    
    # Add calculated columns after join
    final_df = joined_df.withColumn(
        "population_per_base_ratio",
        round(col("Region_Population") / col("Region_Base_Count"), 0)
    ).withColumn(
        "base_density_score",  
        round(col("AREA") / col("Region_Population") * 1000000, 4)
    )
    
    script_logger.info("Joined data with calculated metrics:")
    final_df.select("Site_Name", "COMPONENT", "Region", "Coast_Type", 
                   "AREA", "Region_Population", "population_per_base_ratio", 
                   "base_density_score").show(truncate=False)
    
    return filtered_df, aggregated_df, final_df

def generate_summary_statistics(df_spark):
    """Generate comprehensive summary statistics"""
    script_logger.info("\n=== SUMMARY STATISTICS ===")
    
    # Basic statistics
    script_logger.info("\n--- Basic Dataset Statistics ---")
    print(f"Total number of military bases: {df_spark.count()}")
    
    df_spark.groupBy("COMPONENT").count().orderBy(desc("count")).show()
    df_spark.groupBy("Oper_Stat").count().show()
    df_spark.groupBy("Region").count().show()
    
    # Numerical statistics
    script_logger.info("\n--- Numerical Column Statistics ---")
    numerical_stats = df_spark.select("AREA", "PERIMETER", "Shape_Leng", "Shape_Area").describe()
    numerical_stats.show()

def part2():
    """Main function to execute all tasks"""
    script_logger.info("Starting PySpark Military Bases Analysis")
    
    # Initialize Spark
    spark = initialize_spark()
    
    try:
        # Try to load data from database first
        df_pandas = load_data_from_database()
        
        # If database loading fails, use sample data
        if df_pandas is None:
            script_logger.info("Using sample data for demonstration")
            df_pandas = create_sample_data()
        
        # Convert to Spark DataFrame
        df_spark = spark.createDataFrame(df_pandas)
        script_logger.info("Converted pandas DataFrame to Spark DataFrame")
        
        # Show schema and sample data
        script_logger.info("\n--- DataFrame Schema ---")
        df_spark.printSchema()
        
        script_logger.info("\n--- Sample Data ---")
        df_spark.show(5, truncate=False)
        
        # Execute Task A: Spark SQL queries
        result1, result2 = task_a_spark_sql_queries(spark, df_spark)
        
        # Execute Task B: DataFrame operations
        filtered_df, aggregated_df, joined_df = task_b_dataframe_operations(spark, df_spark)
        
        # Generate summary statistics
        generate_summary_statistics(df_spark)
        
        script_logger.info("\n=== ANALYSIS COMPLETE ===")
        script_logger.info("All tasks completed successfully!")
        
        # Cache important results for potential further analysis
        joined_df.cache()
        script_logger.info("Final joined dataset cached for further use")
        
    except Exception as e:
        script_logger.error(f"An error occurred during analysis: {e}", exc_info=True)
        
    finally:
        # Clean up Spark session  
        spark.stop()
        script_logger.info("Spark session stopped")

if __name__ == "__main__":
    # Add a top-level import for sys, used in the modified outlier_detection.py
    part1()
    part2()