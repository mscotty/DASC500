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
import sys
import sqlite3
import pandas as pd
import numpy as np
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

#sys.path.append(r'D:\Mitchell\School\2025_Winter\DASC500\github\DASC500\.venv9\Scripts')
sys.path.append(r'C:\Program Files\Java\jdk-17\bin')

# Set environment variables before importing pyspark
"""import shutil
python_path = r'D:\Mitchell\School\2025_Winter\DASC500\github\DASC500\.venv9\Scripts\python.exe'
#python_path = shutil.which("python")
if python_path and os.path.exists(python_path):
    os.environ['PYSPARK_PYTHON'] = python_path
    os.environ['PYSPARK_DRIVER_PYTHON'] = python_path
else:
    raise EnvironmentError("No valid Python executable found for Spark workers.")"""

os.environ['PYTHONHASHSEED'] = '0'

def initialize_spark_robust():
    """Initialize Spark session with robust Windows configuration"""
    import os
    import tempfile
    
    # Set environment variables for better Windows compatibility
    """os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable"""
    os.environ['PYTHONHASHSEED'] = '0'
    
    # Create temp directory for Spark
    temp_dir = tempfile.mkdtemp()
    
    spark = SparkSession.builder \
        .appName("MilitaryBasesAnalysis") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.driver.host", "localhost") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .config("spark.python.worker.reuse", "false") \
        .config("spark.python.worker.timeout", "600") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.local.dir", temp_dir) \
        .config("spark.driver.maxResultSize", "8g") \
        .config("spark.executor.heartbeatInterval", "60s") \
        .config("spark.network.timeout", "600s") \
        .config("spark.sql.execution.pyspark.udf.faulthandler.enabled", "true") \
        .config("spark.python.worker.faulthandler.enabled", "true") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "8g") \
        .config("spark.sql.shuffle.partitions", "8") \
        .master("local[1]") \
        .getOrCreate()  # Use only 1 core to avoid worker issues
    
    spark.sparkContext.setLogLevel("ERROR")  # Reduce logging noise
    script_logger.info("Spark session initialized with robust Windows configuration")
    return spark


def load_data_from_database():
    """Load military bases data from SQLite database with comprehensive cleanup"""
    db_manager = DatabaseManager(db_path=DB_PATH)
    df_pandas = None
    
    try:
        with db_manager:
            script_logger.info(f"Loading table '{TABLE_NAME}' from '{DB_PATH}'")
            df_pandas = db_manager.execute_select_query(f'SELECT * FROM `{TABLE_NAME}`')
        
        if df_pandas is None or df_pandas.empty:
            script_logger.error(f"Failed to load data from table '{TABLE_NAME}' or table is empty")
            return None
        
        script_logger.info(f"Raw data loaded: {len(df_pandas)} rows, {len(df_pandas.columns)} columns")
        
        # COMPREHENSIVE DATA CLEANUP
        
        # 1. Clean column names - replace spaces with underscores and normalize
        original_columns = df_pandas.columns.tolist()
        df_pandas.columns = df_pandas.columns.str.replace(' ', '_', regex=False)
        df_pandas.columns = df_pandas.columns.str.replace('-', '_', regex=False)
        df_pandas.columns = df_pandas.columns.str.strip()
        script_logger.info(f"Cleaned column names from {original_columns} to {df_pandas.columns.tolist()}")
        
        # 2. Clean ALL string columns - strip whitespace and handle empty strings
        string_columns = df_pandas.select_dtypes(include=['object']).columns
        for col in string_columns:
            if col in df_pandas.columns:
                # Convert to string first, then strip whitespace
                df_pandas[col] = df_pandas[col].astype(str).str.strip()
                # Replace 'nan', 'None', empty strings with actual NaN
                df_pandas[col] = df_pandas[col].replace(['nan', 'None', '', '  ', 'null', 'NULL'], np.nan)
                # For remaining strings, limit length to prevent serialization issues
                df_pandas[col] = df_pandas[col].apply(lambda x: x[:500] if isinstance(x, str) and len(x) > 500 else x)
                
        script_logger.info(f"Cleaned {len(string_columns)} string columns: {string_columns.tolist()}")
        
        # 3. Handle specific problematic columns (Geo columns often cause issues)
        geo_columns = ['Geo_Point', 'Geo_Shape']
        for col in geo_columns:
            if col in df_pandas.columns:
                # Convert complex geo data to simple string representation
                df_pandas[col] = df_pandas[col].astype(str)
                # Truncate extremely long geo strings that might cause serialization issues
                df_pandas[col] = df_pandas[col].apply(lambda x: x[:200] + "..." if len(str(x)) > 200 else str(x))
                script_logger.info(f"Cleaned geo column {col}")
        
        # 4. Clean and validate numeric columns
        numeric_columns = ['OBJECTID_1', 'OBJECTID', 'PERIMETER', 'AREA', 'Shape_Leng', 'Shape_Area']
        for col in numeric_columns:
            if col in df_pandas.columns:
                # Convert to numeric, coercing errors to NaN
                original_dtype = df_pandas[col].dtype
                df_pandas[col] = pd.to_numeric(df_pandas[col], errors='coerce')
                
                # Handle infinite values
                df_pandas[col] = df_pandas[col].replace([np.inf, -np.inf], np.nan)
                
                # Cap extremely large values that might cause issues
                if col in ['AREA', 'PERIMETER', 'Shape_Area', 'Shape_Leng']:
                    max_reasonable = df_pandas[col].quantile(0.99) * 10  # 10x the 99th percentile
                    df_pandas[col] = df_pandas[col].clip(upper=max_reasonable)
                
                script_logger.info(f"Cleaned numeric column {col}: {original_dtype} -> {df_pandas[col].dtype}")
        
        # 5. Handle Joint_Base column specifically
        if "Joint_Base" in df_pandas.columns:
            df_pandas["Joint_Base"] = df_pandas["Joint_Base"].fillna("N/A").astype(str)
            df_pandas["Joint_Base"] = df_pandas["Joint_Base"].str.strip()
            script_logger.info("Cleaned Joint_Base column")
        
        # 6. Add Region column with robust state mapping
        def assign_region(state):
            """Assign region based on state/territory with robust handling"""
            if pd.isna(state) or state == 'nan' or state == '':
                return "Unknown"
            
            state = str(state).strip()
            
            # Define regional mappings - handle both abbreviations and full names
            west_mapping = {
                'CA': 'West', 'California': 'West',
                'OR': 'West', 'Oregon': 'West', 
                'WA': 'West', 'Washington': 'West',
                'NV': 'West', 'Nevada': 'West',
                'ID': 'West', 'Idaho': 'West',
                'UT': 'West', 'Utah': 'West',
                'AZ': 'West', 'Arizona': 'West',
                'MT': 'West', 'Montana': 'West',
                'WY': 'West', 'Wyoming': 'West',
                'CO': 'West', 'Colorado': 'West',
                'NM': 'West', 'New Mexico': 'West',
                'AK': 'West', 'Alaska': 'West',
                'HI': 'West', 'Hawaii': 'West'
            }
            
            south_mapping = {
                'TX': 'South', 'Texas': 'South',
                'OK': 'South', 'Oklahoma': 'South',
                'AR': 'South', 'Arkansas': 'South',
                'LA': 'South', 'Louisiana': 'South',
                'MS': 'South', 'Mississippi': 'South',
                'AL': 'South', 'Alabama': 'South',
                'TN': 'South', 'Tennessee': 'South',
                'KY': 'South', 'Kentucky': 'South',
                'WV': 'South', 'West Virginia': 'South',
                'VA': 'South', 'Virginia': 'South',
                'NC': 'South', 'North Carolina': 'South',
                'SC': 'South', 'South Carolina': 'South',
                'GA': 'South', 'Georgia': 'South',
                'FL': 'South', 'Florida': 'South',
                'DE': 'South', 'Delaware': 'South',
                'MD': 'South', 'Maryland': 'South',
                'DC': 'South', 'District of Columbia': 'South'
            }
            
            midwest_mapping = {
                'ND': 'Midwest', 'North Dakota': 'Midwest',
                'SD': 'Midwest', 'South Dakota': 'Midwest',
                'NE': 'Midwest', 'Nebraska': 'Midwest',
                'KS': 'Midwest', 'Kansas': 'Midwest',
                'MN': 'Midwest', 'Minnesota': 'Midwest',
                'IA': 'Midwest', 'Iowa': 'Midwest',
                'MO': 'Midwest', 'Missouri': 'Midwest',
                'WI': 'Midwest', 'Wisconsin': 'Midwest',
                'IL': 'Midwest', 'Illinois': 'Midwest',
                'IN': 'Midwest', 'Indiana': 'Midwest',
                'MI': 'Midwest', 'Michigan': 'Midwest',
                'OH': 'Midwest', 'Ohio': 'Midwest'
            }
            
            northeast_mapping = {
                'ME': 'Northeast', 'Maine': 'Northeast',
                'NH': 'Northeast', 'New Hampshire': 'Northeast',
                'VT': 'Northeast', 'Vermont': 'Northeast',
                'MA': 'Northeast', 'Massachusetts': 'Northeast',
                'RI': 'Northeast', 'Rhode Island': 'Northeast',
                'CT': 'Northeast', 'Connecticut': 'Northeast',
                'NY': 'Northeast', 'New York': 'Northeast',
                'NJ': 'Northeast', 'New Jersey': 'Northeast',
                'PA': 'Northeast', 'Pennsylvania': 'Northeast'
            }
            
            # Combine all mappings
            all_mappings = {**west_mapping, **south_mapping, **midwest_mapping, **northeast_mapping}
            
            # Try exact match first
            if state in all_mappings:
                return all_mappings[state]
            
            # Try case-insensitive match
            for key, value in all_mappings.items():
                if state.lower() == key.lower():
                    return value
            
            # Handle territories and other cases
            territories = ['PR', 'Puerto Rico', 'GU', 'Guam', 'VI', 'US Virgin Islands', 
                          'AS', 'American Samoa', 'MP', 'Northern Mariana Islands']
            if any(state.lower() == t.lower() for t in territories):
                return "Territory"
            
            return "Other"
        
        # Apply region mapping - check for state column
        state_column = None
        possible_state_columns = ['State_Terr', 'State', 'STATE', 'state', 'Territory', 'TERRITORY']
        for col in possible_state_columns:
            if col in df_pandas.columns:
                state_column = col
                break
        
        if state_column:
            df_pandas['Region'] = df_pandas[state_column].apply(assign_region)
            script_logger.info(f"Added Region column based on {state_column}")
            region_counts = df_pandas['Region'].value_counts()
            script_logger.info(f"Region distribution: {region_counts.to_dict()}")
        else:
            df_pandas['Region'] = "Unknown"
            script_logger.warning("No state column found, assigned 'Unknown' to all regions")
        
        # 7. Final data validation and cleanup
        # Remove any rows that are completely empty
        df_pandas = df_pandas.dropna(how='all')
        
        # Ensure all object columns are properly stringified and not too long
        for col in df_pandas.select_dtypes(include=['object']).columns:
            df_pandas[col] = df_pandas[col].astype(str)
            df_pandas[col] = df_pandas[col].apply(lambda x: x[:300] if len(str(x)) > 300 else x)
        
        # Log final data statistics
        script_logger.info(f"Final cleaned data: {len(df_pandas)} rows, {len(df_pandas.columns)} columns")
        script_logger.info(f"Memory usage: {df_pandas.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        script_logger.info(f"Null value counts by column:")
        null_counts = df_pandas.isnull().sum()
        for col, count in null_counts.items():
            if count > 0:
                script_logger.info(f"  {col}: {count} nulls ({count/len(df_pandas)*100:.1f}%)")
        
        return df_pandas
        
    except Exception as e:
        script_logger.error(f"Error loading data from database: {e}")
        return None

def create_spark_dataframe_robust(spark, df_pandas):
    """Create Spark DataFrame with robust schema and additional safeguards"""
    script_logger.info("Converting pandas DataFrame to Spark DataFrame with robust schema")
    
    try:
        # Further cleanup before Spark conversion
        script_logger.info("Performing final cleanup before Spark conversion...")
        
        # 1. Handle any remaining problematic values
        df_clean = df_pandas.copy()
        
        # 2. Replace any remaining problematic string values
        string_cols = df_clean.select_dtypes(include=['object']).columns
        for col in string_cols:
            # Replace any remaining problematic values
            df_clean[col] = df_clean[col].replace(['inf', '-inf', 'infinity', '-infinity'], 'Unknown')
            # Ensure no None strings
            df_clean[col] = df_clean[col].fillna('Unknown')
        
        # 3. Handle numeric columns - ensure no inf values
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
            # Fill remaining NaN with reasonable defaults
            if col in ['AREA', 'PERIMETER', 'Shape_Area', 'Shape_Leng']:
                df_clean[col] = df_clean[col].fillna(0.0)
            else:
                df_clean[col] = df_clean[col].fillna(0)
        
        # 4. Create explicit schema to avoid inference issues
        from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
        
        schema_fields = []
        
        # Define schema based on actual columns present
        for col in df_clean.columns:
            if col in ['OBJECTID_1', 'OBJECTID']:
                schema_fields.append(StructField(col, IntegerType(), True))
            elif col in ['PERIMETER', 'AREA', 'Shape_Leng', 'Shape_Area']:
                schema_fields.append(StructField(col, DoubleType(), True))
            else:
                schema_fields.append(StructField(col, StringType(), True))
        
        schema = StructType(schema_fields)
        
        script_logger.info(f"Created schema with {len(schema_fields)} fields")
        
        # 5. Convert data types to match schema
        for field in schema.fields:
            col_name = field.name
            if col_name in df_clean.columns:
                if isinstance(field.dataType, IntegerType):
                    df_clean[col_name] = df_clean[col_name].astype('Int64')  # Nullable integer
                elif isinstance(field.dataType, DoubleType):
                    df_clean[col_name] = df_clean[col_name].astype('float64')
                else:  # StringType
                    df_clean[col_name] = df_clean[col_name].astype('str')
        
        # 6. Sample the data for testing if it's very large
        if len(df_clean) > 10000:
            script_logger.warning(f"Large dataset detected ({len(df_clean)} rows). Consider sampling for testing.")
            # Uncomment the next line if you want to sample for testing
            # df_clean = df_clean.sample(n=5000, random_state=42)
        
        # 7. Create Spark DataFrame with explicit schema
        df_spark = spark.createDataFrame(df_clean, schema=schema)
        
        # 8. Immediately cache and force evaluation to catch any issues early
        df_spark = df_spark.cache()
        count = df_spark.count()  # Force evaluation
        script_logger.info(f"Successfully created Spark DataFrame with {count} rows")
        
        # 9. Repartition for better performance (optional)
        optimal_partitions = max(1, min(8, count // 1000))  # Roughly 1000 rows per partition, max 8 partitions
        if optimal_partitions > 1:
            df_spark = df_spark.repartition(optimal_partitions)
            script_logger.info(f"Repartitioned DataFrame to {optimal_partitions} partitions")
        
        return df_spark
        
    except Exception as e:
        script_logger.error(f"Error creating Spark DataFrame: {e}")
        script_logger.error(f"DataFrame shape: {df_pandas.shape}")
        script_logger.error(f"DataFrame dtypes: {df_pandas.dtypes}")
        raise

def task_a_spark_sql_queries(spark, df_spark):
    """Task A: Perform Spark SQL queries with error handling"""
    script_logger.info("=== TASK A: SPARK SQL QUERIES ===")
    
    # Register DataFrame as temporary view
    df_spark.createOrReplaceTempView("military_bases")
    script_logger.info("Registered DataFrame as temporary view 'military_bases'")
    
    try:
        # First, let's see what data we have
        script_logger.info("Checking data structure...")
        spark.sql("SELECT COUNT(*) as total_count FROM military_bases").show()
        spark.sql("SELECT DISTINCT COMPONENT FROM military_bases LIMIT 10").show()
        
        # Query 1: Simplified aggregation to avoid worker crashes
        script_logger.info("\n--- Query 1: Average area by component (simplified) ---")
        query1 = """
        SELECT 
            COMPONENT,
            COUNT(*) as base_count,
            ROUND(AVG(CAST(AREA as DOUBLE)), 2) as avg_area,
            ROUND(MAX(CAST(AREA as DOUBLE)), 2) as max_area,
            ROUND(MIN(CAST(AREA as DOUBLE)), 2) as min_area
        FROM (
            SELECT * FROM military_bases 
            WHERE AREA IS NOT NULL 
            AND COMPONENT IS NOT NULL 
            AND CAST(AREA AS DOUBLE) IS NOT NULL
        )
        WHERE AREA IS NOT NULL AND COMPONENT IS NOT NULL
        GROUP BY COMPONENT
        ORDER BY avg_area DESC
        """
        
        result1 = spark.sql(query1)
        script_logger.info("Query 1 results:")
        rows = result1.collect()
        for row in rows:
            print(row)

    except Exception as e:
        script_logger.error(f"Error in Query 1: {e}")
        result1 = None
    
    try:
        # Query 2: Even simpler query to avoid complexity
        script_logger.info("\n--- Query 2: Basic filtering and case statements ---")
        query2 = """
        SELECT 
            Site_Name,
            COMPONENT,
            CAST(AREA as DOUBLE) as area_numeric,
            CASE 
                WHEN CAST(AREA as DOUBLE) > 250 THEN 'Very Large'
                WHEN CAST(AREA as DOUBLE) > 150 THEN 'Large'
                WHEN CAST(AREA as DOUBLE) > 100 THEN 'Medium'
                ELSE 'Small'
            END as size_category
        FROM military_bases 
        WHERE AREA IS NOT NULL 
            AND CAST(AREA as DOUBLE) > 100 
            AND Oper_Stat = 'Active'
        ORDER BY CAST(AREA as DOUBLE) DESC
        LIMIT 20
        """
        
        result2 = spark.sql(query2)
        script_logger.info("Query 2 results:")
        rows = result2.collect()  # Collect first to ensure it works
        for row in rows:
            print(row)
        
    except Exception as e:
        script_logger.error(f"Error in Query 2: {e}")
        result2 = None
    
    return result1, result2

def task_b_dataframe_operations(spark, df_spark):
    """Task B: Use PySpark DataFrame functions"""
    script_logger.info("\n=== TASK B: PYSPARK DATAFRAME OPERATIONS ===")
    
    try:
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
        try:
            filtered_df.select("Site_Name", "COMPONENT", "AREA", "area_category", 
                              "area_efficiency", "is_joint_base", "base_priority").show(5, truncate=False)
        except Exception as e:
            script_logger.error(f"Error showing filtered data: {e}")
            # Try showing just basic info
            script_logger.info(f"Filtered dataset has {filtered_df.count()} rows")
        
        # B2: Grouping and aggregation with safer approach
        script_logger.info("\n--- B2: Grouping and Aggregation ---")
        
        try:
            # Check if Region column exists and has non-null values
            region_count = df_spark.filter(col("Region").isNotNull()).count()
            script_logger.info(f"Records with non-null Region: {region_count}")
            
            if region_count > 0:
                aggregated_df = df_spark.groupBy("COMPONENT", "Region").agg(
                    count("*").alias("base_count"),
                    round(avg("AREA"), 2).alias("avg_area"),
                    round(avg("PERIMETER"), 2).alias("avg_perimeter"),
                    max("AREA").alias("max_area"),
                    min("AREA").alias("min_area"),
                    round(stddev("AREA"), 2).alias("area_std_dev"),
                    round(sum("AREA"), 2).alias("total_area")
                ).orderBy(desc("total_area"))
            else:
                # Fallback: Group by COMPONENT only
                script_logger.warning("No valid Region data found, grouping by COMPONENT only")
                aggregated_df = df_spark.groupBy("COMPONENT").agg(
                    count("*").alias("base_count"),
                    round(avg("AREA"), 2).alias("avg_area"),
                    round(avg("PERIMETER"), 2).alias("avg_perimeter"),
                    max("AREA").alias("max_area"),
                    min("AREA").alias("min_area"),
                    round(stddev("AREA"), 2).alias("area_std_dev"),
                    round(sum("AREA"), 2).alias("total_area")
                ).orderBy(desc("total_area"))
            
            script_logger.info("Grouped and aggregated data:")
            # Use limit to avoid potential memory issues
            aggregated_df.limit(10).show(truncate=False)
            
        except Exception as e:
            script_logger.error(f"Error in aggregation: {e}")
            # Create a simple fallback aggregation
            aggregated_df = df_spark.groupBy("COMPONENT").count().orderBy(desc("count"))
            script_logger.info("Fallback aggregation - count by component:")
            aggregated_df.show()
        
        # B3: Join operations with safer approach
        script_logger.info("\n--- B3: Join Operations ---")
        
        try:
            # Create a simpler regional dataset that matches what we actually have
            unique_regions = [row['Region'] for row in df_spark.select("Region").distinct().collect()]
            script_logger.info(f"Unique regions in data: {unique_regions}")
            
            # Create regional data that matches our actual regions
            regional_data = []
            for region in unique_regions:
                if region == "West":
                    regional_data.append((region, "Pacific", 5, 850000))
                elif region == "South":
                    regional_data.append((region, "Atlantic/Gulf", 8, 1200000))
                elif region == "Midwest":
                    regional_data.append((region, "Great Lakes", 2, 180000))
                elif region == "Northeast":
                    regional_data.append((region, "Atlantic", 1, 95000))
                else:
                    regional_data.append((region, "Unknown", 1, 100000))
            
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
                           "base_density_score").limit(10).show(truncate=False)
            
        except Exception as e:
            script_logger.error(f"Error in join operations: {e}")
            # Return the original dataframe as fallback
            final_df = df_spark
        
        return filtered_df, aggregated_df, final_df
        
    except Exception as e:
        script_logger.error(f"Error in task_b_dataframe_operations: {e}")
        # Return the original dataframe for all outputs as fallback
        return df_spark, df_spark, df_spark

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
    """Main function to execute all tasks with enhanced data cleanup"""
    script_logger.info("Starting PySpark Military Bases Analysis with Enhanced Data Cleanup")
    
    # Initialize Spark with even more conservative settings
    spark = initialize_spark_robust()
    
    try:
        # Load data with comprehensive cleanup
        df_pandas = load_data_from_database()
        
        if df_pandas is None:
            script_logger.error("Failed to load data from database. Stopping execution.")
            return
        
        # Convert to Spark DataFrame with robust handling
        df_spark = create_spark_dataframe_robust(spark, df_pandas)
        
        # Show schema and sample data
        script_logger.info("\n--- DataFrame Schema ---")
        df_spark.printSchema()
        
        script_logger.info("\n--- Sample Data (First 3 rows) ---")
        df_spark.show(3, truncate=True)
        
        # Test basic operations before proceeding
        script_logger.info("Testing basic operations...")
        try:
            total_count = df_spark.count()
            script_logger.info(f"Total records: {total_count}")
            
            # Test a simple aggregation
            component_counts = df_spark.groupBy("COMPONENT").count().collect()
            script_logger.info(f"Component distribution: {[(row['COMPONENT'], row['count']) for row in component_counts]}")
            
        except Exception as e:
            script_logger.error(f"Basic operations test failed: {e}")
            # Try with even simpler operations
            try:
                script_logger.info("Trying simplified operations...")
                df_spark.select("Site_Name", "COMPONENT").limit(5).show()
            except Exception as e2:
                script_logger.error(f"Even simplified operations failed: {e2}")
                return
        
        # If we get here, basic operations work, proceed with tasks
        script_logger.info("Basic operations successful, proceeding with full analysis...")
        
        # Execute Task A: Spark SQL queries
        result1, result2 = task_a_spark_sql_queries(spark, df_spark)
        
        # Execute Task B: DataFrame operations
        filtered_df, aggregated_df, joined_df = task_b_dataframe_operations(spark, df_spark)
        
        # Generate summary statistics
        generate_summary_statistics(df_spark)
        
        script_logger.info("\n=== ANALYSIS COMPLETE ===")
        script_logger.info("All tasks completed successfully!")
        
    except Exception as e:
        script_logger.error(f"An error occurred during analysis: {e}", exc_info=True)
        
    finally:
        # Clean up Spark session  
        spark.stop()
        script_logger.info("Spark session stopped")

def part2_old():
    """Main function to execute all tasks"""
    script_logger.info("Starting PySpark Military Bases Analysis")
    
    # Initialize Spark
    spark = initialize_spark_robust()
    
    try:
        # Try to load data from database first
        df_pandas = load_data_from_database()
        
        # If database loading fails, use sample data
        if df_pandas is None:
            script_logger.info("Error loading data from database. Stopping execution.")
            raise ValueError("Failed to load data from database. Stopping execution.")
        
        # Convert to Spark DataFrame
        from pyspark.sql.types import StringType, StructType, IntegerType, DoubleType, StructField

        schema = StructType([
            StructField("Geo_Point", StringType(), True),
            StructField("Geo_Shape", StringType(), True),
            StructField("OBJECTID_1", IntegerType(), True),
            StructField("OBJECTID", IntegerType(), True),
            StructField("COMPONENT", StringType(), True),
            StructField("Site_Name", StringType(), True),
            StructField("Joint_Base", StringType(), True),
            StructField("State_Terr", StringType(), True),
            StructField("COUNTRY", StringType(), True),
            StructField("Oper_Stat", StringType(), True),
            StructField("PERIMETER", DoubleType(), True),
            StructField("AREA", DoubleType(), True),
            StructField("Shape_Leng", DoubleType(), True),
            StructField("Shape_Area", DoubleType(), True),
            StructField("Region", StringType(), True),
        ])

        df_spark = spark.createDataFrame(df_pandas, schema=schema)
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
    #part2()