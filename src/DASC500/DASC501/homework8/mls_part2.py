import os
import sqlite3
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Import from your provided files (keep your existing imports)
# from DASC500.classes.DatabaseManager import DatabaseManager
# from DASC500.utilities.get_top_level_module import get_top_level_module_path

# --- Configuration (using your existing config) ---
# FOLDER = os.path.join(get_top_level_module_path(), '../..')
# DB_PATH = os.path.join(FOLDER, "data/DASC501/homework6/DataViz501.db")

# Simplified paths for demonstration - adjust these to your actual paths
from DASC500.utilities.get_top_level_module import get_top_level_module_path

FOLDER = os.path.join(get_top_level_module_path(), '../..')
DB_PATH = os.path.join(FOLDER, "data/DASC501/homework6/DataViz501.db")
TABLE_NAME = "military-bases"  # Your existing table name
COLUMNS_TO_ANALYZE = ['PERIMETER', 'AREA', 'Shape_Leng', 'Shape_Area']

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
script_logger = logging.getLogger(__name__)

class DatabaseManager:
    """Simplified DatabaseManager for this demonstration"""
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None
    
    def __enter__(self):
        self.conn = sqlite3.connect(self.db_path)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()
    
    def execute_select_query(self, query):
        try:
            return pd.read_sql_query(query, self.conn)
        except Exception as e:
            script_logger.error(f"Database query failed: {e}")
            return None

def load_and_clean_data():
    """Load and clean data from your existing database"""
    script_logger.info("Loading data from database...")
    
    db_manager = DatabaseManager(db_path=DB_PATH)
    df = None
    
    try:
        with db_manager:
            script_logger.info(f"Loading table '{TABLE_NAME}' from '{DB_PATH}'")
            df = db_manager.execute_select_query(f'SELECT * FROM `{TABLE_NAME}`')
        
        if df is None or df.empty:
            script_logger.error(f"Failed to load data from table '{TABLE_NAME}' or table is empty")
            return None
        
        script_logger.info(f"Successfully loaded {len(df)} rows from table '{TABLE_NAME}'")
        script_logger.info(f"DataFrame columns: {df.columns.tolist()}")
        
        # Clean data using your existing cleanup logic
        df = clean_military_bases_data(df)
        
        return df
        
    except Exception as e:
        script_logger.error(f"Error loading data from database: {e}")
        return None

def clean_military_bases_data(df):
    """Clean the military bases data using your existing cleanup logic"""
    script_logger.info("Cleaning military bases data...")
    
    # Drop ID column if it exists
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])
        script_logger.info("Dropped 'ID' column from DataFrame")
    
    # Clean column names - replace spaces with underscores
    original_columns = df.columns.tolist()
    df.columns = df.columns.str.replace(' ', '_', regex=False)
    df.columns = df.columns.str.replace('-', '_', regex=False)
    df.columns = df.columns.str.strip()
    
    # Clean string columns
    string_columns = df.select_dtypes(include=['object']).columns
    for col in string_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace(['nan', 'None', '', '  ', 'null', 'NULL'], np.nan)
    
    # Clean numeric columns
    numeric_columns = ['OBJECTID_1', 'OBJECTID', 'PERIMETER', 'AREA', 'Shape_Leng', 'Shape_Area']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    
    # Add Region column using your existing logic
    def assign_region(state):
        if pd.isna(state) or state == 'nan' or state == '':
            return "Unknown"
        
        state = str(state).strip()
        
        west_states = ['CA', 'California', 'OR', 'Oregon', 'WA', 'Washington', 'NV', 'Nevada', 
                      'ID', 'Idaho', 'UT', 'Utah', 'AZ', 'Arizona', 'MT', 'Montana', 
                      'WY', 'Wyoming', 'CO', 'Colorado', 'NM', 'New Mexico', 'AK', 'Alaska', 'HI', 'Hawaii']
        
        south_states = ['TX', 'Texas', 'OK', 'Oklahoma', 'AR', 'Arkansas', 'LA', 'Louisiana',
                       'MS', 'Mississippi', 'AL', 'Alabama', 'TN', 'Tennessee', 'KY', 'Kentucky',
                       'WV', 'West Virginia', 'VA', 'Virginia', 'NC', 'North Carolina', 
                       'SC', 'South Carolina', 'GA', 'Georgia', 'FL', 'Florida', 'DE', 'Delaware',
                       'MD', 'Maryland', 'DC', 'District of Columbia']
        
        midwest_states = ['ND', 'North Dakota', 'SD', 'South Dakota', 'NE', 'Nebraska', 'KS', 'Kansas',
                         'MN', 'Minnesota', 'IA', 'Iowa', 'MO', 'Missouri', 'WI', 'Wisconsin',
                         'IL', 'Illinois', 'IN', 'Indiana', 'MI', 'Michigan', 'OH', 'Ohio']
        
        northeast_states = ['ME', 'Maine', 'NH', 'New Hampshire', 'VT', 'Vermont', 'MA', 'Massachusetts',
                           'RI', 'Rhode Island', 'CT', 'Connecticut', 'NY', 'New York', 'NJ', 'New Jersey',
                           'PA', 'Pennsylvania']
        
        if any(state.lower() == s.lower() for s in west_states):
            return "West"
        elif any(state.lower() == s.lower() for s in south_states):
            return "South"
        elif any(state.lower() == s.lower() for s in midwest_states):
            return "Midwest"
        elif any(state.lower() == s.lower() for s in northeast_states):
            return "Northeast"
        else:
            return "Other"
    
    # Find state column
    state_column = None
    possible_state_columns = ['State_Terr', 'State', 'STATE', 'state', 'Territory', 'TERRITORY']
    for col in possible_state_columns:
        if col in df.columns:
            state_column = col
            break
    
    if state_column:
        df['Region'] = df[state_column].apply(assign_region)
        script_logger.info(f"Added Region column based on {state_column}")
    else:
        df['Region'] = "Unknown"
        script_logger.warning("No state column found, assigned 'Unknown' to all regions")
    
    script_logger.info(f"Data cleaning complete. Final shape: {df.shape}")
    return df

def task_a_sql_queries_with_sqlite(df):
    """Task A: SQL queries using SQLite (equivalent to Spark SQL)"""
    script_logger.info("\n=== TASK A: SQL QUERIES USING SQLITE ===")
    
    # Create a temporary SQLite connection and load the DataFrame
    conn = sqlite3.connect(':memory:')
    df.to_sql('military_bases', conn, index=False, if_exists='replace')
    
    try:
        # Query 1: Grouping and aggregation
        script_logger.info("\n--- Query 1: Average area by component with aggregation ---")
        
        query1 = """
        SELECT 
            COMPONENT,
            COUNT(*) as base_count,
            ROUND(AVG(CAST(AREA as REAL)), 2) as avg_area,
            ROUND(MAX(CAST(AREA as REAL)), 2) as max_area,
            ROUND(MIN(CAST(AREA as REAL)), 2) as min_area,
            ROUND(AVG(CAST(PERIMETER as REAL)), 2) as avg_perimeter
        FROM military_bases 
        WHERE AREA IS NOT NULL 
            AND COMPONENT IS NOT NULL 
            AND CAST(AREA AS REAL) > 0
        GROUP BY COMPONENT
        ORDER BY avg_area DESC
        """
        
        result1 = pd.read_sql_query(query1, conn)
        print("Query 1 Results - Component Statistics:")
        print(result1)
        
    except Exception as e:
        script_logger.error(f"Error in Query 1: {e}")
        result1 = None
    
    try:
        # Query 2: Complex filtering with CASE WHEN
        script_logger.info("\n--- Query 2: Complex filtering with CASE statements ---")
        
        query2 = """
        SELECT 
            Site_Name,
            COMPONENT,
            CAST(AREA as REAL) as area_numeric,
            CASE 
                WHEN CAST(AREA as REAL) > 250 THEN 'Very Large'
                WHEN CAST(AREA as REAL) > 150 THEN 'Large'
                WHEN CAST(AREA as REAL) > 100 THEN 'Medium'
                ELSE 'Small'
            END as size_category,
            Region,
            Oper_Stat
        FROM military_bases 
        WHERE AREA IS NOT NULL 
            AND CAST(AREA as REAL) > 100 
            AND Oper_Stat = 'Active'
        ORDER BY CAST(AREA as REAL) DESC
        LIMIT 15
        """
        
        result2 = pd.read_sql_query(query2, conn)
        print("\nQuery 2 Results - Filtered and Categorized Active Bases:")
        print(result2)
        
    except Exception as e:
        script_logger.error(f"Error in Query 2: {e}")
        result2 = None
    
    # Query 3: Regional analysis with complex aggregation
    try:
        script_logger.info("\n--- Query 3: Regional analysis with complex aggregation ---")
        
        query3 = """
        SELECT 
            Region,
            COUNT(*) as total_bases,
            COUNT(CASE WHEN Oper_Stat = 'Active' THEN 1 END) as active_bases,
            ROUND(AVG(CAST(AREA as REAL)), 2) as avg_area,
            ROUND(SUM(CAST(AREA as REAL)), 2) as total_area,
            ROUND(COUNT(CASE WHEN Oper_Stat = 'Active' THEN 1 END) * 100.0 / COUNT(*), 1) as active_percentage
        FROM military_bases 
        WHERE AREA IS NOT NULL AND Region IS NOT NULL
        GROUP BY Region
        ORDER BY total_area DESC
        """
        
        result3 = pd.read_sql_query(query3, conn)
        print("\nQuery 3 Results - Regional Analysis:")
        print(result3)
        
    except Exception as e:
        script_logger.error(f"Error in Query 3: {e}")
        result3 = None
    
    conn.close()
    return result1, result2, result3

def task_b_dataframe_operations(df):
    """Task B: DataFrame operations using pandas (equivalent to PySpark DataFrame functions)"""
    script_logger.info("\n=== TASK B: DATAFRAME OPERATIONS (PANDAS EQUIVALENT TO PYSPARK) ===")
    
    # B1: Filtering and transformation on columns
    script_logger.info("\n--- B1: Filtering and Transformation ---")
    
    # Filter active bases with area > 100 (equivalent to PySpark .filter())
    filtered_df = df[(df['Oper_Stat'] == 'Active') & (df['AREA'] > 100)].copy()
    
    # Add transformed columns (equivalent to PySpark .withColumn())
    filtered_df['area_category'] = filtered_df['AREA'].apply(
        lambda x: 'Very Large' if x > 250 else 
                 'Large' if x > 150 else 
                 'Medium' if x > 100 else 'Small'
    )
    
    filtered_df['area_efficiency'] = (filtered_df['AREA'] / filtered_df['PERIMETER']).round(3)
    filtered_df['is_joint_base'] = filtered_df['COMPONENT'] == 'Joint'
    
    # Add priority ranking
    def assign_priority(row):
        if row['COMPONENT'] == 'Joint':
            return 1
        elif row['area_category'] == 'Very Large':
            return 2
        elif row['COMPONENT'] in ['Army', 'Navy', 'Air Force']:
            return 3
        else:
            return 4
    
    filtered_df['base_priority'] = filtered_df.apply(assign_priority, axis=1)
    
    print("B1 Results - Filtered and Transformed Data (first 10 rows):")
    display_cols = ['Site_Name', 'COMPONENT', 'AREA', 'area_category', 'area_efficiency', 'is_joint_base', 'base_priority']
    available_cols = [col for col in display_cols if col in filtered_df.columns]
    print(filtered_df[available_cols].head(10))
    
    # B2: Grouping and aggregation (equivalent to PySpark .groupBy())
    script_logger.info("\n--- B2: Grouping and Aggregation ---")
    
    try:
        aggregated_df = df.groupby(['COMPONENT', 'Region']).agg({
            'AREA': ['count', 'mean', 'max', 'min', 'std', 'sum'],
            'PERIMETER': 'mean'
        }).round(2)
        
        # Flatten column names
        aggregated_df.columns = ['base_count', 'avg_area', 'max_area', 'min_area', 'area_std_dev', 'total_area', 'avg_perimeter']
        aggregated_df = aggregated_df.sort_values('total_area', ascending=False)
        
        print("B2 Results - Grouped and Aggregated Data:")
        print(aggregated_df.head(10))
        
    except Exception as e:
        script_logger.error(f"Error in grouping: {e}")
        # Fallback to simpler grouping
        aggregated_df = df.groupby('COMPONENT').agg({
            'AREA': ['count', 'mean', 'max']
        }).round(2)
        print("B2 Fallback Results - Simple Component Grouping:")
        print(aggregated_df)
    
    # B3: Join operations (equivalent to PySpark .join())
    script_logger.info("\n--- B3: Join Operations ---")
    
    # Create regional supplementary data
    regional_data = pd.DataFrame({
        'Region': ['West', 'South', 'Northeast', 'Midwest', 'Other'],
        'Coast_Type': ['Pacific', 'Atlantic/Gulf', 'Atlantic', 'Great Lakes', 'None'],
        'Region_Population': [850000, 1200000, 95000, 180000, 50000],
        'Region_Base_Count': [5, 8, 1, 2, 1]
    })
    
    # Perform inner join (equivalent to PySpark DataFrame.join())
    joined_df = pd.merge(df, regional_data, on='Region', how='inner')
    
    # Add calculated columns after join
    joined_df['population_per_base_ratio'] = (
        joined_df['Region_Population'] / joined_df['Region_Base_Count']
    ).round(0)
    
    joined_df['base_density_score'] = (
        joined_df['AREA'] / joined_df['Region_Population'] * 1000000
    ).round(4)
    
    print("B3 Results - Joined Data with Calculated Metrics (first 10 rows):")
    join_display_cols = ['Site_Name', 'COMPONENT', 'Region', 'Coast_Type', 'AREA', 'Region_Population', 'population_per_base_ratio', 'base_density_score']
    available_join_cols = [col for col in join_display_cols if col in joined_df.columns]
    print(joined_df[available_join_cols].head(10))
    
    return filtered_df, aggregated_df, joined_df

def generate_summary_statistics(df):
    """Generate comprehensive summary statistics"""
    script_logger.info("\n=== SUMMARY STATISTICS ===")
    
    print("\n--- Basic Dataset Statistics ---")
    print(f"Total number of military bases: {len(df)}")
    
    if 'COMPONENT' in df.columns:
        print("\nComponent Distribution:")
        print(df['COMPONENT'].value_counts())
    
    if 'Oper_Stat' in df.columns:
        print("\nOperational Status Distribution:")
        print(df['Oper_Stat'].value_counts())
    
    if 'Region' in df.columns:
        print("\nRegional Distribution:")
        print(df['Region'].value_counts())
    
    print("\n--- Numerical Column Statistics ---")
    numerical_cols = [col for col in COLUMNS_TO_ANALYZE if col in df.columns]
    if numerical_cols:
        numerical_stats = df[numerical_cols].describe()
        print(numerical_stats.round(2))
    
    # Additional insights
    print("\n--- Additional Insights ---")
    if 'Region' in df.columns and 'AREA' in df.columns:
        avg_area_by_region = df.groupby('Region')['AREA'].mean().sort_values(ascending=False)
        print("Average Area by Region:")
        print(avg_area_by_region.round(2))
    
    if 'AREA' in df.columns:
        largest_bases_cols = ['Site_Name', 'COMPONENT', 'Region', 'AREA']
        available_largest_cols = [col for col in largest_bases_cols if col in df.columns]
        largest_bases = df.nlargest(5, 'AREA')[available_largest_cols]
        print("\nTop 5 Largest Bases:")
        print(largest_bases)

def main():
    """Main function to execute the complete analysis"""
    script_logger.info("Starting Military Bases Analysis using SQLite + Pandas")
    
    try:
        # Load and clean data from your existing database
        df = load_and_clean_data()
        
        if df is None:
            script_logger.error("Failed to load data. Exiting.")
            return
        
        print("\n" + "="*80)
        print("MILITARY BASES ANALYSIS - SQLITE + PANDAS")
        print("="*80)
        
        print(f"\nDataset Overview:")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Execute Task A: SQL queries using SQLite
        result1, result2, result3 = task_a_sql_queries_with_sqlite(df)
        
        # Execute Task B: DataFrame operations using pandas
        filtered_df, aggregated_df, joined_df = task_b_dataframe_operations(df)
        
        # Generate summary statistics
        generate_summary_statistics(df)
        
        script_logger.info("\n=== ANALYSIS COMPLETE ===")
        script_logger.info("All tasks completed successfully!")
        
        # Optional: Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"military_analysis_{timestamp}"
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            df.to_csv(f"{output_dir}/cleaned_military_data.csv", index=False)
            if result1 is not None:
                result1.to_csv(f"{output_dir}/component_statistics.csv", index=False)
            if joined_df is not None:
                joined_df.to_csv(f"{output_dir}/joined_analysis.csv", index=False)
            print(f"\nResults saved to directory: {output_dir}")
        except Exception as e:
            script_logger.warning(f"Could not save results: {e}")
        
        return {
            'df': df,
            'sql_results': (result1, result2, result3),
            'dataframe_results': (filtered_df, aggregated_df, joined_df)
        }
        
    except Exception as e:
        script_logger.error(f"An error occurred during analysis: {e}", exc_info=True)

if __name__ == "__main__":
    # Update DB_PATH to point to your actual database file
    # DB_PATH = "path/to/your/DataViz501.db"  # Update this line
    
    results = main()