import os
import pandas as pd
import numpy as np
import sqlite3 
import re
import logging 

from DASC500.utilities.get_top_level_module import get_top_level_module_path
from DASC500.utilities.print.redirect_print import redirect_print

# Set pathing for input csv files and output directory
TOP_PATH = os.path.join(get_top_level_module_path(), "../..")
DATA_PATH = os.path.join(TOP_PATH, "data/DASC501/homework7")
ucs_satellite_db_path = os.path.join(DATA_PATH, "UCS-Satellite-Database-1-1-2023.csv")
military_bases_db_path = os.path.join(DATA_PATH, "military-bases.csv")
OUTPUT_PATH = os.path.join(TOP_PATH, "outputs/DASC501/homework7")

redirect_print(
    os.path.join(OUTPUT_PATH, "mls_homework7_output.txt"), also_to_stdout=True
)

# Configure basic logging for the script
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Question 1: Read CSV files ---
print("--- Question 1: Reading CSV files ---")

# Load the satellite database
try:
    # UCS-Satellite-Database-1-1-2023.csv
    satellite_df = pd.read_csv(ucs_satellite_db_path)
    logging.info("Satellite Database successfully loaded.")
except Exception as e:
    logging.error(f"Error loading satellite database: {e}")
    try:
        logging.info(
            "Attempting to load satellite database with ISO-8859-1 encoding..."
        )
        satellite_df = pd.read_csv(ucs_satellite_db_path, encoding="ISO-8859-1")
        logging.info("Satellite Database successfully loaded with ISO-8859-1 encoding.")
    except Exception as e_iso:
        logging.error(
            f"Error loading satellite database with ISO-8859-1 encoding: {e_iso}"
        )
        # Later try not needed but demonstrates handling of a more complicated file
        try:
            logging.info(
                "Attempting to load satellite database by skipping bad lines..."
            )
            satellite_df = pd.read_csv(
                ucs_satellite_db_path, encoding="ISO-8859-1", on_bad_lines="skip"
            )
            logging.info(
                "Satellite Database successfully loaded with ISO-8859-1 encoding and skipping bad lines."
            )
        except Exception as e_skip:
            logging.error(f"Could not load satellite database: {e_skip}")
            satellite_df = pd.DataFrame()

# Load the military bases database (this one is simpler and does not require encoding)
try:
    military_bases_df = pd.read_csv(military_bases_db_path, delimiter=";")
    logging.info("Military Bases Database successfully loaded.")
except Exception as e:
    logging.error(f"Error loading military bases database: {e}")
    military_bases_df = pd.DataFrame()

# Check the loaded DataFrames
# Inspect the satellite_df
if not satellite_df.empty:
    print("\n--- Satellite Database Info ---")
    print("Head:")
    print(satellite_df.head())
    print("\nInfo:")
    satellite_df.info()
    print("\nColumns before cleaning:")
    print(satellite_df.columns)
    original_satellite_columns = satellite_df.columns
    satellite_df.columns = [
        re.sub(r"\W+", "_", col).lower().strip("_") for col in satellite_df.columns
    ]
    print("\nCleaned Satellite DF Columns:")
    print(satellite_df.columns)

# Inspect the military_bases_df
if not military_bases_df.empty:
    print("\n--- Military Bases Database Info ---")
    print("Head:")
    print(military_bases_df.head())
    print("\nInfo:")
    military_bases_df.info()
    print("\nColumns before cleaning:")
    print(military_bases_df.columns)
    original_military_columns = military_bases_df.columns
    military_bases_df.columns = [
        re.sub(r"\W+", "_", col).lower().strip("_") for col in military_bases_df.columns
    ]
    print("\nCleaned Military Bases DF Columns:")
    print(military_bases_df.columns)


# Database Manager (simplified this class for this example)
class DatabaseManager:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None
        self.logger = logging.getLogger(__name__)  # Uses the logging imported above
        # self.logger.setLevel(logging.INFO) # Ensure logger level is set if not using basicConfig root

    def connect(self):
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.logger.info(f"Connected to database: {self.db_path}")
        except sqlite3.Error as e:
            self.logger.error(f"Error connecting to database {self.db_path}: {e}")
            raise

    def disconnect(self):
        if self.conn:
            self.conn.close()
            self.conn = None  # Set to None after closing
            self.logger.info(f"Disconnected from database: {self.db_path}")

    def execute_query(self, query, params=None):
        if not self.conn:
            self.connect()
        try:
            cursor = self.conn.cursor()
            cursor.execute(query, params or ())
            self.conn.commit()
            return cursor
        except sqlite3.Error as e:
            self.logger.error(
                f"Error executing query: {query} with params {params}. Error: {e}"
            )
            raise

    def fetch_query(self, query, params=None):
        if not self.conn:
            self.connect()
        try:
            cursor = self.conn.cursor()
            cursor.execute(query, params or ())
            return cursor.fetchall()
        except sqlite3.Error as e:
            self.logger.error(
                f"Error fetching query: {query} with params {params}. Error: {e}"
            )
            raise

    def df_to_table(self, df, table_name, if_exists="replace", index=False):
        if not self.conn:
            self.connect()
        try:
            # Clean column names for SQL compatibility if not already done
            clean_cols = [
                re.sub(r"\W+", "_", col).lower().strip("_") for col in df.columns
            ]
            df_copy = df.copy()
            df_copy.columns = clean_cols
            df_copy.to_sql(table_name, self.conn, if_exists=if_exists, index=index)
            self.logger.info(
                f"DataFrame successfully loaded into table '{table_name}'. Cleaned columns: {clean_cols}"
            )
        except Exception as e:  # Catch pandas or sqlite3 errors
            self.logger.error(f"Error loading DataFrame into table '{table_name}': {e}")
            raise

# Instantiate DatabaseManager for later problems
db_manager = DatabaseManager("analysis_database.sqlite")

# Fixed Question 2: Pivot Tables in Pandas vs. SQL (Satellite Data)
print("\n\n--- Question 2: Pivot Tables (Satellite Data) ---")
if not satellite_df.empty:
    # Column names after cleaning (refer to print(satellite_df.columns) output)
    name_col_sat = "name_of_satellite_alternate_names"
    country_col = "country_of_operator_owner"
    purpose_col = "purpose"
    detailed_purpose_col = "detailed_purpose"
    orbit_class_col = "class_of_orbit"
    mass_col = "launch_mass_kg"
    power_col = "power_watts"
    launch_date_col = "date_of_launch"
    lifetime_col = "expected_lifetime_yrs"

    required_q2_cols = [
        name_col_sat,
        country_col,
        purpose_col,
        detailed_purpose_col,
        orbit_class_col,
        mass_col,
        power_col,
        launch_date_col,
        lifetime_col,
    ]
    actual_sat_cols = satellite_df.columns
    missing_q2_cols = [col for col in required_q2_cols if col not in actual_sat_cols]

    if missing_q2_cols:
        logging.error(
            f"Q2: Missing critical satellite columns after cleaning: {missing_q2_cols}. Available: {actual_sat_cols}"
        )
    else:
        logging.info("Q2: All required satellite columns are present.")
        
        # Table 1: Basic satellite information
        satellite_info_df = satellite_df[
            [
                name_col_sat,
                country_col,
                purpose_col,
                detailed_purpose_col,
                orbit_class_col,
            ]
        ].copy()
        
        # Table 2: Technical specifications
        satellite_tech_df = satellite_df[
            [name_col_sat, mass_col, power_col, launch_date_col, lifetime_col]
        ].copy()

        # Clean numeric columns in technical specs table
        for col_to_clean in [mass_col, power_col]:
            if col_to_clean in satellite_tech_df.columns:
                satellite_tech_df[col_to_clean] = (
                    satellite_tech_df[col_to_clean]
                    .astype(str)
                    .str.replace(",", "", regex=False)
                )
                satellite_tech_df[col_to_clean] = pd.to_numeric(
                    satellite_tech_df[col_to_clean], errors="coerce"
                )

        # Merge the two tables
        merged_satellite_df = pd.merge(
            satellite_info_df, satellite_tech_df, on=name_col_sat, how="inner"
        )
        
        print("\n-- Pandas Implementation (Satellite) --")
        print("Satellite Info Table (head):")
        print(satellite_info_df.head())
        print("\nSatellite Technical Specs Table (head):")
        print(satellite_tech_df.head())
        print("\nMerged Satellite Table (head):")
        print(merged_satellite_df.head())

        try:
            # Create comprehensive pivot table with multiple aggregations
            satellite_pivot_pandas = pd.pivot_table(
                merged_satellite_df,
                index=[purpose_col, detailed_purpose_col],  # Two-level hierarchy
                columns=country_col,
                values=[name_col_sat, mass_col, power_col],  # Multiple values
                aggfunc={
                    name_col_sat: 'count',    # Count of satellites
                    mass_col: 'mean',         # Average launch mass
                    power_col: 'mean'         # Average power
                },
                fill_value=0
            )
            print("\nPandas Pivot Table (Satellite - Multi-level with Multiple Aggregations):")
            print(satellite_pivot_pandas.head(10))
            
            # Save to CSV
            satellite_pivot_pandas.to_csv(
                os.path.join(OUTPUT_PATH, "satellite_pivot_pandas.csv")
            )
            logging.info(
                "Satellite Pandas pivot table saved to 'satellite_pivot_pandas.csv'"
            )
        except Exception as e:
            logging.error(f"Error creating Pandas pivot table for satellites: {e}")

        print("\n-- SQL Implementation (Satellite) --")
        try:
            db_manager.connect()
            
            # Load data into SQL tables
            db_manager.df_to_table(
                satellite_info_df, "satellite_info", if_exists="replace"
            )
            db_manager.df_to_table(
                satellite_tech_df, "satellite_tech_specs", if_exists="replace"
            )
            logging.info("Satellite data loaded into SQL tables.")

            # SQL query to replicate the pivot table logic
            sql_query_satellite = f"""
            SELECT
                si."{purpose_col}", 
                si."{detailed_purpose_col}", 
                si."{country_col}",
                COUNT(si."{name_col_sat}") AS satellite_count,
                AVG(CAST(st."{mass_col}" AS FLOAT)) AS avg_launch_mass,
                AVG(CAST(st."{power_col}" AS FLOAT)) AS avg_power
            FROM satellite_info si
            JOIN satellite_tech_specs st ON si."{name_col_sat}" = st."{name_col_sat}"
            WHERE st."{mass_col}" IS NOT NULL 
                AND st."{power_col}" IS NOT NULL
                AND si."{country_col}" IS NOT NULL
            GROUP BY si."{purpose_col}", si."{detailed_purpose_col}", si."{country_col}"
            ORDER BY si."{purpose_col}", si."{detailed_purpose_col}", si."{country_col}";
            """
            
            logging.info("Executing SQL query for satellite pivot...")
            satellite_pivot_sql_results = db_manager.fetch_query(sql_query_satellite)

            if satellite_pivot_sql_results:
                # Convert SQL results to DataFrame
                satellite_pivot_sql_df = pd.DataFrame(
                    satellite_pivot_sql_results,
                    columns=[
                        purpose_col,
                        detailed_purpose_col,
                        country_col,
                        "satellite_count",
                        "avg_launch_mass",
                        "avg_power",
                    ],
                )
                
                # Create pivot table from SQL results to match Pandas format
                satellite_pivot_sql_pivoted = satellite_pivot_sql_df.pivot_table(
                    index=[purpose_col, detailed_purpose_col],
                    columns=country_col,
                    values=["satellite_count", "avg_launch_mass", "avg_power"],
                    fill_value=0
                )
                
                print("\nSQL Pivot Table (Satellite - Multi-level with Multiple Aggregations):")
                print(satellite_pivot_sql_pivoted.head(10))
                
                # Save SQL results
                satellite_pivot_sql_pivoted.to_csv(
                    os.path.join(OUTPUT_PATH, "satellite_pivot_sql.csv")
                )
                logging.info("SQL pivot table saved to 'satellite_pivot_sql.csv'")
            else:
                logging.info("SQL query for satellite data returned no results.")
                
        except Exception as e:
            logging.error(f"Error during SQL implementation for satellites: {e}")
        finally:
            db_manager.disconnect()

# Fixed Question 3: Advanced Group-by Operations (Military Base Data)
print("\n\n--- Question 3: Advanced Group-by (Military Base Data) ---")
if not military_bases_df.empty:
    site_name_col_mil = "site_name"
    state_terr_col = "state_terr"
    component_col = "component"
    joint_base_col = "joint_base"
    oper_stat_col = "oper_stat"
    area_col = "area"
    perimeter_col = "perimeter"
    shape_leng_col = "shape_leng"
    shape_area_col = "shape_area"

    required_q3_cols = [
        site_name_col_mil,
        state_terr_col,
        component_col,
        joint_base_col,
        oper_stat_col,
        area_col,
        perimeter_col,
        shape_leng_col,
        shape_area_col,
    ]
    actual_mil_cols = military_bases_df.columns
    missing_q3_cols = [col for col in required_q3_cols if col not in actual_mil_cols]

    if missing_q3_cols:
        logging.error(
            f"Q3: Missing critical military base columns after cleaning: {missing_q3_cols}. Available: {actual_mil_cols}"
        )
    else:
        logging.info("Q3: All required military base columns are present.")
        
        # Table 1: Basic base information
        base_info_df = military_bases_df[
            [
                site_name_col_mil,
                state_terr_col,
                component_col,
                joint_base_col,
                oper_stat_col,
            ]
        ].copy()
        
        # Table 2: Geographical and size metrics
        base_geo_df = military_bases_df[
            [site_name_col_mil, area_col, perimeter_col, shape_leng_col, shape_area_col]
        ].copy()

        # Clean numeric columns
        for col_to_num in [area_col, perimeter_col, shape_leng_col, shape_area_col]:
            if col_to_num in base_geo_df.columns:
                # Handle comma as decimal separator
                if base_geo_df[col_to_num].dtype == "object":
                    base_geo_df[col_to_num] = base_geo_df[col_to_num].str.replace(
                        ",", ".", regex=False
                    )
                base_geo_df[col_to_num] = pd.to_numeric(
                    base_geo_df[col_to_num], errors="coerce"
                )

        # Merge the two tables
        merged_bases_df = pd.merge(
            base_info_df, base_geo_df, on=site_name_col_mil, how="inner"
        )
        
        print("\n-- Pandas Implementation (Military Bases) --")
        print("Base Info Table (head):")
        print(base_info_df.head())
        print("\nBase Geographical Metrics Table (head):")
        print(base_geo_df.head())
        print("\nMerged Military Bases Table (head):")
        print(merged_bases_df.head())

        # Define strategic importance function
        def calculate_strategic_importance(group):
            """Calculate strategic importance score for a group of bases"""
            # Map operational status to numeric values
            oper_stat_mapping = {"ACTIVE": 1.0, "INACTIVE": 0.5, "UNKNOWN": 0.25}
            
            # Convert operational status to numeric and get mean
            oper_stat_numeric = group[oper_stat_col].map(oper_stat_mapping).fillna(0.1)
            mean_oper_stat = oper_stat_numeric.mean()
            
            # Get mean area and perimeter (handle NaN values)
            mean_area = group[area_col].mean() if group[area_col].notna().any() else 0
            mean_perimeter = group[perimeter_col].mean() if group[perimeter_col].notna().any() else 0
            
            # Calculate weighted strategic importance score
            # Normalize area and perimeter to similar scales for fair weighting
            normalized_area = mean_area / 1000 if mean_area > 0 else 0  # Divide by 1000 to normalize
            normalized_perimeter = mean_perimeter / 100 if mean_perimeter > 0 else 0  # Divide by 100 to normalize
            
            score = (
                (0.4 * normalized_area) +           # 40% weight on area
                (0.3 * normalized_perimeter) +      # 30% weight on perimeter  
                (0.3 * mean_oper_stat * 100)        # 30% weight on operational status (scaled up)
            )
            
            return score if pd.notna(score) else 0

        try:
            # Advanced groupby operations
            military_groupby_pandas = merged_bases_df.groupby([state_terr_col, component_col]).agg({
                site_name_col_mil: 'count',                    # Count of bases
                area_col: 'mean',                              # Average area
                shape_leng_col: 'std',                         # Standard deviation of Shape_Leng
                # Apply custom function to entire group for strategic importance
                oper_stat_col: lambda x: calculate_strategic_importance(
                    merged_bases_df[merged_bases_df.index.isin(x.index)]
                )
            }).round(4)
            
            # Rename columns for clarity
            military_groupby_pandas.columns = [
                'base_count',
                'average_area', 
                'std_dev_shape_length',
                'strategic_importance_score'
            ]
            
            military_groupby_pandas = military_groupby_pandas.reset_index()
            
            print("\nPandas Groupby Results (Military Bases):")
            print(military_groupby_pandas.head(10))
            
        except Exception as e:
            logging.error(f"Error during Pandas groupby for military bases: {e}")

        print("\n-- SQL Implementation (Military Bases) --")
        try:
            db_manager.connect()
            
            # Load data into SQL tables
            db_manager.df_to_table(base_info_df, "base_info", if_exists="replace")
            db_manager.df_to_table(base_geo_df, "base_geometrics", if_exists="replace")
            logging.info("Military base data loaded into SQL tables.")

            # Register custom aggregate function for standard deviation
            class StdevAgg:
                def __init__(self):
                    self.values = []

                def step(self, value):
                    if value is not None:
                        self.values.append(float(value))

                def finalize(self):
                    if len(self.values) < 2:
                        return 0
                    return np.std(self.values, ddof=1)

            db_manager.conn.create_aggregate("stdev", 1, StdevAgg)
            logging.info("Registered 'stdev' aggregate function in SQLite.")

            # SQL query with strategic importance calculation
            sql_query_military = f"""
            SELECT
                bi."{state_terr_col}", 
                bi."{component_col}",
                COUNT(bi."{site_name_col_mil}") AS base_count,
                AVG(CAST(bg."{area_col}" AS FLOAT)) AS average_area,
                STDEV(CAST(bg."{shape_leng_col}" AS FLOAT)) AS std_dev_shape_length,
                -- Strategic importance calculation
                (0.4 * AVG(COALESCE(CAST(bg."{area_col}" AS FLOAT), 0)) / 1000.0) + 
                (0.3 * AVG(COALESCE(CAST(bg."{perimeter_col}" AS FLOAT), 0)) / 100.0) + 
                (0.3 * AVG(
                    CASE UPPER(bi."{oper_stat_col}")
                        WHEN 'ACTIVE' THEN 1.0 
                        WHEN 'INACTIVE' THEN 0.5 
                        WHEN 'UNKNOWN' THEN 0.25 
                        ELSE 0.1
                    END
                ) * 100) AS strategic_importance_score
            FROM base_info bi 
            JOIN base_geometrics bg ON bi."{site_name_col_mil}" = bg."{site_name_col_mil}"
            GROUP BY bi."{state_terr_col}", bi."{component_col}"
            ORDER BY bi."{state_terr_col}", bi."{component_col}";
            """
            
            logging.info("Executing SQL query for military groupby...")
            military_groupby_sql_results = db_manager.fetch_query(sql_query_military)

            if military_groupby_sql_results:
                military_groupby_sql_df = pd.DataFrame(
                    military_groupby_sql_results,
                    columns=[
                        state_terr_col,
                        component_col,
                        "base_count",
                        "average_area",
                        "std_dev_shape_length",
                        "strategic_importance_score",
                    ],
                )
                print("\nSQL Groupby Results (Military Bases):")
                print(military_groupby_sql_df.head(10))
            else:
                logging.info("SQL query for military base data returned no results.")
                
        except Exception as e:
            logging.error(f"Error during SQL implementation for military bases: {e}")
        finally:
            db_manager.disconnect()

        # State Strategic Profile
        print("\n-- State Strategic Profile --")
        if 'military_groupby_pandas' in locals() and not military_groupby_pandas.empty:
            state_strategic_profile_pandas = (
                military_groupby_pandas.groupby(state_terr_col)
                .agg({
                    'base_count': 'sum',                        # Total bases per state
                    'average_area': 'mean',                     # Average of component averages
                    'std_dev_shape_length': 'mean',             # Average geographical spread
                    'strategic_importance_score': 'mean'        # Average strategic importance
                })
                .round(4)
                .reset_index()
            )
            
            # Rename columns for clarity
            state_strategic_profile_pandas.columns = [
                state_terr_col,
                'total_bases',
                'avg_base_area_state',
                'avg_geographical_spread',
                'avg_strategic_score_state'
            ]
            
            print("\nState Strategic Profile (from Pandas data):")
            print(state_strategic_profile_pandas.head())
            
            state_strategic_profile_pandas.to_csv(
                os.path.join(OUTPUT_PATH, "state_strategic_profile_pandas.csv"),
                index=False,
            )
            logging.info(
                "State strategic profile (Pandas) saved to 'state_strategic_profile_pandas.csv'"
            )

        if 'military_groupby_sql_df' in locals() and not military_groupby_sql_df.empty:
            state_strategic_profile_sql = (
                military_groupby_sql_df.groupby(state_terr_col)
                .agg({
                    'base_count': 'sum',
                    'average_area': 'mean',
                    'std_dev_shape_length': 'mean',
                    'strategic_importance_score': 'mean'
                })
                .round(4)
                .reset_index()
            )
            
            # Rename columns for clarity
            state_strategic_profile_sql.columns = [
                state_terr_col,
                'total_bases',
                'avg_base_area_state', 
                'avg_geographical_spread',
                'avg_strategic_score_state'
            ]
            
            print("\nState Strategic Profile (from SQL data):")
            print(state_strategic_profile_sql.head())
            
            state_strategic_profile_sql.to_csv(
                os.path.join(OUTPUT_PATH, "state_strategic_profile_sql.csv"),
                index=False,
            )
            logging.info(
                "State strategic profile (SQL) saved to 'state_strategic_profile_sql.csv'"
            )

# Fixed Question 4: Multi-Level Data Transformation and Pivoting (Satellite Data)
print("\n\n--- Question 4: Multi-Level Pivoting (Satellite Data) ---")
if not satellite_df.empty and name_col_sat in satellite_df.columns:
    logging.info("Starting Question 4: Multi-Level Pivoting (Satellite Data)")
    q4_sat_df = satellite_df.copy()

    if not all(
        col in q4_sat_df.columns for col in [mass_col, launch_date_col, country_col]
    ):
        logging.error(
            f"Q4: Missing columns for satellite advanced pivoting. Need: {mass_col}, {launch_date_col}, {country_col}. Available: {q4_sat_df.columns}"
        )
    else:
        # Clean mass column if needed
        if q4_sat_df[mass_col].dtype == "object":
            q4_sat_df[mass_col] = (
                q4_sat_df[mass_col].astype(str).str.replace(",", "", regex=False)
            )
            q4_sat_df[mass_col] = pd.to_numeric(q4_sat_df[mass_col], errors="coerce")

        # Create size categories based on launch mass
        def categorize_satellite_size(mass):
            if pd.isna(mass):
                return None
            elif mass < 500:
                return "Small"
            elif mass <= 2000:
                return "Medium"
            else:
                return "Large"

        q4_sat_df["size_category"] = q4_sat_df[mass_col].apply(categorize_satellite_size)
        
        print("\nSatellites with Size Category (sample):")
        size_sample = q4_sat_df[[name_col_sat, mass_col, "size_category"]].dropna(subset=[mass_col])
        print(size_sample.head(10))
        print(f"\nSize category distribution:")
        print(q4_sat_df["size_category"].value_counts())

        # Process launch dates and create decades
        q4_sat_df[launch_date_col] = pd.to_datetime(q4_sat_df[launch_date_col], errors="coerce")
        q4_sat_df["launch_year"] = q4_sat_df[launch_date_col].dt.year

        def categorize_decade(year):
            if pd.isna(year):
                return None
            elif 1990 <= year <= 1999:
                return "1990s"
            elif 2000 <= year <= 2009:
                return "2000s"
            elif 2010 <= year <= 2019:
                return "2010s"
            elif 2020 <= year <= 2029:
                return "2020s"
            else:
                return "Other"

        q4_sat_df["launch_decade"] = q4_sat_df["launch_year"].apply(categorize_decade)
        
        print("\nSatellites with Launch Decade (sample):")
        decade_sample = q4_sat_df[[name_col_sat, launch_date_col, "launch_year", "launch_decade"]].dropna(subset=["launch_year"])
        print(decade_sample.head(10))
        print(f"\nDecade distribution:")
        print(q4_sat_df["launch_decade"].value_counts())

        # Filter out rows with missing key data
        q4_filtered_df = q4_sat_df.dropna(subset=[
            "launch_decade", "size_category", country_col, name_col_sat
        ])
        
        if not q4_filtered_df.empty:
            try:
                # Create multi-level pivot table (counts)
                satellite_multilevel_pivot_counts = pd.pivot_table(
                    q4_filtered_df,
                    index=[country_col, "size_category"],  # Multi-level index
                    columns="launch_decade",               # Decades as columns
                    values=name_col_sat,                   # Count satellites
                    aggfunc="count",
                    fill_value=0
                )
                
                print("\nMulti-level Pivot Table (Satellite Counts by Country, Size, Decade):")
                print(satellite_multilevel_pivot_counts.head(15))
                
                satellite_multilevel_pivot_counts.to_csv(
                    os.path.join(OUTPUT_PATH, "satellite_multilevel_pivot_counts.csv")
                )
                logging.info(
                    "Satellite multi-level pivot (counts) saved to 'satellite_multilevel_pivot_counts.csv'"
                )

                # Calculate percentage growth between decades
                decades_present = [col for col in ["1990s", "2000s", "2010s", "2020s"] 
                                 if col in satellite_multilevel_pivot_counts.columns]
                
                if len(decades_present) > 1:
                    satellite_growth_pivot = satellite_multilevel_pivot_counts[decades_present].copy()
                    
                    # Calculate growth between consecutive decades
                    for i in range(1, len(decades_present)):
                        current_decade = decades_present[i]
                        previous_decade = decades_present[i-1]
                        growth_col_name = f"Growth_{previous_decade}_to_{current_decade}_pct"
                        
                        # Calculate percentage growth: ((Current - Previous) / Previous) * 100
                        # Handle division by zero
                        with np.errstate(divide='ignore', invalid='ignore'):
                            growth = ((satellite_growth_pivot[current_decade] - 
                                     satellite_growth_pivot[previous_decade]) / 
                                    satellite_growth_pivot[previous_decade].replace(0, np.nan)) * 100
                        
                        satellite_growth_pivot[growth_col_name] = growth.fillna(0)
                    
                    # Keep only growth columns (remove original count columns)
                    growth_columns = [col for col in satellite_growth_pivot.columns 
                                    if col.startswith("Growth_")]
                    satellite_growth_pivot = satellite_growth_pivot[growth_columns]
                    
                    # Replace infinite values with 0
                    satellite_growth_pivot = satellite_growth_pivot.replace([np.inf, -np.inf], 0)
                    
                    print("\nMulti-level Pivot Table (Satellite Percentage Growth between Decades):")
                    print(satellite_growth_pivot.head(15))
                    
                    satellite_growth_pivot.to_csv(
                        os.path.join(OUTPUT_PATH, "satellite_percentage_growth_pivot.csv")
                    )
                    logging.info(
                        "Satellite percentage growth pivot saved to 'satellite_percentage_growth_pivot.csv'"
                    )
                else:
                    logging.info("Q4: Not enough decade columns to calculate growth.")
                    
            except Exception as e:
                logging.error(f"Error creating multi-level pivot table for satellites (Q4): {e}")
        else:
            logging.warning("Q4: No valid data remaining after filtering for pivot table.")
else:
    logging.warning(
        "Satellite DataFrame is empty or key columns missing, skipping Question 4."
    )