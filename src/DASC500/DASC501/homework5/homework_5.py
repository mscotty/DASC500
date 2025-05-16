# filename: run_homework_script.py
import os
import logging
import sqlite3 # For sqlite3.Binary
import pandas as pd # For dummy data creation and pd.Timestamp

# Assuming DatabaseManager.py is in the same directory or PYTHONPATH
try:
    from DASC500.classes.DatabaseManager import DatabaseManager, FileType
except ImportError:
    print("ERROR: DatabaseManager.py not found. Make sure it's in the same directory or accessible in PYTHONPATH.")
    exit(1)

from DASC500.utilities.get_top_level_module import get_top_level_module_path

# --- 1. Configuration & Setup ---
SCRIPT_TIMESTAMP = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')

# Base directory of the script
BASE_DIR = os.path.join(get_top_level_module_path(), '../..')
INPUT_DIR = os.path.join(BASE_DIR, 'data/DASC501/homework5')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs/DASC501/homework5')

# Database configuration
DB_FILENAME = f"airforce_maintenance_homework_{SCRIPT_TIMESTAMP}.db" # Unique DB per run for cleaner testing
DB_PATH = os.path.join(INPUT_DIR, DB_FILENAME)

# Logging configuration
LOG_FILENAME = f"homework_5_{SCRIPT_TIMESTAMP}.log"
LOG_PATH = os.path.join(OUTPUT_DIR, LOG_FILENAME)

# Data and Image paths (assuming subdirectories)
DATA_DIR = INPUT_DIR
IMAGE_DIR = os.path.join(INPUT_DIR, "images")

H5_FILE = os.path.join(DATA_DIR, "aicraftgroup1.h5")
PKL_FILE = os.path.join(DATA_DIR, "aicraftgroup2.pkl")
XML_FILE = os.path.join(DATA_DIR, "aicraftgroup3.xml")
JSON_FILE = os.path.join(DATA_DIR, "aircraftgroup4.json")

# Assumed image files for specific aircraft
IMAGE_F16 = os.path.join(IMAGE_DIR, "F-16.jpg")
IMAGE_C130 = os.path.join(IMAGE_DIR, "B-2.jpg")
AIRCRAFT1_NAME = "F-16 Fighting Falcon"       # Must match a name in your data
AIRCRAFT2_NAME = "B-2 Spirit"    # Must match a name in your data

# Target table name in the database
TABLE_NAME = "maintenance_log"

# --- Detailed Logger Setup ---
# Configure the root logger, which DatabaseManager might also use if not configured separately
logging.basicConfig(level=logging.DEBUG, # Capture all levels to the file
                    format='%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s',
                    filename=LOG_PATH,
                    filemode='w') # 'w' for overwrite, 'a' for append

# Script-specific logger
script_logger = logging.getLogger("HomeworkScript")
script_logger.setLevel(logging.DEBUG) # Ensure script logger also captures debug

# Console Handler for script_logger (for higher-level script progress)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO) # Show INFO and above on console
console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - SCRIPT: %(message)s')
console_handler.setFormatter(console_formatter)
script_logger.addHandler(console_handler)

script_logger.info(f"Logging initialized. Detailed logs will be written to: {LOG_PATH}")
script_logger.info(f"Database for this run: {DB_PATH}")


# --- Main Script Logic ---
def main_script_flow():
    script_logger.info("===================================================================")
    script_logger.info("Starting Air Force Maintenance Data Processing Script")
    script_logger.info(f"Using Database: {DB_PATH}")
    script_logger.info("===================================================================")

    # Create dummy files if you don't have actual data/image files in specified paths.
    # Comment out if you have your own files.
    #if not create_dummy_data_files():
    #    script_logger.critical("Failed to create dummy files. If you don't have your own files, the script might not run correctly.")
        # return # Or proceed, and it will likely fail at file loading

    # Instantiate DatabaseManager
    # The DatabaseManager logs to the root logger, which we've configured to go to our file.
    db_manager = DatabaseManager(db_path=DB_PATH, log_level=logging.DEBUG) # Manager logs at DEBUG

    try:
        with db_manager: # Ensures connection is opened, and closed (with commit/rollback)
            script_logger.info("DatabaseManager context entered. Connection should be established.")

            # --- Task 1 & 2: Load data from various file formats ---
            # {"path": H5_FILE, "type": FileType.HDF5, "options": {"hdf_read_options": {"key": "maintenance_records"}}},
            # {"path": XML_FILE, "type": FileType.XML, "options": {"xml_read_options": {"xpath": ".//visit"}}},
            # {"path": JSON_FILE, "type": FileType.JSON, "options": {"json_read_options": {"orient": "records", "dtype": {'duration_hours': float}}}} # ensure float
            script_logger.info("--- TASK 1 & 2: Load data from H5, PKL, XML, JSON files ---")
            files_to_load_config = [
                {"path": H5_FILE, "type": FileType.HDF5, "options": {}},
                {"path": PKL_FILE, "type": FileType.PICKLE, "options": {}}, # No special options for basic pickle
                {"path": XML_FILE, "type": FileType.XML, "options": {}},
                {"path": JSON_FILE, "type": FileType.JSON, "options": {}} # ensure float
            ]

            # For this homework, we load all into the same table.
            # The `clear_table_first` will be True only for the first *attempted* load to this table.
            # `insert_strategy` helps if different files have overlapping `visit_id`s.
            table_cleared_once = False
            for file_info in files_to_load_config:
                file_path = file_info["path"]
                file_type_enum = file_info["type"]
                load_opts = file_info["options"]

                script_logger.info(f"Processing file: {file_path} (Type: {file_type_enum.name})")

                if not os.path.exists(file_path):
                    script_logger.error(f"Data file not found: {file_path}. Skipping this file.")
                    continue

                try:
                    should_clear = not table_cleared_once
                    script_logger.debug(f"Calling load_data for {file_path}. clear_table_first={should_clear}")
                    
                    success = db_manager.load_data(
                        file_path=file_path,
                        table_name=TABLE_NAME,
                        file_type=file_type_enum,
                        clear_table_first=should_clear,
                        insert_strategy="INSERT OR IGNORE", # Or REPLACE, depending on desired outcome for duplicates
                        validation_rules=None, # Add validation rules if needed
                        load_options=load_opts
                    )
                    if success:
                        script_logger.info(f"Successfully initiated loading from {file_path} into table '{TABLE_NAME}'.")
                        table_cleared_once = True # From now on, append or use insert strategy
                    else:
                        script_logger.warning(f"Loading from {file_path} reported as unsuccessful by DatabaseManager.")
                except FileNotFoundError: # Should be caught by os.path.exists
                     script_logger.error(f"FileNotFoundError for {file_path} despite pre-check. Critical issue.")
                except Exception as e:
                    script_logger.error(f"An unexpected error occurred while loading {file_path}: {e}", exc_info=True)
            
            # --- Task 3: Create a new column "aircraft_image" ---
            script_logger.info("--- TASK 3: Add 'aircraft_image' column to the table ---")
            if not db_manager.table_exists(TABLE_NAME):
                script_logger.error(f"Table '{TABLE_NAME}' does not exist after load attempts. Cannot add 'aircraft_image' column. Please check data loading steps.")
            else:
                try:
                    cols_info = db_manager.get_table_columns(TABLE_NAME)
                    if cols_info and not any(col[0].lower() == 'aircraft_image' for col in cols_info):
                        script_logger.info(f"Column 'aircraft_image' not found in '{TABLE_NAME}'. Adding it now.")
                        alter_sql = f"ALTER TABLE {TABLE_NAME} ADD COLUMN aircraft_image BLOB;"
                        db_manager.execute_sql(alter_sql)
                        # No explicit commit here; context manager handles it on successful exit.
                        script_logger.info(f"Successfully added 'aircraft_image' BLOB column to table '{TABLE_NAME}'.")
                    elif cols_info:
                         script_logger.info(f"Column 'aircraft_image' already exists in table '{TABLE_NAME}'. No action taken.")
                    else: # Should not happen if table_exists is true
                        script_logger.error(f"Could not get columns for existing table '{TABLE_NAME}'.")
                except Exception as e:
                    script_logger.error(f"Error during 'aircraft_image' column addition to '{TABLE_NAME}': {e}", exc_info=True)

            # --- Task 4: "Download" (use local files) and load two images ---
            script_logger.info("--- TASK 4: Load aircraft images ---")
            images_to_process = [
                {'"Aircraft Type"': AIRCRAFT1_NAME, "image_path": IMAGE_F16},
                {'"Aircraft Type"': AIRCRAFT2_NAME, "image_path": IMAGE_C130}
            ]

            if not db_manager.table_exists(TABLE_NAME) or not any(col[0].lower() == 'aircraft_image' for col in (db_manager.get_table_columns(TABLE_NAME) or [])):
                 script_logger.error(f"Table '{TABLE_NAME}' or column 'aircraft_image' does not exist. Skipping image loading.")
            else:
                for img_details in images_to_process:
                    aircraft_id_name = img_details['"Aircraft Type"']
                    img_path = img_details["image_path"]
                    script_logger.info(f"Processing image for '{aircraft_id_name}' from path '{img_path}'.")

                    if not os.path.exists(img_path):
                        script_logger.error(f"Image file not found: {img_path}. Cannot load for '{aircraft_id_name}'.")
                        continue
                    
                    try:
                        with open(img_path, 'rb') as f_image:
                            binary_image = f_image.read()
                        script_logger.debug(f"Read {len(binary_image)} bytes from image file {img_path}.")

                        # Update a record. This assumes ''Aircraft Type'' can identify the record.
                        # If ''Aircraft Type'' is not unique, this might update multiple rows or the first one SQLite finds.
                        # For homework, usually updating one matching record is fine.
                        # A more robust way would be to use a unique ID if available.
                        # Example: Update first record found matching the 'Aircraft Type'
                        update_query = f"""
                            UPDATE {TABLE_NAME} 
                            SET aircraft_image = ? 
                            WHERE rowid = (
                                SELECT rowid FROM {TABLE_NAME} 
                                WHERE 'Aircraft Type' = ? 
                                LIMIT 1
                            );
                        """
                        # To update ALL records with that name:
                        # update_query = f"UPDATE {TABLE_NAME} SET aircraft_image = ? WHERE 'Aircraft Type' = ?;"
                        
                        script_logger.debug(f"Executing SQL: {update_query.strip()} with params (BINARY_DATA, {aircraft_id_name})")
                        db_manager.execute_sql(update_query, (sqlite3.Binary(binary_image), aircraft_id_name))
                        script_logger.info(f"Successfully updated image for aircraft '{aircraft_id_name}' in table '{TABLE_NAME}'.")
                    except Exception as e:
                        script_logger.error(f"Error loading image for '{aircraft_id_name}' from {img_path}: {e}", exc_info=True)
            
            script_logger.info("--- Verifying loaded data (first 5 records and image presence) ---")
            if db_manager.table_exists(TABLE_NAME):
                try:
                    script_logger.debug(f"Fetching top 5 records from {TABLE_NAME}")
                    top_5_records = db_manager.execute_select_query(f"SELECT * FROM {TABLE_NAME} LIMIT 100;")
                    if top_5_records is not None and not top_5_records.empty:
                        script_logger.info(f"Top 5 records in '{TABLE_NAME}':\n{top_5_records.to_string()}")
                    else:
                        script_logger.info(f"No records found in '{TABLE_NAME}' or query failed.")

                    script_logger.debug(f"Fetching records with aircraft_image from {TABLE_NAME}")
                    image_check_records = db_manager.execute_select_query(
                        f"SELECT 'Aircraft Type', LENGTH(aircraft_image) as image_size FROM {TABLE_NAME} WHERE aircraft_image IS NOT NULL;"
                    )
                    if image_check_records is not None and not image_check_records.empty:
                        script_logger.info(f"Aircraft with images loaded in '{TABLE_NAME}':\n{image_check_records.to_string()}")
                    else:
                        script_logger.info(f"No records found with image data in '{TABLE_NAME}' or query failed.")
                except Exception as e:
                    script_logger.error(f"Error during data verification: {e}", exc_info=True)

        script_logger.info("DatabaseManager context is about to exit. Changes will be committed or rolled back.")
    except Exception as e:
        script_logger.critical(f"A critical error occurred in the main script flow: {e}", exc_info=True)
    finally:
        script_logger.info("===================================================================")
        script_logger.info("Air Force Maintenance Data Processing Script Finished.")
        script_logger.info(f"Final Database located at: {DB_PATH}")
        script_logger.info(f"Detailed Log located at: {LOG_PATH}")
        script_logger.info("===================================================================")
        logging.shutdown() # Cleanly close logging handlers

if __name__ == "__main__":
    main_script_flow()