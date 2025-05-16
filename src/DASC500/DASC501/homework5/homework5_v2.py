# homework_5.py (Updated)
# filename: run_homework_script.py
import os
import sys
import logging
import sqlite3 # For sqlite3.Binary
import pandas as pd

try:
    from DASC500.classes.DatabaseManager import DatabaseManager, FileType
except ImportError:
    # Fallback for local testing if DASC500 structure isn't set up
    print("Attempting local import of DatabaseManager for DASC500.")
    try:
        from DatabaseManager import DatabaseManager, FileType # Assuming it's in the same dir
    except ImportError:
        print("ERROR: DatabaseManager.py not found. Ensure it's in the same directory or accessible in PYTHONPATH.")
        exit(1)


# Assuming get_top_level_module_path might not be available in all environments
# For simplicity, define BASE_DIR relative to this script if that function fails
try:
    from DASC500.utilities.get_top_level_module import get_top_level_module_path
    BASE_DIR_PROJECT_ROOT = os.path.join(get_top_level_module_path(), '../..')
except ImportError:
    print("get_top_level_module_path not found. Using script's directory for paths.")
    BASE_DIR_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) # Adjust if needed


# --- 1. Configuration & Setup ---
SCRIPT_TIMESTAMP = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')

BASE_DIR = os.path.join(get_top_level_module_path(), '../..')
INPUT_DIR = os.path.join(BASE_DIR, 'data/DASC501/homework5')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs/DASC501/homework5')

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(INPUT_DIR, exist_ok=True) # Also ensure input dir exists, though files should be there

# Database configuration
DB_FILENAME = f"airforce_maintenance_homework_{SCRIPT_TIMESTAMP}.db"
DB_PATH = os.path.join(INPUT_DIR, DB_FILENAME) # DB will be in 'data' for this setup

# Logging configuration
LOG_FILENAME = f"homework_5_{SCRIPT_TIMESTAMP}.log"
LOG_PATH = os.path.join(OUTPUT_DIR, LOG_FILENAME)

# Data and Image paths (assuming subdirectories within INPUT_DIR)
DATA_DIR_INPUT = INPUT_DIR # Redundant, but for clarity
IMAGE_DIR_INPUT = os.path.join(INPUT_DIR, "images") # Expect 'data/images'
os.makedirs(IMAGE_DIR_INPUT, exist_ok=True)


H5_FILE = os.path.join(DATA_DIR_INPUT, "aicraftgroup1.h5")
PKL_FILE = os.path.join(DATA_DIR_INPUT, "aicraftgroup2.pkl")
XML_FILE = os.path.join(DATA_DIR_INPUT, "aicraftgroup3.xml")
JSON_FILE = os.path.join(DATA_DIR_INPUT, "aircraftgroup4.json") # Corrected typo from aicraft to aircraft

# Assumed image files for specific aircraft
IMAGE_F16 = os.path.join(IMAGE_DIR_INPUT, "F-16.jpg")
IMAGE_B2 = os.path.join(IMAGE_DIR_INPUT, "B-2.jpg") # Changed from C130 to B2 to match aircraft name
AIRCRAFT1_NAME = "F-16 Fighting Falcon"
AIRCRAFT2_NAME = "B-2 Spirit"

# Target table name in the database
TABLE_NAME = "maintenance_log"

# Standardized column name for aircraft type (used in image update)
# This should be the name AFTER DatabaseManager standardizes it (space to underscore)
AIRCRAFT_TYPE_COLUMN_STD = "Aircraft_Type"


# --- Detailed Logger Setup ---
# Configure the root logger
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s',
                    filename=LOG_PATH,
                    filemode='w')

script_logger = logging.getLogger("HomeworkScript")
# Script logger doesn't need separate file handler if basicConfig is set for root.
# It will inherit level from root or can set its own.
# For console output specifically from this script:
console_handler_script = logging.StreamHandler(sys.stdout)
console_handler_script.setLevel(logging.INFO)
console_formatter_script = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - SCRIPT: %(message)s')
console_handler_script.setFormatter(console_formatter_script)
script_logger.addHandler(console_handler_script)
script_logger.propagate = False # Prevent script_logger messages from going to root's console handler if any default is set up by basicConfig that also prints to console.

script_logger.info(f"Logging initialized. Detailed logs will be written to: {LOG_PATH}")
script_logger.info(f"Database for this run: {DB_PATH}")


# --- Main Script Logic ---
def main_script_flow():
    script_logger.info("===================================================================")
    script_logger.info("Starting Air Force Maintenance Data Processing Script")
    script_logger.info(f"Using Database: {DB_PATH}")
    script_logger.info("===================================================================")

    # Instantiate DatabaseManager
    # log_level for DatabaseManager can be set independently
    db_manager = DatabaseManager(db_path=DB_PATH, log_level=logging.DEBUG)

    try:
        with db_manager: # Handles connect, close, commit/rollback
            script_logger.info("DatabaseManager context entered. Connection should be established.")

            script_logger.info("--- TASK 1 & 2: Load data from H5, PKL, XML, JSON files ---")
            files_to_load_config = [
                {
                    "path": H5_FILE, "type": FileType.HDF5,
                    "options": {} # IMPORTANT: Set your HDF5 key
                },
                {
                    "path": PKL_FILE, "type": FileType.PICKLE,
                    "options": {}
                },
                {
                    "path": XML_FILE, "type": FileType.XML,
                    "options": {} # Good practice for your XML structure
                },
                {
                    "path": JSON_FILE, "type": FileType.JSON,
                    "options": {"json_read_options": {"orient": "records"}}
                }
            ]

            table_cleared_once = False
            for file_info in files_to_load_config:
                file_path = file_info["path"]
                file_type_enum = file_info["type"]
                load_opts = file_info["options"]

                script_logger.info(f"Processing file: {file_path} (Type: {file_type_enum.name})")

                if not os.path.exists(file_path):
                    script_logger.error(f"Data file not found: {file_path}. Skipping this file.")
                    # Create dummy file for testing if it doesn't exist
                    # For actual run, this should be an error or handled based on requirements
                    # Example: if file_type_enum == FileType.HDF5: create_dummy_h5(file_path)
                    continue

                try:
                    should_clear = not table_cleared_once
                    script_logger.debug(f"Calling load_data for {file_path}. clear_table_first={should_clear}")
                    
                    success = db_manager.load_data(
                        file_path=file_path,
                        table_name=TABLE_NAME,
                        file_type=file_type_enum,
                        clear_table_first=should_clear,
                        insert_strategy="INSERT OR IGNORE",
                        validation_rules=None,
                        load_options=load_opts
                    )
                    if success:
                        script_logger.info(f"Successfully initiated loading from {file_path} into table '{TABLE_NAME}'.")
                        table_cleared_once = True
                    else:
                        script_logger.warning(f"Loading from {file_path} reported as unsuccessful by DatabaseManager.")
                except Exception as e:
                    script_logger.error(f"An unexpected error occurred while loading {file_path}: {e}", exc_info=True)
            
            script_logger.info("--- TASK 3: Create a new column 'aircraft_image' ---")
            # Standardized column name is already underscore friendly
            image_column_name_std = "aircraft_image"
            if not db_manager.table_exists(TABLE_NAME):
                script_logger.error(f"Table '{TABLE_NAME}' does not exist after load attempts. Cannot add '{image_column_name_std}' column.")
            else:
                try:
                    cols_info = db_manager.get_table_columns(TABLE_NAME)
                    # Column names from DB should also be standardized if created by this manager
                    if cols_info is not None and not any(col[0].lower() == image_column_name_std.lower() for col in cols_info):
                        script_logger.info(f"Column '{image_column_name_std}' not found in '{TABLE_NAME}'. Adding it now.")
                        # Use quotes for safety, though not strictly needed for underscore names
                        alter_sql = f'ALTER TABLE "{TABLE_NAME}" ADD COLUMN "{image_column_name_std}" BLOB;'
                        db_manager.execute_sql(alter_sql)
                        script_logger.info(f"Successfully added '{image_column_name_std}' BLOB column to table '{TABLE_NAME}'.")
                    elif cols_info is not None:
                         script_logger.info(f"Column '{image_column_name_std}' already exists in table '{TABLE_NAME}'. No action taken.")
                    else:
                        script_logger.error(f"Could not get columns for existing table '{TABLE_NAME}'.")
                except Exception as e:
                    script_logger.error(f"Error during '{image_column_name_std}' column addition to '{TABLE_NAME}': {e}", exc_info=True)

            script_logger.info("--- TASK 4: Load aircraft images ---")
            images_to_process = [
                {AIRCRAFT_TYPE_COLUMN_STD: AIRCRAFT1_NAME, "image_path": IMAGE_F16},
                {AIRCRAFT_TYPE_COLUMN_STD: AIRCRAFT2_NAME, "image_path": IMAGE_B2} # Make sure IMAGE_B2 is defined
            ]

            can_load_images = False
            if db_manager.table_exists(TABLE_NAME):
                cols_info_img_check = db_manager.get_table_columns(TABLE_NAME)
                if cols_info_img_check and any(col[0].lower() == image_column_name_std.lower() for col in cols_info_img_check):
                    can_load_images = True
            
            if not can_load_images:
                 script_logger.error(f"Table '{TABLE_NAME}' or column '{image_column_name_std}' does not exist. Skipping image loading.")
            else:
                for img_details in images_to_process:
                    # Use the standardized column name as the key for aircraft name
                    aircraft_id_name_val = img_details[AIRCRAFT_TYPE_COLUMN_STD]
                    img_path = img_details["image_path"]
                    script_logger.info(f"Processing image for '{aircraft_id_name_val}' from path '{img_path}'.")

                    if not os.path.exists(img_path):
                        script_logger.error(f"Image file not found: {img_path}. Cannot load for '{aircraft_id_name_val}'.")
                        # TODO: Create dummy image file for testing if it doesn't exist
                        # Example: create_dummy_jpg(img_path)
                        continue
                    
                    try:
                        with open(img_path, 'rb') as f_image:
                            binary_image = f_image.read()
                        script_logger.debug(f"Read {len(binary_image)} bytes from image file {img_path}.")

                        # Use the standardized column name in the SQL query.
                        # Since it's standardized (no spaces), quotes might not be strictly needed
                        # but are safer if any non-alphanumeric (besides _) could appear.
                        update_query = f"""
                            UPDATE "{TABLE_NAME}" 
                            SET "{image_column_name_std}" = ? 
                            WHERE rowid = (
                                SELECT rowid FROM "{TABLE_NAME}" 
                                WHERE "{AIRCRAFT_TYPE_COLUMN_STD}" = ? 
                                LIMIT 1
                            );
                        """
                        script_logger.debug(f"Executing SQL to update image for {aircraft_id_name_val}")
                        db_manager.execute_sql(update_query, (sqlite3.Binary(binary_image), aircraft_id_name_val))
                        script_logger.info(f"Successfully initiated image update for aircraft '{aircraft_id_name_val}' in table '{TABLE_NAME}'.")
                    except Exception as e:
                        script_logger.error(f"Error loading image for '{aircraft_id_name_val}' from {img_path}: {e}", exc_info=True)
            
            script_logger.info("--- Verifying loaded data ---")
            if db_manager.table_exists(TABLE_NAME):
                try:
                    script_logger.debug(f"Fetching top records from {TABLE_NAME}")
                    # Fetch all columns to see the structure
                    top_records_df = db_manager.execute_select_query(f'SELECT * FROM "{TABLE_NAME}" LIMIT 10;')
                    if top_records_df is not None and not top_records_df.empty:
                        script_logger.info(f"Top records in '{TABLE_NAME}':\n{top_records_df.to_string()}")
                    else:
                        script_logger.info(f"No records found in '{TABLE_NAME}' or query failed.")

                    script_logger.debug(f"Fetching records with aircraft_image from {TABLE_NAME}")
                    # Use standardized column names in SELECT as well
                    image_check_query = f'SELECT "{AIRCRAFT_TYPE_COLUMN_STD}", LENGTH("{image_column_name_std}") as image_size FROM "{TABLE_NAME}" WHERE "{image_column_name_std}" IS NOT NULL;'
                    image_check_df = db_manager.execute_select_query(image_check_query)
                    if image_check_df is not None and not image_check_df.empty:
                        script_logger.info(f"Aircraft with images loaded in '{TABLE_NAME}':\n{image_check_df.to_string()}")
                    else:
                        script_logger.info(f"No records found with image data in '{TABLE_NAME}' or query failed (this is expected if updates didn't match or images were not found).")
                except Exception as e:
                    script_logger.error(f"Error during data verification: {e}", exc_info=True)

        script_logger.info("DatabaseManager context exited. Changes should be committed or rolled back.")
    except Exception as e:
        script_logger.critical(f"A critical error occurred in the main script flow: {e}", exc_info=True)
    finally:
        script_logger.info("===================================================================")
        script_logger.info("Air Force Maintenance Data Processing Script Finished.")
        script_logger.info(f"Final Database located at: {DB_PATH}")
        script_logger.info(f"Detailed Log located at: {LOG_PATH}")
        script_logger.info("===================================================================")
        logging.shutdown()

if __name__ == "__main__":
    # Before running, ensure dummy files exist if you don't have real ones
    # Example: create_dummy_h5(H5_FILE, "your_h5_dataset_key_here")
    # create_dummy_pkl(PKL_FILE)
    # create_dummy_xml(XML_FILE) # Using the structure you provided earlier
    # create_dummy_json(JSON_FILE)
    # create_dummy_jpg(IMAGE_F16)
    # create_dummy_jpg(IMAGE_B2)
    main_script_flow()