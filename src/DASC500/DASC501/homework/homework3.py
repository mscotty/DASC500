# filename: tutoring_queries.py
# Import necessary libraries
import sqlite3
import pandas as pd
import os
import logging
import re  # For validation regex
from datetime import datetime  # For date validation
import sys  # To potentially modify path

# Assume DatabaseManager.py is in the same directory and is the updated version
# --- IMPORTANT: Update this import path if DatabaseManager is elsewhere ---
try:
    from DASC500.classes.DatabaseManager import DatabaseManager
except ImportError:
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), "..", "classes"))
        from DatabaseManager import DatabaseManager

        logging.info("Loaded DatabaseManager using relative path.")
    except ImportError:
        print(
            "Error: DatabaseManager.py not found. Please ensure it's in the correct path."
        )
        print(f"Current sys.path: {sys.path}")
        exit()

# --- Optional: Get Project Folder Path (adjust if needed) ---
try:
    from DASC500.utilities.get_top_level_module import get_top_level_module_path

    IN_FOLDER = os.path.normpath(
        os.path.join(get_top_level_module_path(), r"..\..\data\DASC501\homework3")
    )
    OUT_FOLDER = os.path.normpath(
        os.path.join(get_top_level_module_path(), r"..\..\outputs\DASC501\homework3")
    )
    os.makedirs(OUT_FOLDER, exist_ok=True)
    logging.info(f"Using IN_FOLDER: {IN_FOLDER}")
    logging.info(f"Using OUT_FOLDER: {OUT_FOLDER}")
except ImportError:
    logging.warning(
        "get_top_level_module_path utility not found. Using script's directory for folders."
    )
    print(
        "Warning: get_top_level_module_path utility not found. Assuming folders relative to script."
    )
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    IN_FOLDER = os.path.join(SCRIPT_DIR, "data")
    OUT_FOLDER = os.path.join(SCRIPT_DIR, "outputs")
    os.makedirs(OUT_FOLDER, exist_ok=True)
    logging.info(f"Defaulting IN_FOLDER: {IN_FOLDER}")
    logging.info(f"Defaulting OUT_FOLDER: {OUT_FOLDER}")


# --- Configuration ---
DB_FILE = os.path.join(OUT_FOLDER, "TutoringBusiness_Actual_v1.db")  # Use V1
STUDENT_EXCEL = os.path.join(IN_FOLDER, "tblStudent.xlsx")
TUTOR_EXCEL = os.path.join(IN_FOLDER, "tblTutor-1.xlsx")
CONTRACT_EXCEL = os.path.join(IN_FOLDER, "tblContract.xlsx")
LOG_FILE = os.path.join(OUT_FOLDER, "tutoring_business_log_actual_v1.log")
BACKUP_FILE = os.path.join(OUT_FOLDER, "TutoringBusiness_Actual_v1_backup.db")

# Configure logging for this script
root_logger = logging.getLogger()
if root_logger.hasHandlers():
    root_logger.handlers.clear()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="w"), logging.StreamHandler()],
)


# --- Validation Functions ---
def is_valid_zipcode(zip_val):
    if pd.isna(zip_val):
        return True
    zip_str = str(zip_val).split(".")[0]
    return len(zip_str) == 5 and zip_str.isdigit()


def is_valid_phone(phone_val):
    if pd.isna(phone_val):
        return True
    phone_str = str(phone_val)
    cleaned_phone = re.sub(r"\D", "", phone_str)
    return len(cleaned_phone) == 10


def is_not_empty(value):
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def is_valid_date(date_val):
    """Checks if value is a valid datetime object (like pandas Timestamp). Allows None/NaN."""
    if pd.isna(date_val):
        return True
    # Check if it's a pandas Timestamp or Python datetime
    is_valid = isinstance(date_val, (datetime, pd.Timestamp))
    if not is_valid:
        logging.debug(
            f"Date validation failed for value '{date_val}' (type: {type(date_val)}). Expected datetime object."
        )
    return is_valid


def is_valid_bool_int(bool_val):
    if pd.isna(bool_val):
        return True
    try:
        return float(bool_val) in [0.0, 1.0]
    except (ValueError, TypeError):
        return False


def is_non_negative(num_val):
    if pd.isna(num_val):
        return True
    try:
        return float(num_val) >= 0
    except (ValueError, TypeError):
        return False


# --- Data Transformation Function ---
def transform_boolean(df, column_name):
    if column_name in df.columns:
        logging.debug(f"Transforming boolean column: {column_name}")
        str_series = df[column_name].astype(str).str.lower()
        df[column_name] = str_series.map({"true": 1, "false": 0})
        logging.debug(
            f"Column {column_name} transformed. NaN count: {df[column_name].isna().sum()}"
        )
    else:
        logging.warning(
            f"Boolean transformation skipped: Column '{column_name}' not found."
        )
    return df


# --- Main Script Logic ---
def main():
    logging.info(f"Initializing Database Manager for database: {DB_FILE}")
    with DatabaseManager(
        db_path=DB_FILE, log_file=LOG_FILE, log_level=logging.INFO
    ) as db_manager:
        try:
            # --- 1. Create Schema ---
            logging.info("Checking/Creating schema...")
            db_manager.execute_sql("PRAGMA foreign_keys = ON;")
            # Schema definitions (assuming these are correct now)
            create_student_table_sql = """
            CREATE TABLE IF NOT EXISTS tblStudent (
                StudentID TEXT PRIMARY KEY NOT NULL, FirstName TEXT NOT NULL, LastName TEXT NOT NULL, Address TEXT,
                City TEXT, State TEXT, Zip TEXT CHECK(LENGTH(Zip) = 5 OR Zip IS NULL),
                HomePhone TEXT CHECK(LENGTH(REPLACE(REPLACE(REPLACE(HomePhone, '-', ''), '(', ''), ')', '')) = 10 OR HomePhone IS NULL),
                CellPhone TEXT CHECK(LENGTH(REPLACE(REPLACE(REPLACE(CellPhone, '-', ''), '(', ''), ')', '')) = 10 OR CellPhone IS NULL),
                BirthDate TEXT, Gender TEXT CHECK(Gender IN ('M', 'F', 'Other', NULL))
            );"""
            create_tutor_table_sql = """
            CREATE TABLE IF NOT EXISTS tblTutor (
                TutorID TEXT PRIMARY KEY NOT NULL, FirstName TEXT NOT NULL, LastName TEXT NOT NULL, Major TEXT,
                YearInSchool TEXT, School TEXT, HireDate TEXT, Groups INTEGER CHECK(Groups IN (0, 1) OR Groups IS NULL)
            );"""
            create_contract_table_sql = """
            CREATE TABLE IF NOT EXISTS tblContract (
                ContractID INTEGER PRIMARY KEY NOT NULL, StudentID TEXT, TutorID TEXT, ContractDate TEXT, SessionType TEXT,
                Length REAL, NumSessions INTEGER, Cost REAL CHECK(Cost >= 0 OR Cost IS NULL),
                Assessment INTEGER CHECK(Assessment IN (0, 1) OR Assessment IS NULL),
                FOREIGN KEY (StudentID) REFERENCES tblStudent (StudentID) ON DELETE SET NULL ON UPDATE CASCADE,
                FOREIGN KEY (TutorID) REFERENCES tblTutor (TutorID) ON DELETE SET NULL ON UPDATE CASCADE
            );"""
            if not db_manager.table_exists("tblStudent"):
                db_manager.execute_sql(create_student_table_sql)
                logging.info("Created tblStudent.")
            if not db_manager.table_exists("tblTutor"):
                db_manager.execute_sql(create_tutor_table_sql)
                logging.info("Created tblTutor.")
            if not db_manager.table_exists("tblContract"):
                db_manager.execute_sql(create_contract_table_sql)
                logging.info("Created tblContract.")
            db_manager.commit()
            logging.info("Schema check/creation complete.")

            # --- 2. Import Data ---
            logging.info("Importing data with validation and transformation...")

            # Define validation rules (should match Excel columns)
            student_validation = {
                "FirstName": is_not_empty,
                "LastName": is_not_empty,
                "Zip": is_valid_zipcode,
                "HomePhone": is_valid_phone,
                "CellPhone": is_valid_phone,
                "BirthDate": is_valid_date,  # Validate the datetime object
            }
            tutor_validation = {
                "FirstName": is_not_empty,
                "LastName": is_not_empty,
                "HireDate": is_valid_date,  # Validate the datetime object
                "Groups": is_valid_bool_int,  # Validate AFTER transformation
            }
            contract_validation = {
                "ContractDate": is_valid_date,  # Validate the datetime object
                "Cost": is_non_negative,
                "Assessment": is_valid_bool_int,  # Validate AFTER transformation
            }

            # -- Define dtypes for reading (excluding dates, let pandas parse) --
            student_dtypes = {
                "StudentID": str,
                "Zip": str,
                "HomePhone": str,
                "CellPhone": str,
            }
            tutor_dtypes = {"TutorID": str, "Groups": str}  # Read bool as str
            contract_dtypes = {
                "ContractID": str,
                "StudentID": str,
                "TutorID": str,
                "Assessment": str,
            }  # Read bool as str

            # Date columns expected
            student_date_cols = ["BirthDate"]
            tutor_date_cols = ["HireDate"]
            contract_date_cols = ["ContractDate"]

            # --- Import Process ---
            TEMP_FOLDER = OUT_FOLDER  # Use OUT_FOLDER for temp files

            # -- Import Students --
            TEMP_STUDENT_XLSX = os.path.join(TEMP_FOLDER, "temp_student_processed.xlsx")
            if os.path.exists(STUDENT_EXCEL):
                logging.info(f"Processing students from {STUDENT_EXCEL}...")
                try:
                    # Read Excel, let pandas parse dates
                    df_student = pd.read_excel(STUDENT_EXCEL, dtype=student_dtypes)
                    logging.info(f"Read {len(df_student)} student rows.")

                    # Apply validation (including date objects)
                    df_validated_student = db_manager._validate_dataframe(
                        df_student, student_validation, "skip"
                    )

                    if not df_validated_student.empty:
                        # Convert valid date columns to 'YYYY-MM-DD' strings AFTER validation
                        for col in student_date_cols:
                            if col in df_validated_student.columns:
                                # Use .dt.strftime, handle NaT errors gracefully
                                df_validated_student[col] = pd.to_datetime(
                                    df_validated_student[col], errors="coerce"
                                ).dt.strftime("%Y-%m-%d")

                        # Save processed data to temp file
                        df_validated_student.to_excel(
                            TEMP_STUDENT_XLSX, index=False, engine="openpyxl"
                        )
                        logging.info(
                            f"Processed student data saved to {TEMP_STUDENT_XLSX}"
                        )

                        # Import from the processed temp file
                        success_student = db_manager.import_data_from_excel(
                            table_name="tblStudent",
                            excel_file_path=TEMP_STUDENT_XLSX,
                            clear_table_first=True,
                            insert_strategy="INSERT OR IGNORE",
                            validation_rules=None,  # Validation already done
                            on_validation_error="skip",  # Not relevant here
                        )
                        log_import_status("Student", success_student, STUDENT_EXCEL)
                    else:
                        logging.warning(
                            "No valid student data after validation. Skipping import."
                        )
                        log_import_status("Student", False, STUDENT_EXCEL)

                except Exception as e:
                    logging.error(
                        f"Failed to process or import {STUDENT_EXCEL}: {e}",
                        exc_info=True,
                    )
                    log_import_status("Student", False, STUDENT_EXCEL)
                finally:
                    if os.path.exists(TEMP_STUDENT_XLSX):
                        os.remove(TEMP_STUDENT_XLSX)
                        logging.info("Removed temp student file.")
            else:
                logging.warning(f"File not found: {STUDENT_EXCEL}. Skipping.")

            # -- Import Tutors --
            TEMP_TUTOR_XLSX = os.path.join(TEMP_FOLDER, "temp_tutor_processed.xlsx")
            if os.path.exists(TUTOR_EXCEL):
                logging.info(f"Processing tutors from {TUTOR_EXCEL}...")
                try:
                    df_tutor = pd.read_excel(TUTOR_EXCEL, dtype=tutor_dtypes)
                    logging.info(f"Read {len(df_tutor)} tutor rows.")
                    df_tutor = transform_boolean(
                        df_tutor, "Groups"
                    )  # Transform boolean FIRST

                    # Apply validation (including date objects and transformed bool)
                    df_validated_tutor = db_manager._validate_dataframe(
                        df_tutor, tutor_validation, "skip"
                    )

                    if not df_validated_tutor.empty:
                        # Convert valid date columns to 'YYYY-MM-DD' strings AFTER validation
                        for col in tutor_date_cols:
                            if col in df_validated_tutor.columns:
                                df_validated_tutor[col] = pd.to_datetime(
                                    df_validated_tutor[col], errors="coerce"
                                ).dt.strftime("%Y-%m-%d")

                        # Save processed data to temp file
                        df_validated_tutor.to_excel(
                            TEMP_TUTOR_XLSX, index=False, engine="openpyxl"
                        )
                        logging.info(f"Processed tutor data saved to {TEMP_TUTOR_XLSX}")

                        success_tutor = db_manager.import_data_from_excel(
                            table_name="tblTutor",
                            excel_file_path=TEMP_TUTOR_XLSX,
                            clear_table_first=True,
                            insert_strategy="INSERT OR IGNORE",
                            validation_rules=None,
                            on_validation_error="skip",
                        )
                        log_import_status("Tutor", success_tutor, TUTOR_EXCEL)
                    else:
                        logging.warning(
                            "No valid tutor data after validation. Skipping import."
                        )
                        log_import_status("Tutor", False, TUTOR_EXCEL)

                except Exception as e:
                    logging.error(
                        f"Failed to process or import {TUTOR_EXCEL}: {e}", exc_info=True
                    )
                    log_import_status("Tutor", False, TUTOR_EXCEL)
                finally:
                    if os.path.exists(TEMP_TUTOR_XLSX):
                        os.remove(TEMP_TUTOR_XLSX)
                        logging.info("Removed temp tutor file.")
            else:
                logging.warning(f"File not found: {TUTOR_EXCEL}. Skipping.")

            # -- Import Contracts --
            TEMP_CONTRACT_XLSX = os.path.join(
                TEMP_FOLDER, "temp_contract_processed.xlsx"
            )
            if os.path.exists(CONTRACT_EXCEL):
                logging.info(f"Processing contracts from {CONTRACT_EXCEL}...")
                try:
                    df_contract = pd.read_excel(CONTRACT_EXCEL, dtype=contract_dtypes)
                    logging.info(f"Read {len(df_contract)} contract rows.")
                    df_contract = transform_boolean(
                        df_contract, "Assessment"
                    )  # Transform boolean FIRST

                    # Apply validation (including date objects and transformed bool)
                    df_validated_contract = db_manager._validate_dataframe(
                        df_contract, contract_validation, "skip"
                    )

                    if not df_validated_contract.empty:
                        # Convert valid date columns to 'YYYY-MM-DD' strings AFTER validation
                        for col in contract_date_cols:
                            if col in df_validated_contract.columns:
                                df_validated_contract[col] = pd.to_datetime(
                                    df_validated_contract[col], errors="coerce"
                                ).dt.strftime("%Y-%m-%d")

                        # Save processed data to temp file
                        df_validated_contract.to_excel(
                            TEMP_CONTRACT_XLSX, index=False, engine="openpyxl"
                        )
                        logging.info(
                            f"Processed contract data saved to {TEMP_CONTRACT_XLSX}"
                        )

                        # Check parent tables non-empty BEFORE importing
                        student_count_res = db_manager.execute_select_query(
                            "SELECT COUNT(*) FROM tblStudent", return_dataframe=False
                        )
                        tutor_count_res = db_manager.execute_select_query(
                            "SELECT COUNT(*) FROM tblTutor", return_dataframe=False
                        )
                        s_count = (
                            student_count_res[0][0]
                            if student_count_res and isinstance(student_count_res, list)
                            else 0
                        )
                        t_count = (
                            tutor_count_res[0][0]
                            if tutor_count_res and isinstance(tutor_count_res, list)
                            else 0
                        )

                        if s_count > 0 and t_count > 0:
                            success_contract = db_manager.import_data_from_excel(
                                table_name="tblContract",
                                excel_file_path=TEMP_CONTRACT_XLSX,
                                clear_table_first=True,
                                insert_strategy="INSERT OR IGNORE",
                                validation_rules=None,
                                on_validation_error="skip",
                            )
                            log_import_status(
                                "Contract", success_contract, CONTRACT_EXCEL
                            )
                        else:
                            logging.warning(
                                f"Skipping contract import as Student ({s_count}) or Tutor ({t_count}) tables appear empty."
                            )
                            log_import_status(
                                "Contract", False, CONTRACT_EXCEL
                            )  # Log as failed if skipped
                    else:
                        logging.warning(
                            "No valid contract data after validation. Skipping import."
                        )
                        log_import_status("Contract", False, CONTRACT_EXCEL)

                except Exception as e:
                    logging.error(
                        f"Failed to process or import {CONTRACT_EXCEL}: {e}",
                        exc_info=True,
                    )
                    log_import_status("Contract", False, CONTRACT_EXCEL)
                finally:
                    if os.path.exists(TEMP_CONTRACT_XLSX):
                        os.remove(TEMP_CONTRACT_XLSX)
                        logging.info("Removed temp contract file.")
            else:
                logging.warning(f"File not found: {CONTRACT_EXCEL}. Skipping.")

            # --- 3. Query 1: Semi-private & Assessment ---
            # (Queries remain the same as previous version)
            logging.info(
                "Querying: Students with semi-private lessons who took assessment..."
            )
            query1_sql = """
            SELECT s.FirstName, s.LastName, s.City, s.HomePhone, s.CellPhone
            FROM tblStudent s JOIN tblContract c ON s.StudentID = c.StudentID
            WHERE c.SessionType = 'Semi-private' AND c.Assessment = 1;
            """
            run_and_print_query(
                db_manager,
                "Query 1 Results (Semi-Private Lessons, Assessment Taken)",
                query1_sql,
            )

            # --- 4. Query 2: Students with no contract ---
            logging.info("Querying: Students with no associated contract data...")
            query2_sql = """
            SELECT s.FirstName, s.LastName,
                   s.Address || ', ' || s.City || ', ' || s.State || ' ' || s.Zip AS Address,
                   'Home: ' || COALESCE(s.HomePhone, 'N/A') || ' | Cell: ' || COALESCE(s.CellPhone, 'N/A') AS Phone
            FROM tblStudent s LEFT JOIN tblContract c ON s.StudentID = c.StudentID
            WHERE c.ContractID IS NULL;"""
            run_and_print_query(
                db_manager, "Query 2 Results (Students with No Contracts)", query2_sql
            )

            # --- 5. Query 3: Cost by city ---
            logging.info("Querying: Cost statistics by student city...")
            query3_sql = """
            SELECT s.City, MIN(c.Cost) AS MinCost, AVG(c.Cost) AS MeanCost, MAX(c.Cost) AS MaxCost
            FROM tblStudent s JOIN tblContract c ON s.StudentID = c.StudentID
            WHERE s.City IS NOT NULL AND c.Cost IS NOT NULL
            GROUP BY s.City ORDER BY MeanCost ASC;"""
            run_and_print_query(
                db_manager,
                "Query 3 Results (Cost Statistics by City)",
                query3_sql,
                format_currency=True,
            )

            # --- 6. Backup Database ---
            logging.info("Attempting to backup the database...")
            backup_success = db_manager.backup_database(BACKUP_FILE)
            if backup_success:
                print(f"\nDatabase successfully backed up to {BACKUP_FILE}")
            else:
                print("\nDATABASE BACKUP FAILED.")

        except sqlite3.Error as db_err:
            logging.error(f"A database error occurred: {db_err}", exc_info=True)
            print(f"\nDATABASE ERROR: {db_err}")
        except FileNotFoundError as fnf_err:
            logging.error(f"A required file was not found: {fnf_err}", exc_info=False)
            print(f"\nFILE NOT FOUND ERROR: {fnf_err}")
        except ImportError as imp_err:
            logging.critical(f"Missing required library: {imp_err}", exc_info=False)
            print(f"\nIMPORT ERROR: {imp_err}.")
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}", exc_info=True)
            print(f"\nUNEXPECTED ERROR: {e}")

    logging.info("Script finished.")
    print(f"\nProcessing complete. Check console output and log file '{LOG_FILE}'.")
    print(f"Database file is '{DB_FILE}'.")


# --- Helper Functions (Keep as before) ---
def log_import_status(data_type, success, filename):
    if success:
        logging.info(f"{data_type} data import task completed for {filename}.")
    else:
        logging.warning(
            f"{data_type} data import task failed or encountered errors for {filename}."
        )


def run_and_print_query(db_manager, title, sql, params=None, format_currency=False):
    # Added check for db_manager existence
    if not db_manager:
        logging.error(f"DatabaseManager not initialized for query '{title}'.")
        print(f"\n--- {title} ---")
        print("Database connection not available.")
        return

    results_df = db_manager.execute_select_query(
        sql, params=params, return_dataframe=True
    )
    print(f"\n--- {title} ---")
    if results_df is not None and not results_df.empty:
        if format_currency:
            currency_cols = [col for col in results_df.columns if "cost" in col.lower()]
            format_dict = {col: "${:,.2f}".format for col in currency_cols}
            valid_format_dict = {
                k: v for k, v in format_dict.items() if k in results_df.columns
            }
            print(
                results_df.to_string(
                    index=False,
                    formatters=valid_format_dict if valid_format_dict else None,
                )
            )
        else:
            print(results_df.to_string(index=False))
        logging.info(f"Query '{title}' finished. Found {len(results_df)} records.")
    elif results_df is not None and results_df.empty:
        print("No records found matching the criteria.")
        logging.info(f"Query '{title}' finished. Found 0 records.")
    else:  # results_df is None
        print("Query execution failed. Check logs.")
        logging.error(f"Query execution failed for '{title}'.")


# --- Run the main function ---
if __name__ == "__main__":
    files_exist = True
    for f in [STUDENT_EXCEL, TUTOR_EXCEL, CONTRACT_EXCEL]:
        if not os.path.exists(f):
            logging.error(f"Required input file not found: {f}")
            print(f"ERROR: Required input file not found: {f}")
            files_exist = False

    if files_exist:
        main()
    else:
        print("\nAborting script because required input files are missing.")
        logging.error("Aborting script because required input files are missing.")
