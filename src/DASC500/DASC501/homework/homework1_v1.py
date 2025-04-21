import sqlite3
import pandas as pd
import os
import sys

from DASC500.utilities.get_top_level_module import get_top_level_module_path
from DASC500.classes.DatabaseManager import DatabaseManager

# --- Main Execution Block ---
if __name__ == "__main__":
    # Configuration for Jane's Tutoring Business
    DATABASE_FILE = 'TutoringBusiness_GenericOOP.db'
    EXCEL_DATA_FILE = get_top_level_module_path() + '/../../data/DASC501/tblTutor.xlsx' # Assumed to be in the same directory
    TABLE_NAME = 'tblTutor'
    TABLE_COLUMNS = ['TutorID', 'FirstName', 'LastName', 'Major', 'YearInSchool', 'School', 'HireDate']
    OUTPUT_LOG_FILE = get_top_level_module_path() + '/../../outputs/DASC501/homework1/tutoring_business_output.txt' # Define the output filename

    # --- Setup Output Redirection ---
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    print(f"--- Script starting. Output will be redirected to '{OUTPUT_LOG_FILE}' ---") # Message to console

    try:
        # Open the output file in write mode, redirect stdout and stderr
        with open(OUTPUT_LOG_FILE, 'w', encoding='utf-8') as outfile:
            sys.stdout = outfile
            sys.stderr = outfile

            # --- Start of Database Operations ---
            print("="*60)
            print(" Tutoring Business DB Management - Generic Methods Execution ")
            print(f" Database file: {DATABASE_FILE}")
            print(f" Excel source:  {EXCEL_DATA_FILE}")
            # Check if Excel file exists before proceeding (optional but good practice)
            if not os.path.exists(EXCEL_DATA_FILE):
                 print(f"\nWARNING: Excel data file not found at specified path: {EXCEL_DATA_FILE}")
                 print("         Import step will likely report a warning or fail.")
            print(f" Output Log:    {OUTPUT_LOG_FILE}")
            print(f" Execution Time: {pd.Timestamp.now()}")
            print("="*60)

            query_results = {} # To store results if needed later

            # Nested try/except for the database operations themselves
            try:
                # Use context manager for automatic connection/disconnection
                with DatabaseManager(db_path=DATABASE_FILE, excel_path=EXCEL_DATA_FILE) as db_manager:

                    # Task 1: Create database file (Handled implicitly by DatabaseManager connection)
                    print("\n--- Task 1: Create Database File (Implicit via connect) ---")
                    # Connection established by `with` statement entering context.
                    # Confirmation should be printed by the connect() method.

                    # Task 2: Create table tblTutor
                    print("\n--- Task 2: Create Table ---")
                    create_table_sql = f"""
                    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                        TutorID      INTEGER PRIMARY KEY,
                        FirstName    TEXT NOT NULL,
                        LastName     TEXT NOT NULL,
                        Major        TEXT,
                        YearInSchool TEXT,
                        School       TEXT,
                        HireDate     TEXT
                    );
                    """
                    db_manager.execute_sql(create_table_sql)
                    db_manager.commit() # Commit DDL changes

                    # Task 3: Import data from tblTutor excel file
                    print("\n--- Task 3: Import Data from Excel ---")
                    import_status = db_manager.import_data_from_excel(
                        table_name=TABLE_NAME,
                        expected_columns=TABLE_COLUMNS
                    )
                    # Import method handles its own commit/rollback

                    # Proceed with queries only if import didn't critically fail
                    if not import_status and os.path.exists(EXCEL_DATA_FILE):
                         print("\nCRITICAL: Halting query execution due to data import errors.")
                    else:
                        print("\n--- Proceeding with Data Queries ---")

                        # Task 4: Query tutors hired after April 2017
                        print("\n--- Task 4: Query Tutors Hired After 2017-04-30 ---")
                        sql_hired_after = f"SELECT FirstName, LastName, HireDate FROM {TABLE_NAME} WHERE HireDate > ?"
                        hired_after_threshold = '2017-04-30'
                        results4 = db_manager.execute_select_query(sql_hired_after, (hired_after_threshold,))
                        query_results['hired_after_2017_04'] = results4
                        print("Results:") # Print header even if no results
                        if results4: # Check if list is not None and not empty
                            print(f"{'FirstName':<15} | {'LastName':<15} | {'HireDate'}")
                            print("-" * 47)
                            for row in results4: print(f"{row[0]:<15} | {row[1]:<15} | {row[2]}")
                        elif results4 is None: # Check if query failed
                            print("Query execution failed.")
                        else: # Query succeeded but returned no rows
                            print("No matching records found.")


                        # Task 5: Query distinct types of Major
                        print("\n--- Task 5: Query Distinct Tutor Majors ---")
                        sql_distinct_majors = f"SELECT DISTINCT Major FROM {TABLE_NAME};"
                        results5 = db_manager.execute_select_query(sql_distinct_majors)
                        query_results['distinct_majors'] = results5
                        print("Results (Distinct Majors):") # Print header
                        print("-------------------------")
                        if results5:
                            distinct_majors_list = sorted([str(row[0]) if row[0] is not None else "N/A" for row in results5])
                            for major in distinct_majors_list: print(major)
                        elif results5 is None:
                             print("Query execution failed.")
                        else:
                            print("No distinct majors found.")

                        # Task 6: Query tutors whose YearInSchool is Graduate
                        print("\n--- Task 6: Query Graduate Tutors ---")
                        sql_graduate_tutors = f"SELECT FirstName, LastName FROM {TABLE_NAME} WHERE YearInSchool = ?"
                        graduate_status = 'Graduate'
                        results6 = db_manager.execute_select_query(sql_graduate_tutors, (graduate_status,))
                        query_results['graduate_tutors'] = results6
                        print("Results (Graduate Tutors):") # Print header
                        if results6:
                            print(f"{'FirstName':<15} | {'LastName':<15}")
                            print("-" * 32)
                            for row in results6: print(f"{row[0]:<15} | {row[1]:<15}")
                        elif results6 is None:
                             print("Query execution failed.")
                        else:
                            print("No matching records found.")

                    # Task 7: Download and submit a legible file (Handled by user saving this script and its output)
                    print("\n--- Task 7: Submit Code and Results (User task) ---")
                    print(f"Output log file generated at: {os.path.abspath(OUTPUT_LOG_FILE)}")


            # Catch errors specific to database operations within the 'with' block
            except ConnectionError as ce:
                 print(f"\nDATABASE CONNECTION ERROR: {ce}. Aborting database operations.", file=sys.stderr)
            except sqlite3.Error as sql_e:
                print(f"\nUNHANDLED SQLITE ERROR: {sql_e}. Aborting database operations.", file=sys.stderr)
            except Exception as e:
                 # Catch unexpected errors during database processing
                print(f"\nUNEXPECTED ERROR during database operations: {type(e).__name__} - {e}. Aborting.", file=sys.stderr)
                # import traceback # Uncomment for detailed debugging
                # traceback.print_exc(file=sys.stderr) # Print traceback to the log file

            # This print statement will now go to the file
            print("\n="*2)
            print(" Script execution finished. ")
            print("="*60)
            # --- End of Database Operations ---

    except Exception as e:
        # If error happens *outside* the main db logic (e.g., file open error)
        # Try to print to original stderr
        print(f"FATAL ERROR during script execution or redirection setup: {e}", file=original_stderr)
        # Also attempt to write to the log file if possible, otherwise it's lost
        try:
             with open(OUTPUT_LOG_FILE, 'a', encoding='utf-8') as errfile:
                  print(f"\nFATAL ERROR during script execution or redirection setup: {e}\n", file=errfile)
        except:
             pass # Ignore errors writing the error message itself
    finally:
        # --- Restore Output Streams ---
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        # This final message goes back to the console
        print(f"--- Script finished. Output was redirected to '{OUTPUT_LOG_FILE}' ---")
