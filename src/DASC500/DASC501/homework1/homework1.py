"""
TutoringBusiness Database Management - OOP Approach

This script utilizes an object-oriented design to manage the creation,
data population, and querying of an SQLite database for a tutoring business.
Each step of the original request is encapsulated within a method of the
TutoringDatabaseManager class.

Author: Gemini AI (Expert in Data Analytics and Python Programming)
Date: 2025-04-04
"""

import sqlite3
import pandas as pd
import os
import sys # Used for better error reporting location


class TutoringDatabaseManager:
    """
    Manages operations for the Tutoring Business SQLite database using OOP.

    Handles database connection lifecycle, table schema definition, data ingestion
    from Excel, and execution of analytical queries. Designed for use as a
    context manager (`with` statement) to ensure robust resource management,
    specifically the automatic closing of the database connection.

    Attributes:
        db_path (str): Filesystem path to the SQLite database file.
        excel_path (str): Filesystem path to the Excel data source.
        conn (sqlite3.Connection | None): Active SQLite database connection object.
        cursor (sqlite3.Cursor | None): Cursor object for executing SQL commands.
    """

    def __init__(self, db_path: str = 'TutoringBusiness.db', excel_path: str = 'tblTutor.xlsx'):
        """
        Initializes the TutoringDatabaseManager.

        Args:
            db_path (str): The path for the SQLite database file.
                           If the file doesn't exist, it will be created upon connection.
            excel_path (str): The path to the Excel file containing tutor data.
        """
        self.db_path = db_path
        self.excel_path = excel_path
        self.conn = None
        self.cursor = None
        self._connect()  # Establish connection immediately upon instantiation

    def _connect(self):
        """Establishes the database connection and initializes the cursor."""
        try:
            # sqlite3.connect implicitly handles Problem 1: creating the DB file if absent.
            self.conn = sqlite3.connect(self.db_path)
            # Setting row_factory can be useful for dictionary-like row access,
            # but default tuple access is used here for simplicity.
            # self.conn.row_factory = sqlite3.Row
            self.cursor = self.conn.cursor()
            print(f"INFO: Successfully connected to database: '{self.db_path}'")
        except sqlite3.Error as e:
            print(f"ERROR: Failed to connect to database '{self.db_path}'. Error: {e}", file=sys.stderr)
            self.conn = None # Ensure connection state reflects failure
            raise ConnectionError(f"Database connection failed: {e}") from e

    def close_connection(self):
        """Closes the database connection if it is currently open."""
        if self.conn:
            try:
                self.conn.commit() # Ensure any pending transactions are committed before closing
                self.conn.close()
                print(f"INFO: Database connection to '{self.db_path}' closed.")
            except sqlite3.Error as e:
                print(f"WARN: Error closing database connection: {e}", file=sys.stderr)
            finally:
                self.conn = None # Reset attributes regardless of close success
                self.cursor = None

    def __enter__(self):
        """Enables use of the class as a context manager (e.g., `with TutoringDatabaseManager(...)`)."""
        if not self.conn or not self.cursor:
            print("INFO: Attempting to re-establish database connection.")
            self._connect() # Ensure connection is active when entering context
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Called upon exiting the `with` block, ensures connection closure."""
        print("INFO: Exiting context manager, ensuring database connection closure.")
        self.close_connection()
        # Returning False (or None implicitly) ensures exceptions are propagated
        if exc_type:
             print(f"ERROR: An exception occurred within the context manager: {exc_val}", file=sys.stderr)
        return False

    # --- Problem-Specific Methods ---

    def problem_1_confirm_db_creation(self) -> bool:
        """
        Confirms database file creation/connection. (Handled by _connect).
        """
        print("\n--- Task 1: Confirm Database File Creation/Connection ---")
        if self.conn and self.cursor:
            print(f"SUCCESS: Database file '{self.db_path}' exists and connection is active.")
            return True
        else:
            print(f"FAILURE: Database connection to '{self.db_path}' is not active.")
            return False

    def problem_2_create_table(self, table_name: str = 'tblTutor') -> None:
        """
        Creates the tutor table with the specified schema if it doesn't exist.

        Args:
            table_name (str): The name for the tutor table.
        """
        print(f"\n--- Task 2: Create Table '{table_name}' ---")
        if not self.cursor:
            raise ConnectionError("Database cursor is unavailable. Cannot create table.")

        # Using TEXT for HireDate allows flexibility but requires consistent formatting (YYYY-MM-DD).
        # Consider REAL (Julian day) or INTEGER (Unix time) for more complex date arithmetic in SQL.
        # Added NOT NULL constraints for essential fields like names.
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            TutorID      INTEGER PRIMARY KEY,
            FirstName    TEXT NOT NULL,
            LastName     TEXT NOT NULL,
            Major        TEXT,
            YearInSchool TEXT,
            School       TEXT,
            HireDate     TEXT -- Recommended format: 'YYYY-MM-DD' for reliable sorting/comparison
        );
        """
        try:
            self.cursor.execute(create_table_sql)
            # DDL (Data Definition Language) like CREATE TABLE is often transactional,
            # but explicit commit ensures visibility/persistence across systems.
            self.conn.commit()
            print(f"SUCCESS: Table '{table_name}' ensured/created in the database.")
        except sqlite3.Error as e:
            print(f"FAILURE: Error creating table '{table_name}': {e}", file=sys.stderr)
            raise # Propagate the error to halt execution if table creation fails

    def problem_3_import_data(self, table_name: str = 'tblTutor'):
        """
        Imports data from the configured Excel file into the specified table.
        Uses pandas for efficient data loading and basic transformation.
        Handles potential file absence and data type conversion issues.

        Args:
            table_name (str): The name of the target table for data import.

        Returns:
            bool: True if import was successful or file was empty/not found (no data to import),
                  False if a critical error occurred during import.
        """
        print(f"\n--- Task 3: Import Data from Excel '{self.excel_path}' into '{table_name}' ---")
        if not self.cursor:
            raise ConnectionError("Database cursor is unavailable. Cannot import data.")

        if not os.path.exists(self.excel_path):
            print(f"WARN: Excel file '{self.excel_path}' not found. No data imported.")
            return True # Not an error state, just no action taken.

        try:
            df = pd.read_excel(self.excel_path, engine='openpyxl')
            print(f"INFO: Read {len(df)} rows from '{self.excel_path}'.")

            if df.empty:
                print("INFO: Excel file is empty. No data imported.")
                return True

            # --- Data Preprocessing & Validation ---
            # Define expected columns based on the table schema
            expected_columns = ['TutorID', 'FirstName', 'LastName', 'Major', 'YearInSchool', 'School', 'HireDate']

            # Check for missing columns (critical)
            missing_cols = [col for col in expected_columns if col not in df.columns]
            if missing_cols:
                print(f"ERROR: Excel file '{self.excel_path}' is missing required columns: {missing_cols}. Import aborted.", file=sys.stderr)
                return False

            # Ensure HireDate is formatted correctly (YYYY-MM-DD text)
            if 'HireDate' in df.columns:
                original_dtype = df['HireDate'].dtype
                try:
                    # Convert to datetime, coercing errors (invalid dates become NaT)
                    df['HireDate'] = pd.to_datetime(df['HireDate'], errors='coerce')
                    # Format valid dates to string; NaT becomes None (which translates to SQL NULL)
                    df['HireDate'] = df['HireDate'].dt.strftime('%Y-%m-%d')
                    print("INFO: 'HireDate' column processed for YYYY-MM-DD format.")
                except Exception as date_err:
                    # Catch unexpected errors during conversion
                     print(f"WARN: Could not reliably convert 'HireDate' column (original type: {original_dtype}). Potential data inconsistencies: {date_err}", file=sys.stderr)
                     # Proceeding with potentially mixed/incorrect date formats if conversion fails

            # Select and order columns to match table definition for safe insertion
            # Replace pandas NaN/NaT with None for SQLite compatibility
            df_ordered = df[expected_columns].where(pd.notnull(df), None)

            # Convert DataFrame rows to a list of tuples for `executemany`
            data_to_insert = [tuple(row) for row in df_ordered.to_numpy()]

            # --- Database Insertion ---
            # Using INSERT OR REPLACE handles potential primary key conflicts by replacing existing rows.
            # Consider 'INSERT OR IGNORE' if you want to keep existing rows and skip duplicates.
            # Use simple 'INSERT' if TutorIDs are guaranteed unique or you want errors on duplicates.
            insert_sql = f"""
            INSERT OR REPLACE INTO {table_name} (TutorID, FirstName, LastName, Major, YearInSchool, School, HireDate)
            VALUES (?, ?, ?, ?, ?, ?, ?);
            """

            # Use `executemany` for significantly better performance on bulk inserts
            self.cursor.executemany(insert_sql, data_to_insert)
            self.conn.commit() # Commit transaction after successful insertion
            print(f"SUCCESS: Data imported successfully into '{table_name}'. {len(data_to_insert)} rows affected.")
            return True

        except FileNotFoundError:
             # This case is handled by the os.path.exists check, but included for completeness
             print(f"WARN: Excel file '{self.excel_path}' not found. No data imported.")
             return True
        except pd.errors.EmptyDataError:
            print("INFO: Excel file is empty. No data imported.")
            return True
        except Exception as e:
            # Catch potential errors during file reading (e.g., corrupted file),
            # data processing, or database insertion.
            print(f"FAILURE: An error occurred during data import: {e}", file=sys.stderr)
            try:
                self.conn.rollback() # Roll back the transaction on error
                print("INFO: Database transaction rolled back.")
            except sqlite3.Error as rb_err:
                print(f"WARN: Error during rollback: {rb_err}", file=sys.stderr)
            return False

    def _execute_query(self, sql: str, params: tuple = None) -> list | None:
        """
        Private helper method to execute a SELECT query and fetch all results.

        Args:
            sql (str): The SELECT SQL query to execute.
            params (tuple, optional): Parameters for parameterized queries. Defaults to None.

        Returns:
            list | None: A list of tuples representing the fetched rows,
                         or None if an error occurs.
        """
        if not self.cursor:
            raise ConnectionError("Database cursor is unavailable. Cannot execute query.")
        try:
            if params:
                self.cursor.execute(sql, params)
            else:
                self.cursor.execute(sql)
            results = self.cursor.fetchall()
            return results
        except sqlite3.Error as e:
            print(f"ERROR: Failed to execute query. Error: {e}", file=sys.stderr)
            print(f"Query: {sql}", file=sys.stderr)
            if params: print(f"Params: {params}", file=sys.stderr)
            return None # Indicate query failure

    def problem_4_query_hired_after(self, table_name: str = 'tblTutor', hire_date_threshold: str = '2017-04-30'):
        """
        Queries tutors hired strictly after a specified date.

        Args:
            table_name (str): The name of the tutor table.
            hire_date_threshold (str): The date threshold in 'YYYY-MM-DD' format.

        Returns:
            list | None: List of (FirstName, LastName, HireDate) tuples, or None on error.
        """
        print(f"\n--- Task 4: Query Tutors Hired After {hire_date_threshold} ---")
        # Comparing dates stored as 'YYYY-MM-DD' text works correctly with standard string comparison.
        query_sql = f"""
        SELECT FirstName, LastName, HireDate
        FROM {table_name}
        WHERE HireDate > ?;
        """
        # Use parameterization to prevent SQL injection vulnerabilities
        results = self._execute_query(query_sql, (hire_date_threshold,))

        if results is not None:
            print(f"SUCCESS: Query executed. Found {len(results)} records.")
        else:
            print("FAILURE: Query execution failed.")
        return results


    def problem_5_query_distinct_majors(self, table_name: str = 'tblTutor'):
        """
        Retrieves a list of unique 'Major' values present in the tutor table.

        Args:
            table_name (str): The name of the tutor table.

        Returns:
            list | None: List of tuples, each containing a distinct major, or None on error.
                         Majors can be None if the column allows NULLs.
        """
        print(f"\n--- Task 5: Query Distinct Tutor Majors from '{table_name}' ---")
        query_sql = f"SELECT DISTINCT Major FROM {table_name};"
        results = self._execute_query(query_sql)

        if results is not None:
             distinct_majors = [row[0] for row in results] # Extract majors from tuples
             print(f"SUCCESS: Query executed. Found {len(distinct_majors)} distinct major values.")
        else:
            print("FAILURE: Query execution failed.")
        return results # Return list of tuples as fetched

    def problem_6_query_graduate_tutors(self, table_name: str = 'tblTutor', year_status: str = 'Graduate'):
        """
        Queries tutors whose 'YearInSchool' status matches the specified value (case-sensitive).

        Args:
            table_name (str): The name of the tutor table.
            year_status (str): The specific 'YearInSchool' value to filter by.

        Returns:
            list | None: List of (FirstName, LastName) tuples for matching tutors, or None on error.
        """
        print(f"\n--- Task 6: Query Tutors where YearInSchool is '{year_status}' ---")
        # Direct string comparison is case-sensitive by default in SQLite unless collation specified.
        # Use LOWER() function on both column and value for case-insensitive search if needed:
        # WHERE LOWER(YearInSchool) = LOWER(?)
        query_sql = f"""
        SELECT FirstName, LastName
        FROM {table_name}
        WHERE YearInSchool = ?;
        """
        results = self._execute_query(query_sql, (year_status,))

        if results is not None:
            print(f"SUCCESS: Query executed. Found {len(results)} records.")
        else:
            print("FAILURE: Query execution failed.")
        return results


# --- Main Execution Block ---
if __name__ == "__main__":
    # Configuration
    DATABASE_FILE = 'TutoringBusiness_OOP.db' # Use a distinct name if needed
    EXCEL_DATA_FILE = 'tblTutor.xlsx'
    TABLE_NAME = 'tblTutor'

    print("="*60)
    print(" Tutoring Business Database Management Script - Main Execution ")
    print(f" Database file: {DATABASE_FILE}")
    print(f" Excel source:  {EXCEL_DATA_FILE}")
    print(f" Execution Time: {pd.Timestamp.now()}") # Leverage pandas for timestamp
    print("="*60)

    query_results = {} # Dictionary to store results for potential later use

    try:
        # Utilize the context manager for robust connection handling
        with TutoringDatabaseManager(db_path=DATABASE_FILE, excel_path=EXCEL_DATA_FILE) as db_manager:

            # Execute tasks sequentially
            db_manager.problem_1_confirm_db_creation()
            db_manager.problem_2_create_table(table_name=TABLE_NAME)

            # Import data - proceed with queries only if import doesn't critically fail
            import_status = db_manager.problem_3_import_data(table_name=TABLE_NAME)

            if not import_status and os.path.exists(EXCEL_DATA_FILE):
                 print("\nCRITICAL: Halting query execution due to data import errors.")
            else:
                # --- Execute and Display Queries ---
                print("\n--- Proceeding with Data Queries ---")

                # Query 4: Hired After Date
                results4 = db_manager.problem_4_query_hired_after(table_name=TABLE_NAME)
                query_results['hired_after_2017_04'] = results4
                if results4 is not None:
                    print("Results (Hired After 2017-04-30):")
                    if results4:
                        print(f"{'FirstName':<15} | {'LastName':<15} | {'HireDate'}")
                        print("-" * 47)
                        for row in results4:
                            print(f"{row[0]:<15} | {row[1]:<15} | {row[2]}")
                    else:
                        print("No matching records found.")

                # Query 5: Distinct Majors
                results5 = db_manager.problem_5_query_distinct_majors(table_name=TABLE_NAME)
                query_results['distinct_majors'] = results5
                if results5 is not None:
                    print("\nResults (Distinct Majors):")
                    print("-------------------------")
                    if results5:
                        # Extract major from each tuple, handling potential None values
                        distinct_majors_list = [str(row[0]) if row[0] is not None else "N/A" for row in results5]
                        for major in sorted(distinct_majors_list): # Sort for consistent output
                            print(major)
                    else:
                        print("No distinct majors found.")

                # Query 6: Graduate Tutors
                results6 = db_manager.problem_6_query_graduate_tutors(table_name=TABLE_NAME, year_status='Graduate')
                query_results['graduate_tutors'] = results6
                if results6 is not None:
                    print("\nResults (Graduate Tutors):")
                    if results6:
                        print(f"{'FirstName':<15} | {'LastName':<15}")
                        print("-" * 32)
                        for row in results6:
                            print(f"{row[0]:<15} | {row[1]:<15}")
                    else:
                        print("No matching records found.")

    except ConnectionError as ce:
         print(f"\nDATABASE CONNECTION ERROR: {ce}. Aborting.", file=sys.stderr)
    except sqlite3.Error as sql_e:
        # Catch SQLite errors not caught within methods (e.g., during commit/rollback)
        print(f"\nUNHANDLED SQLITE ERROR: {sql_e}. Aborting.", file=sys.stderr)
    except Exception as e:
        # Catch any other unexpected errors during the process
        print(f"\nUNEXPECTED ERROR: {type(e).__name__} - {e}. Aborting.", file=sys.stderr)
        # Optionally print traceback for debugging:
        # import traceback
        # traceback.print_exc()

    finally:
        print("\n="*60)
        print(" Script execution finished. ")
        print("="*60)