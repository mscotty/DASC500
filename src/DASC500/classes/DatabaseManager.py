# filename: DatabaseManager.py
import sqlite3
import os
import sys
import shutil
import logging  # Added logging
import pandas as pd  # Using pandas for Excel/CSV import/export
import csv  # For CSV handling
import re  # For potential regex validation
from typing import (
    List,
    Tuple,
    Optional,
    Any,
    Dict,
    Union,
    Literal,
    Callable,
)  # For type hinting

# Define valid SQL insert strategies for type hinting and validation
SqlInsertStrategy = Literal["INSERT", "INSERT OR REPLACE", "INSERT OR IGNORE"]
ValidationErrorStrategy = Literal[
    "skip", "fail", "log"
]  # How to handle validation errors during import

# --- Basic Logger Setup ---
# Configure a default logger format
log_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class DatabaseManager:
    """
    Manages generic SQLite database operations including connection handling,
    SQL execution, schema introspection, data import/export via Excel/CSV,
    validation, backup, and file renaming. Uses Python's logging module.

    Designed for use as a context manager (`with` statement).

    Attributes:
        db_path (str): Path to the SQLite database file.
        excel_path (Optional[str]): Default path to an Excel file for import/export.
        conn (Optional[sqlite3.Connection]): Active SQLite connection.
        cursor (Optional[sqlite3.Cursor]): Cursor for executing SQL.
        logger (logging.Logger): Logger instance for this manager.
    """

    def __init__(
        self,
        db_path: str,
        excel_path: Optional[str] = None,  # Default Excel path (optional)
        log_file: Optional[str] = None,
        log_level: int = logging.INFO,
    ):
        """
        Initializes the DatabaseManager and configures logging.

        Args:
            db_path (str): The path for the SQLite database file.
                           The directory must exist.
            excel_path (Optional[str]): Default path to an Excel file for
                                        import/export operations.
            log_file (Optional[str]): Path to a file for logging output.
                                      If None, logs only to console.
            log_level (int): The logging level (e.g., logging.INFO, logging.DEBUG).
                             Defaults to logging.INFO.
        """
        self.db_path = db_path
        self.excel_path = excel_path  # Store default excel path
        self.conn = None
        self.cursor = None

        # --- Logger Configuration ---
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(log_level)

        if not self.logger.hasHandlers():
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(log_formatter)
            self.logger.addHandler(console_handler)

            if log_file:
                try:
                    log_dir = os.path.dirname(log_file)
                    if log_dir and not os.path.exists(log_dir):
                        os.makedirs(log_dir, exist_ok=True)
                    file_handler = logging.FileHandler(log_file, mode="a")
                    file_handler.setFormatter(log_formatter)
                    self.logger.addHandler(file_handler)
                    self.logger.info(f"Logging configured to file: {log_file}")
                except Exception as e:
                    self.logger.error(
                        f"Failed to configure file logging to '{log_file}': {e}",
                        exc_info=False,
                    )

        self.logger.info(
            f"DatabaseManager initialized for DB: '{self.db_path}'"
            f"{f' | Default Excel: {self.excel_path}' if self.excel_path else ''}"
        )

        # --- Initial Checks ---
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            self.logger.error(f"Directory for database '{db_dir}' does not exist.")
            raise FileNotFoundError(
                f"Directory for database '{db_dir}' does not exist."
            )

    def connect(self) -> None:
        """Establishes the database connection."""
        if self.conn:
            self.logger.debug("Connection already established.")
            return
        try:
            self.logger.info(f"Connecting to database: '{self.db_path}'...")
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row  # Access columns by name
            self.cursor = self.conn.cursor()
            self.logger.info("Database connection established successfully.")
            # Optionally enable foreign keys by default
            # self.execute_sql("PRAGMA foreign_keys = ON;")
        except sqlite3.Error as e:
            self.logger.error(
                f"Failed to connect to database '{self.db_path}'. Error: {e}",
                exc_info=True,
            )
            self.conn = None
            self.cursor = None
            raise ConnectionError(f"Database connection failed: {e}") from e

    def close(self) -> None:
        """Commits changes and closes the database connection."""
        if self.conn:
            try:
                self.logger.info(
                    "Committing final changes (if any) and closing connection..."
                )
                self.conn.commit()
                self.conn.close()
                self.logger.info(f"Database connection to '{self.db_path}' closed.")
            except sqlite3.Error as e:
                self.logger.warning(
                    f"Error during closing sequence (commit/close): {e}", exc_info=True
                )
            finally:
                self.conn = None
                self.cursor = None
        else:
            self.logger.debug("No active connection to close.")

    def __enter__(self):
        """Enters the runtime context, establishing connection."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exits the runtime context, ensuring connection closure."""
        is_exception = exc_type is not None
        if is_exception:
            self.logger.error(
                f"Exception occurred within context: {exc_type.__name__}({exc_val})",
                exc_info=(exc_type, exc_val, exc_tb),
            )
        else:
            self.logger.debug("Exiting context manager normally.")
        self.close()
        return False  # Propagate exceptions

    def execute_sql(
        self, sql_command: str, params: Optional[Tuple[Any, ...]] = None
    ) -> None:
        """Executes a single SQL command."""
        if not self.cursor:
            self.logger.error("execute_sql called but database cursor is unavailable.")
            raise ConnectionError("Database cursor is unavailable. Cannot execute SQL.")
        try:
            self.logger.debug(
                f"Executing SQL: {sql_command[:150]}{'...' if len(sql_command)>150 else ''} | Params: {params if params else 'None'}"
            )
            if params:
                self.cursor.execute(sql_command, params)
            else:
                self.cursor.execute(sql_command)
            self.logger.debug(f"SQL executed successfully.")
        except sqlite3.Error as e:
            self.logger.error(
                f"Failed to execute SQL command. Error: {e}", exc_info=True
            )
            self.logger.error(f"Failed SQL: {sql_command}")
            if params:
                self.logger.error(f"Failed Params: {params}")
            raise

    def execute_script(self, sql_script: str) -> None:
        """Executes multiple SQL statements from a single string."""
        if not self.cursor:
            self.logger.error(
                "execute_script called but database cursor is unavailable."
            )
            raise ConnectionError(
                "Database cursor is unavailable. Cannot execute script."
            )
        try:
            self.logger.info(f"Executing SQL script (length: {len(sql_script)})...")
            self.cursor.executescript(sql_script)
            self.logger.info("SQL script executed successfully.")
        except sqlite3.Error as e:
            self.logger.error(
                f"Failed to execute SQL script. Error: {e}", exc_info=True
            )
            raise

    def commit(self) -> None:
        """Commits the current transaction."""
        if not self.conn:
            self.logger.error("commit called but database connection is unavailable.")
            raise ConnectionError("Database connection is unavailable. Cannot commit.")
        try:
            self.logger.debug("Committing transaction...")
            self.conn.commit()
            self.logger.info("Transaction committed successfully.")
        except sqlite3.Error as e:
            self.logger.error(
                f"Failed to commit transaction. Error: {e}", exc_info=True
            )
            raise

    def rollback(self) -> None:
        """Rolls back the current transaction."""
        if not self.conn:
            self.logger.error("rollback called but database connection is unavailable.")
            raise ConnectionError(
                "Database connection is unavailable. Cannot rollback."
            )
        try:
            self.logger.info("Rolling back transaction...")
            self.conn.rollback()
            self.logger.info("Transaction rolled back successfully.")
        except sqlite3.Error as e:
            self.logger.error(
                f"Failed to rollback transaction. Error: {e}", exc_info=True
            )
            # Decide if rollback failure should raise an exception
            # raise

    def execute_select_query(
        self,
        sql_query: str,
        params: Optional[Tuple[Any, ...]] = None,
        return_dataframe: bool = True,
    ) -> Optional[Union[pd.DataFrame, List[sqlite3.Row]]]:
        """Executes a SELECT query and fetches all results."""
        if not self.cursor:
            self.logger.error(
                "execute_select_query called but database cursor is unavailable."
            )
            raise ConnectionError(
                "Database cursor is unavailable. Cannot execute query."
            )
        try:
            self.logger.debug(
                f"Executing SELECT: {sql_query[:150]}{'...' if len(sql_query)>150 else ''} | Params: {params if params else 'None'}"
            )
            if params:
                self.cursor.execute(sql_query, params)
            else:
                self.cursor.execute(sql_query)
            results = self.cursor.fetchall()
            self.logger.info(
                f"SELECT query executed successfully. Found {len(results)} records."
            )

            if return_dataframe:
                if "pd" not in globals():
                    self.logger.error(
                        "Pandas library is required to return DataFrames but not found."
                    )
                    raise ImportError(
                        "Pandas library is required to return DataFrames."
                    )
                if results:
                    column_names = [
                        description[0] for description in self.cursor.description
                    ]
                    df = pd.DataFrame(results, columns=column_names)
                    self.logger.debug(f"Returning {len(df)} results as DataFrame.")
                    return df
                else:
                    self.logger.debug("Returning empty DataFrame (no results).")
                    return pd.DataFrame()  # Return empty DataFrame for consistency
            else:
                self.logger.debug(
                    f"Returning {len(results)} results as list of Row objects."
                )
                return results  # List of sqlite3.Row objects

        except sqlite3.Error as e:
            self.logger.error(
                f"Failed to execute SELECT query. Error: {e}", exc_info=True
            )
            self.logger.error(f"Failed Query: {sql_query}")
            if params:
                self.logger.error(f"Failed Params: {params}")
            return None  # Indicate query execution error

    # --- Introspection Methods ---

    def table_exists(self, table_name: str) -> bool:
        """Checks if a table exists in the database."""
        self.logger.debug(f"Checking existence of table: {table_name}")
        query = "SELECT name FROM sqlite_master WHERE type='table' AND name=?;"
        try:
            if not self.cursor:
                raise ConnectionError("Database cursor is unavailable.")
            self.cursor.execute(query, (table_name,))
            result = self.cursor.fetchone()
            exists = result is not None
            self.logger.debug(f"Table '{table_name}' exists: {exists}")
            return exists
        except sqlite3.Error as e:
            self.logger.error(
                f"Error checking existence for table '{table_name}': {e}", exc_info=True
            )
            return False
        except ConnectionError as ce:
            self.logger.error(
                f"Cannot check table existence for '{table_name}', cursor unavailable: {ce}"
            )
            raise  # Re-raise connection error

    def get_table_names(self) -> List[str]:
        """Retrieves the names of all user-defined tables."""
        self.logger.debug("Fetching table names...")
        query = "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"  # Exclude system tables
        results = self.execute_select_query(query, return_dataframe=False)
        if results is None:
            return []
        table_names = [row["name"] for row in results]
        self.logger.info(f"Found {len(table_names)} user table(s): {table_names}")
        return table_names

    def get_table_columns(self, table_name: str) -> Optional[List[Tuple[str, str]]]:
        """Retrieves column names and declared types using PRAGMA table_info."""
        self.logger.debug(f"Fetching columns for table: {table_name}...")
        # Use table_exists helper first
        if not self.table_exists(table_name):
            self.logger.warning(
                f"Table '{table_name}' does not exist. Cannot fetch columns."
            )
            return None

        query = f'PRAGMA table_info("{table_name}");'  # Quote table name
        if not self.cursor:
            self.logger.error(
                f"Cannot fetch columns for '{table_name}', cursor unavailable."
            )
            raise ConnectionError("Database cursor is unavailable.")
        try:
            self.cursor.execute(query)
            columns_info = self.cursor.fetchall()
            if (
                not columns_info
            ):  # Should not happen if table_exists passed, but good check
                self.logger.warning(
                    f"PRAGMA table_info returned no columns for '{table_name}'."
                )
                return None
            columns = [(col["name"], col["type"]) for col in columns_info]
            self.logger.debug(f"Columns for '{table_name}': {columns}")
            return columns
        except sqlite3.Error as e:
            self.logger.error(
                f"Failed fetching columns for table '{table_name}': {e}", exc_info=True
            )
            return None

    def get_all_tables_and_columns(self) -> Dict[str, List[Tuple[str, str]]]:
        """Retrieves all user tables and their columns/types."""
        self.logger.info("Fetching all tables and their columns...")
        all_tables_info = {}
        table_names = self.get_table_names()
        if not table_names:
            self.logger.info("No user tables found.")
            return all_tables_info
        for table_name in table_names:
            columns = self.get_table_columns(table_name)
            if columns is not None:
                all_tables_info[table_name] = columns
            # Warning logged by get_table_columns if it fails
        self.logger.info(f"Retrieved column info for {len(all_tables_info)} table(s).")
        return all_tables_info

    # --- Data Import/Export Methods ---

    def _validate_dataframe(
        self,
        df: pd.DataFrame,
        validation_rules: Optional[Dict[str, Callable[[Any], bool]]],
        on_error: ValidationErrorStrategy = "log",
    ) -> pd.DataFrame:
        """Internal helper to validate DataFrame rows before import."""
        if not validation_rules:
            return df  # No validation needed

        valid_rows_mask = pd.Series(True, index=df.index)
        self.logger.info(
            f"Applying {len(validation_rules)} validation rule(s) to {len(df)} rows..."
        )
        validation_errors = 0

        for col_name, validator_func in validation_rules.items():
            if col_name not in df.columns:
                self.logger.warning(
                    f"Validation rule specified for non-existent column '{col_name}'. Skipping."
                )
                continue

            self.logger.debug(f"Validating column '{col_name}'...")
            try:
                # Apply validator function; handle potential exceptions in the function itself if needed
                col_valid_mask = df[col_name].apply(
                    lambda x: validator_func(x) if pd.notna(x) else True
                )  # Assume NaN/None is valid unless rule handles it
                failed_indices = df.index[
                    ~col_valid_mask & valid_rows_mask
                ]  # Only check rows not already marked invalid

                if not failed_indices.empty:
                    self.logger.warning(
                        f"Validation failed for column '{col_name}' on {len(failed_indices)} row(s). Indices: {failed_indices.tolist()}"
                    )
                    validation_errors += len(failed_indices)
                    if on_error == "fail":
                        self.logger.error(
                            "Validation failed and on_error='fail'. Aborting import."
                        )
                        raise ValueError(
                            f"Validation failed for column '{col_name}'. Aborting."
                        )
                    elif on_error == "skip" or on_error == "log":
                        valid_rows_mask &= col_valid_mask  # Update overall mask
                    # 'log' strategy implicitly handled by logging the warning

            except Exception as e:
                self.logger.error(
                    f"Error during validation of column '{col_name}': {e}",
                    exc_info=True,
                )
                raise ValueError(
                    f"Error applying validation function for column '{col_name}'."
                ) from e

        if validation_errors > 0:
            self.logger.warning(f"Total validation failures: {validation_errors}")
            if on_error == "skip":
                original_count = len(df)
                df_validated = df[valid_rows_mask].copy()
                self.logger.info(
                    f"Skipping {original_count - len(df_validated)} invalid rows due to on_error='skip'."
                )
                return df_validated
            elif on_error == "log":
                self.logger.info(
                    "Proceeding with all rows despite validation warnings (on_error='log')."
                )
                # Continue with the original df, errors were just logged
                return df
        else:
            self.logger.info("All rows passed validation.")

        return df  # Return original if on_error='log' or no errors

    def import_data_from_excel(
        self,
        table_name: str,
        excel_file_path: Optional[str] = None,
        sheet_name: Union[str, int] = 0,
        expected_columns: Optional[List[str]] = None,
        clear_table_first: bool = False,
        insert_strategy: SqlInsertStrategy = "INSERT OR REPLACE",
        validation_rules: Optional[Dict[str, Callable[[Any], bool]]] = None,
        on_validation_error: ValidationErrorStrategy = "log",
    ) -> bool:
        """Imports data from Excel, with optional pre-validation."""
        effective_excel_path = excel_file_path or self.excel_path
        self.logger.info(
            f"Starting Excel import: '{effective_excel_path}'[Sheet:{sheet_name}] -> Table:'{table_name}'"
        )
        self.logger.info(
            f"Import options - Strategy: {insert_strategy}, Clear: {clear_table_first}, Validation: {'Yes' if validation_rules else 'No'} (on error: {on_validation_error})"
        )

        if not self.cursor:
            raise ConnectionError("Database cursor is unavailable.")
        if not effective_excel_path:
            raise ValueError("No Excel file path specified for import.")
        if not os.path.exists(effective_excel_path):
            raise FileNotFoundError(f"Excel file '{effective_excel_path}' not found.")
        if "pd" not in globals():
            raise ImportError("Pandas library is required for Excel import.")
        try:
            import openpyxl
        except ImportError:
            raise ImportError(
                "The 'openpyxl' library is required for reading .xlsx files."
            )

        try:
            df = pd.read_excel(
                effective_excel_path, sheet_name=sheet_name, engine="openpyxl"
            )
            self.logger.info(f"Read {len(df)} rows from Excel sheet '{sheet_name}'.")
            if df.empty:
                self.logger.info("Excel sheet is empty. No data to import.")
                return True

            # Column Selection/Validation
            actual_excel_columns = df.columns.tolist()
            db_columns_info = self.get_table_columns(table_name)
            if not db_columns_info:
                self.logger.error(
                    f"Could not retrieve columns for target table '{table_name}'. Aborting import."
                )
                return False
            db_column_names = [col[0] for col in db_columns_info]

            if expected_columns:
                missing_cols = [
                    col for col in expected_columns if col not in actual_excel_columns
                ]
                if missing_cols:
                    self.logger.error(
                        f"Excel sheet missing expected columns: {missing_cols}. Aborting import."
                    )
                    return False
                cols_to_insert = expected_columns
            else:
                # Use columns present in both Excel and DB table
                cols_to_insert = [
                    col for col in actual_excel_columns if col in db_column_names
                ]
                ignored_excel_cols = [
                    col for col in actual_excel_columns if col not in db_column_names
                ]
                if ignored_excel_cols:
                    self.logger.warning(
                        f"Ignoring Excel columns not found in table '{table_name}': {ignored_excel_cols}"
                    )
                if not cols_to_insert:
                    self.logger.error(
                        f"No matching columns found between Excel and table '{table_name}'. Aborting."
                    )
                    return False

            df_processed = df[cols_to_insert].copy()

            # --- Pre-import Validation ---
            df_validated = self._validate_dataframe(
                df_processed, validation_rules, on_validation_error
            )
            if df_validated.empty and not df_processed.empty:
                self.logger.warning(
                    "All rows failed validation or were skipped. No data will be imported."
                )
                return True  # Technically successful (no errors), but nothing imported
            if df_validated.empty and df_processed.empty:
                self.logger.info("Initial DataFrame was empty. No data to import.")
                return True

            # Basic Data Cleaning (NaN -> None) on validated data
            df_final = df_validated.where(pd.notnull(df_validated), None)
            data_to_insert = [tuple(row) for row in df_final.to_numpy()]

            if not data_to_insert:
                self.logger.info(
                    "No valid data remaining after validation/cleaning. Nothing to insert."
                )
                return True

            # Database Insertion
            if clear_table_first:
                if self.table_exists(table_name):
                    self.logger.info(
                        f"Clearing all data from table '{table_name}' before import..."
                    )
                    delete_sql = f'DELETE FROM "{table_name}";'
                    try:
                        self.execute_sql(delete_sql)
                        self.commit()  # Commit the delete
                        self.logger.info(f"Table '{table_name}' cleared successfully.")
                    except Exception as del_e:
                        self.logger.error(
                            f"Failed to clear table '{table_name}': {del_e}",
                            exc_info=True,
                        )
                        self.rollback()
                        return False
                else:
                    self.logger.warning(
                        f"Table '{table_name}' does not exist. Cannot clear."
                    )
                    # Might want to raise error or just proceed to let CREATE happen if schema is dynamic

            # Prepare and execute bulk insert
            placeholders = ", ".join(["?"] * len(cols_to_insert))
            cols_string = ", ".join(
                f'"{col}"' for col in cols_to_insert
            )  # Use final columns used
            if insert_strategy not in [
                "INSERT",
                "INSERT OR REPLACE",
                "INSERT OR IGNORE",
            ]:
                self.logger.warning(
                    f"Invalid insert_strategy '{insert_strategy}'. Defaulting to 'INSERT OR REPLACE'."
                )
                insert_strategy = "INSERT OR REPLACE"
            insert_sql = f'{insert_strategy} INTO "{table_name}" ({cols_string}) VALUES ({placeholders});'

            self.logger.info(
                f"Executing bulk insert ({insert_strategy}) with {len(data_to_insert)} validated rows into '{table_name}'..."
            )
            self.cursor.executemany(insert_sql, data_to_insert)
            self.commit()
            self.logger.info(
                f"Bulk insert operation completed for table '{table_name}'."
            )
            return True

        except FileNotFoundError:
            raise  # Re-raise specific error
        except ImportError as e:
            raise  # Re-raise specific error
        except ValueError as ve:  # Catch validation errors or other value errors
            self.logger.error(f"Import failed due to value error: {ve}", exc_info=True)
            self.rollback()
            return False
        except pd.errors.EmptyDataError:
            self.logger.info(
                f"Excel sheet '{sheet_name}' is empty (pandas detected). No data imported."
            )
            return True
        except sqlite3.Error as db_err:  # Catch DB errors during insert
            self.logger.error(
                f"Database error during bulk insert: {db_err}", exc_info=True
            )
            self.logger.error(f"Failed SQL (template): {insert_sql}")
            # Log first few data rows? Be careful with sensitive data.
            # self.logger.error(f"First few data rows (potential issue): {data_to_insert[:3]}")
            self.rollback()
            return False
        except Exception as e:
            self.logger.error(
                f"An unexpected error occurred during Excel import: {type(e).__name__} - {e}",
                exc_info=True,
            )
            self.rollback()
            return False

    def import_data_from_csv(
        self,
        table_name: str,
        csv_file_path: str,
        expected_columns: Optional[List[str]] = None,
        clear_table_first: bool = False,
        insert_strategy: SqlInsertStrategy = "INSERT OR REPLACE",
        validation_rules: Optional[Dict[str, Callable[[Any], bool]]] = None,
        on_validation_error: ValidationErrorStrategy = "log",
        csv_read_options: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Imports data from CSV, with optional pre-validation."""
        self.logger.info(
            f"Starting CSV import: '{csv_file_path}' -> Table:'{table_name}'"
        )
        self.logger.info(
            f"Import options - Strategy: {insert_strategy}, Clear: {clear_table_first}, Validation: {'Yes' if validation_rules else 'No'} (on error: {on_validation_error})"
        )

        if not self.cursor:
            raise ConnectionError("Database cursor is unavailable.")
        if not os.path.exists(csv_file_path):
            raise FileNotFoundError(f"CSV file '{csv_file_path}' not found.")
        if "pd" not in globals():
            raise ImportError("Pandas library is required for CSV import.")

        # Default CSV read options (can be overridden)
        default_csv_options = {
            "keep_default_na": True,
            "na_values": [""],
        }  # Treat empty strings as NaN/None
        read_options = default_csv_options.copy()
        if csv_read_options:
            read_options.update(csv_read_options)
        self.logger.debug(f"Using pandas read_csv options: {read_options}")

        try:
            df = pd.read_csv(csv_file_path, **read_options)
            self.logger.info(f"Read {len(df)} rows from CSV file.")
            if df.empty:
                self.logger.info("CSV file is empty. No data to import.")
                return True

            # Column Selection/Validation (same logic as Excel import)
            actual_csv_columns = df.columns.tolist()
            db_columns_info = self.get_table_columns(table_name)
            if not db_columns_info:
                self.logger.error(
                    f"Could not retrieve columns for target table '{table_name}'. Aborting import."
                )
                return False
            db_column_names = [col[0] for col in db_columns_info]

            if expected_columns:
                missing_cols = [
                    col for col in expected_columns if col not in actual_csv_columns
                ]
                if missing_cols:
                    self.logger.error(
                        f"CSV file missing expected columns: {missing_cols}. Aborting import."
                    )
                    return False
                cols_to_insert = expected_columns
            else:
                cols_to_insert = [
                    col for col in actual_csv_columns if col in db_column_names
                ]
                ignored_csv_cols = [
                    col for col in actual_csv_columns if col not in db_column_names
                ]
                if ignored_csv_cols:
                    self.logger.warning(
                        f"Ignoring CSV columns not found in table '{table_name}': {ignored_csv_cols}"
                    )
                if not cols_to_insert:
                    self.logger.error(
                        f"No matching columns found between CSV and table '{table_name}'. Aborting."
                    )
                    return False

            df_processed = df[cols_to_insert].copy()

            # --- Pre-import Validation ---
            df_validated = self._validate_dataframe(
                df_processed, validation_rules, on_validation_error
            )
            if df_validated.empty and not df_processed.empty:
                self.logger.warning(
                    "All rows failed validation or were skipped. No data will be imported."
                )
                return True
            if df_validated.empty and df_processed.empty:
                self.logger.info("Initial DataFrame was empty. No data to import.")
                return True

            # Basic Data Cleaning (NaN -> None) on validated data
            df_final = df_validated.where(pd.notnull(df_validated), None)
            data_to_insert = [tuple(row) for row in df_final.to_numpy()]

            if not data_to_insert:
                self.logger.info(
                    "No valid data remaining after validation/cleaning. Nothing to insert."
                )
                return True

            # Database Insertion (same logic as Excel import)
            if clear_table_first:
                if self.table_exists(table_name):
                    self.logger.info(
                        f"Clearing all data from table '{table_name}' before import..."
                    )
                    delete_sql = f'DELETE FROM "{table_name}";'
                    try:
                        self.execute_sql(delete_sql)
                        self.commit()
                        self.logger.info(f"Table '{table_name}' cleared successfully.")
                    except Exception as del_e:
                        self.logger.error(
                            f"Failed to clear table '{table_name}': {del_e}",
                            exc_info=True,
                        )
                        self.rollback()
                        return False
                else:
                    self.logger.warning(
                        f"Table '{table_name}' does not exist. Cannot clear."
                    )

            placeholders = ", ".join(["?"] * len(cols_to_insert))
            cols_string = ", ".join(f'"{col}"' for col in cols_to_insert)
            if insert_strategy not in [
                "INSERT",
                "INSERT OR REPLACE",
                "INSERT OR IGNORE",
            ]:
                self.logger.warning(
                    f"Invalid insert_strategy '{insert_strategy}'. Defaulting to 'INSERT OR REPLACE'."
                )
                insert_strategy = "INSERT OR REPLACE"
            insert_sql = f'{insert_strategy} INTO "{table_name}" ({cols_string}) VALUES ({placeholders});'

            self.logger.info(
                f"Executing bulk insert ({insert_strategy}) with {len(data_to_insert)} validated rows into '{table_name}'..."
            )
            self.cursor.executemany(insert_sql, data_to_insert)
            self.commit()
            self.logger.info(
                f"Bulk insert operation completed for table '{table_name}'."
            )
            return True

        except FileNotFoundError:
            raise
        except ImportError as e:
            raise
        except ValueError as ve:  # Catch validation or other value errors
            self.logger.error(f"Import failed due to value error: {ve}", exc_info=True)
            self.rollback()
            return False
        except pd.errors.EmptyDataError:
            self.logger.info(f"CSV file is empty (pandas detected). No data imported.")
            return True
        except sqlite3.Error as db_err:
            self.logger.error(
                f"Database error during bulk insert: {db_err}", exc_info=True
            )
            self.logger.error(f"Failed SQL (template): {insert_sql}")
            self.rollback()
            return False
        except Exception as e:
            self.logger.error(
                f"An unexpected error occurred during CSV import: {type(e).__name__} - {e}",
                exc_info=True,
            )
            self.rollback()
            return False

    def export_table_to_excel(
        self,
        table_name: Optional[str] = None,  # Make table_name optional if query provided
        excel_file_path: Optional[str] = None,
        sheet_name: str = "Sheet1",
        sql_query: Optional[str] = None,
        params: Optional[Tuple[Any, ...]] = None,
    ) -> bool:
        """Exports data from a database table or query to an Excel file."""
        effective_excel_path = excel_file_path or self.excel_path
        if not effective_excel_path:
            self.logger.error(
                "No Excel file path specified for export (neither explicit nor default)."
            )
            raise ValueError("No Excel file path specified for export.")

        export_source = (
            f"table '{table_name}'"
            if table_name and not sql_query
            else "custom SQL query"
        )
        self.logger.info(
            f"Starting data export from {export_source} to Excel '{effective_excel_path}' (Sheet: {sheet_name})..."
        )

        if not self.cursor:
            raise ConnectionError("Database cursor is unavailable.")
        if "pd" not in globals():
            raise ImportError("Pandas library is required for Excel export.")
        try:
            import openpyxl
        except ImportError:
            raise ImportError(
                "The 'openpyxl' library is required for writing .xlsx files."
            )

        try:
            # Determine query
            if sql_query:
                self.logger.info(f"Exporting based on provided SQL query.")
                query_to_run = sql_query
            elif table_name:
                if not self.table_exists(table_name):
                    self.logger.error(
                        f"Table '{table_name}' does not exist. Cannot export."
                    )
                    return False
                self.logger.info(f"Exporting all data from table '{table_name}'.")
                query_to_run = f'SELECT * FROM "{table_name}";'
                params = None  # Override params if exporting whole table
            else:
                self.logger.error(
                    "Must provide either table_name or sql_query for export."
                )
                raise ValueError("Must provide table_name or sql_query for export.")

            # Fetch data
            df_to_export = self.execute_select_query(
                query_to_run, params=params, return_dataframe=True
            )

            if df_to_export is None:
                self.logger.error("Failed to fetch data from database for export.")
                return False
            elif df_to_export.empty:
                self.logger.info("Query returned no data. Exporting empty Excel sheet.")
            else:
                self.logger.info(f"Fetched {len(df_to_export)} rows for export.")

            # Export DataFrame to Excel
            self.logger.info(
                f"Writing data to '{effective_excel_path}', sheet '{sheet_name}'..."
            )
            excel_dir = os.path.dirname(effective_excel_path)
            if excel_dir and not os.path.exists(excel_dir):
                os.makedirs(excel_dir, exist_ok=True)
                self.logger.info(f"Created directory '{excel_dir}' for Excel file.")

            df_to_export.to_excel(
                effective_excel_path,
                sheet_name=sheet_name,
                index=False,
                engine="openpyxl",
            )
            self.logger.info(f"Data exported successfully to '{effective_excel_path}'.")
            return True

        except ImportError as e:
            raise
        except ValueError as ve:
            raise  # Let ValueErrors from checks propagate
        except sqlite3.Error as db_e:
            self.logger.error(
                f"Database error during data fetch for export: {db_e}", exc_info=True
            )
            return False
        except Exception as e:
            self.logger.error(
                f"An error occurred during data export to Excel: {type(e).__name__} - {e}",
                exc_info=True,
            )
            return False

    def export_table_to_csv(
        self,
        table_name: Optional[str] = None,
        csv_file_path: Optional[str] = None,
        sql_query: Optional[str] = None,
        params: Optional[Tuple[Any, ...]] = None,
        csv_write_options: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Exports data from a database table or query to a CSV file."""
        if not csv_file_path:
            self.logger.error("No CSV file path specified for export.")
            raise ValueError("No CSV file path specified for export.")

        export_source = (
            f"table '{table_name}'"
            if table_name and not sql_query
            else "custom SQL query"
        )
        self.logger.info(
            f"Starting data export from {export_source} to CSV '{csv_file_path}'..."
        )

        if not self.cursor:
            raise ConnectionError("Database cursor is unavailable.")
        if "pd" not in globals():
            raise ImportError("Pandas library is required for CSV export.")

        # Default CSV write options
        default_csv_options = {"index": False, "quoting": csv.QUOTE_NONNUMERIC}
        write_options = default_csv_options.copy()
        if csv_write_options:
            write_options.update(csv_write_options)
        self.logger.debug(f"Using pandas to_csv options: {write_options}")

        try:
            # Determine query (same logic as Excel export)
            if sql_query:
                self.logger.info(f"Exporting based on provided SQL query.")
                query_to_run = sql_query
            elif table_name:
                if not self.table_exists(table_name):
                    self.logger.error(
                        f"Table '{table_name}' does not exist. Cannot export."
                    )
                    return False
                self.logger.info(f"Exporting all data from table '{table_name}'.")
                query_to_run = f'SELECT * FROM "{table_name}";'
                params = None
            else:
                self.logger.error(
                    "Must provide either table_name or sql_query for export."
                )
                raise ValueError("Must provide table_name or sql_query for export.")

            # Fetch data
            df_to_export = self.execute_select_query(
                query_to_run, params=params, return_dataframe=True
            )

            if df_to_export is None:
                self.logger.error("Failed to fetch data from database for export.")
                return False
            elif df_to_export.empty:
                self.logger.info("Query returned no data. Exporting empty CSV file.")
            else:
                self.logger.info(f"Fetched {len(df_to_export)} rows for export.")

            # Export DataFrame to CSV
            self.logger.info(f"Writing data to '{csv_file_path}'...")
            csv_dir = os.path.dirname(csv_file_path)
            if csv_dir and not os.path.exists(csv_dir):
                os.makedirs(csv_dir, exist_ok=True)
                self.logger.info(f"Created directory '{csv_dir}' for CSV file.")

            df_to_export.to_csv(csv_file_path, **write_options)
            self.logger.info(f"Data exported successfully to '{csv_file_path}'.")
            return True

        except ImportError as e:
            raise
        except ValueError as ve:
            raise
        except sqlite3.Error as db_e:
            self.logger.error(
                f"Database error during data fetch for export: {db_e}", exc_info=True
            )
            return False
        except Exception as e:
            self.logger.error(
                f"An error occurred during data export to CSV: {type(e).__name__} - {e}",
                exc_info=True,
            )
            return False

    # --- File Management & Backup ---

    def backup_database(self, backup_file_path: Optional[str] = None) -> bool:
        """Creates a backup of the current database using the Online Backup API."""
        if not self.conn:
            self.logger.error("Cannot backup database, no active connection.")
            return False

        if not backup_file_path:
            # Create a default backup name if none provided
            base, ext = os.path.splitext(self.db_path)
            backup_file_path = f"{base}_backup_{pd.Timestamp.now():%Y%m%d_%H%M%S}{ext}"
            self.logger.info(
                f"Backup file path not provided, using default: {backup_file_path}"
            )

        # Ensure backup directory exists
        backup_dir = os.path.dirname(backup_file_path)
        if backup_dir and not os.path.exists(backup_dir):
            try:
                os.makedirs(backup_dir, exist_ok=True)
                self.logger.info(f"Created directory '{backup_dir}' for backup file.")
            except OSError as dir_e:
                self.logger.error(
                    f"Failed to create directory for backup path '{backup_dir}': {dir_e}",
                    exc_info=True,
                )
                return False

        self.logger.info(
            f"Attempting to back up database '{self.db_path}' to '{backup_file_path}'..."
        )
        backup_conn = None
        try:
            backup_conn = sqlite3.connect(backup_file_path)
            with backup_conn:
                self.conn.backup(
                    backup_conn, pages=1, progress=self._backup_progress
                )  # Use online backup API
            self.logger.info("Database backup completed successfully.")
            return True
        except sqlite3.Error as e:
            self.logger.error(f"Database backup failed: {e}", exc_info=True)
            # Clean up potentially incomplete backup file
            if os.path.exists(backup_file_path):
                try:
                    self.logger.warning(
                        f"Removing potentially incomplete backup file: {backup_file_path}"
                    )
                    os.remove(backup_file_path)
                except OSError:
                    pass
            return False
        except Exception as e:
            self.logger.error(
                f"An unexpected error occurred during backup: {type(e).__name__} - {e}",
                exc_info=True,
            )
            return False
        finally:
            if backup_conn:
                backup_conn.close()

    def _backup_progress(self, status, remaining, total):
        """Simple progress logger for the backup operation."""
        if remaining % 100 == 0 or remaining == 0:  # Log every 100 pages or at the end
            self.logger.debug(
                f"Backup progress: {total - remaining}/{total} pages copied..."
            )

    def rename_database(self, new_db_path: str) -> bool:
        """Renames the database file. Closes connection first."""
        original_path = self.db_path
        self.logger.info(
            f"Attempting to rename database from '{original_path}' to '{new_db_path}'..."
        )

        if self.conn:
            self.logger.info("Closing active connection before renaming...")
            self.close()

        if not os.path.exists(original_path):
            self.logger.error(
                f"Original database file '{original_path}' does not exist. Cannot rename."
            )
            return False

        if os.path.abspath(original_path) == os.path.abspath(new_db_path):
            self.logger.warning(
                "New database path is the same as the old path. Rename skipped."
            )
            return True  # No action needed, considered successful

        if os.path.exists(new_db_path):
            try:
                self.logger.warning(
                    f"Target path '{new_db_path}' already exists. Attempting to remove it."
                )
                os.remove(new_db_path)
            except OSError as rm_e:
                self.logger.error(
                    f"Failed to remove existing file at new path '{new_db_path}': {rm_e}",
                    exc_info=True,
                )
                return False  # Cannot proceed if existing file cannot be removed

        new_dir = os.path.dirname(new_db_path)
        if new_dir and not os.path.exists(new_dir):
            try:
                os.makedirs(new_dir, exist_ok=True)
                self.logger.info(
                    f"Created directory '{new_dir}' for new database path."
                )
            except OSError as dir_e:
                self.logger.error(
                    f"Failed to create directory for new path '{new_dir}': {dir_e}",
                    exc_info=True,
                )
                return False

        try:
            # Use shutil.move for better cross-filesystem handling than os.rename
            self.logger.debug(
                f"Executing shutil.move('{original_path}', '{new_db_path}')..."
            )
            shutil.move(original_path, new_db_path)
            self.logger.info(f"Database successfully renamed/moved to '{new_db_path}'")
            self.db_path = new_db_path  # Update internal path ONLY on success
            return True
        except Exception as e:  # Catch shutil errors or others
            self.logger.error(
                f"Failed to rename/move database file: {type(e).__name__} - {e}",
                exc_info=True,
            )
            # If move fails, db_path should remain the original path
            return False


# End of DatabaseManager class definition
