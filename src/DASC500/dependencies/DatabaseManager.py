# DatabaseManager.py (Updated)
# filename: DatabaseManager.py
import sqlite3
import os
import sys
import shutil
import logging
import pandas as pd
import csv
import re # Keep re for potential future use, not strictly needed for space to underscore
from typing import (
    List,
    Tuple,
    Optional,
    Any,
    Dict,
    Union,
    Literal,
    Callable,
)
from enum import Enum

SqlInsertStrategy = Literal["INSERT", "INSERT OR REPLACE", "INSERT OR IGNORE"]
ValidationErrorStrategy = Literal["skip", "fail", "log"]

log_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s" # Added module and lineno
)

class FileType(Enum):
    HDF5 = "h5"
    PICKLE = "pkl"
    XML = "xml"
    JSON = "json"
    EXCEL = "xlsx"
    CSV = "csv"

class DatabaseManager:
    SUPPORTED_FILE_TYPES = FileType

    def __init__(
        self,
        db_path: str,
        excel_path: Optional[str] = None,
        log_file: Optional[str] = None, # This was in your original, ensure it's used or remove
        log_level: int = logging.INFO,
    ):
        self.db_path = db_path
        self.excel_path = excel_path
        self.conn = None
        self.cursor = None

        # Use the module's root logger for DatabaseManager itself
        # The script_logger in homework_5.py configures the root logger for file output
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(log_level)
        # Ensure handlers are not duplicated if root logger is already configured
        if not self.logger.hasHandlers() and not logging.getLogger().hasHandlers():
            console_handler = logging.StreamHandler(sys.stdout) # Log to console by default for the class
            console_handler.setFormatter(log_formatter)
            self.logger.addHandler(console_handler)
            # File logging can be added here too if a specific log_file is given for the manager
            # but homework_5.py already sets up a global file log.

        self.logger.info(
            f"DatabaseManager initialized for DB: '{self.db_path}'"
            f"{f' | Default Excel: {self.excel_path}' if self.excel_path else ''}"
        )
        # db_dir check removed as it was causing issues if path was relative and created later.
        # Consider adding it back in connect() if needed.


    def _standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Converts spaces to underscores and lowercases column names for standardization."""
        # Standardize: replace spaces with underscores, then convert to lowercase for good measure
        # Or just replace spaces:
        # new_columns = [col.replace(' ', '_') for col in df.columns]
        # More robust: replace various problematic characters if needed, then lowercase.
        # For now, just spaces to underscores as requested.
        standardized_columns = {col: col.replace(' ', '_') for col in df.columns}
        df.rename(columns=standardized_columns, inplace=True)
        self.logger.debug(f"Standardized DataFrame columns. New columns: {df.columns.tolist()}")
        return df

    def connect(self) -> None:
        if self.conn:
            self.logger.debug("Connection already established.")
            return

        # Ensure directory for DB exists before connecting
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            try:
                os.makedirs(db_dir, exist_ok=True)
                self.logger.info(f"Created directory for database: '{db_dir}'")
            except OSError as e:
                self.logger.error(f"Failed to create directory '{db_dir}' for database: {e}")
                raise ConnectionError(f"Directory creation failed for database: {e}") from e

        try:
            self.logger.info(f"Connecting to database: '{self.db_path}'...")
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
            self.cursor = self.conn.cursor()
            self.logger.info("Database connection established successfully.")
        except sqlite3.Error as e:
            self.logger.error(
                f"Failed to connect to database '{self.db_path}'. Error: {e}",
                exc_info=True,
            )
            self.conn = None
            self.cursor = None
            raise ConnectionError(f"Database connection failed: {e}") from e

    def close(self) -> None:
        if self.conn:
            try:
                self.logger.info(
                    "Committing final changes (if any) and closing connection..."
                )
                self.conn.close()
                self.logger.info(f"Database connection to '{self.db_path}' closed.")
            except sqlite3.Error as e:
                self.logger.warning(
                    f"Error during closing sequence (close): {e}", exc_info=True
                )
            finally:
                self.conn = None
                self.cursor = None
        else:
            self.logger.debug("No active connection to close.")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        is_exception = exc_type is not None
        if self.conn:
            if is_exception:
                self.logger.error(
                    f"Exception occurred within context: {exc_type.__name__}({exc_val}). Rolling back.",
                    exc_info=(exc_type, exc_val, exc_tb),
                )
                try:
                    self.conn.rollback()
                except sqlite3.Error as rb_err:
                    self.logger.error(f"Failed to rollback on context exit: {rb_err}", exc_info=True)
            else:
                self.logger.debug("Exiting context manager normally. Committing.")
                try:
                    self.conn.commit()
                except sqlite3.Error as commit_err:
                    self.logger.error(f"Failed to commit on context exit: {commit_err}", exc_info=True)
        self.close()
        return False

    def execute_sql(
        self, sql_command: str, params: Optional[Tuple[Any, ...]] = None
    ) -> None:
        if not self.cursor:
            self.logger.error("execute_sql called but database cursor is unavailable.")
            raise ConnectionError("Database cursor is unavailable. Cannot execute SQL.")
        try:
            self.logger.debug(
                f"Executing SQL: {sql_command[:150]}{'...' if len(sql_command)>150 else ''} | Params: {'(hidden)' if isinstance(params, tuple) and any(isinstance(p, (bytes, bytearray)) for p in params) else params if params else 'None'}"
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
                self.logger.error(f"Failed Params (types): {[type(p) for p in params]}")
            raise

    def execute_script(self, sql_script: str) -> None:
        # ... (no changes needed here)
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
        # ... (no changes needed here)
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
        # ... (no changes needed here)
        if not self.conn:
            self.logger.warning("Rollback called but database connection might be unavailable.")
        try:
            self.logger.info("Rolling back transaction...")
            if self.conn:
                self.conn.rollback()
                self.logger.info("Transaction rolled back successfully.")
            else:
                self.logger.warning("Rollback attempted but no active connection object.")
        except sqlite3.Error as e:
            self.logger.error(
                f"Failed to rollback transaction. Error: {e}", exc_info=True
            )


    def execute_select_query(
        self,
        sql_query: str,
        params: Optional[Tuple[Any, ...]] = None,
        return_dataframe: bool = True,
    ) -> Optional[Union[pd.DataFrame, List[sqlite3.Row]]]:
        # ... (no changes needed here other than ensuring Pandas import)
        if "pd" not in globals():
            self.logger.error("Pandas (pd) not imported. Required for DataFrame return.")
            raise ImportError("Pandas library is required for returning DataFrames.")
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
                if results:
                    column_names = [
                        description[0] for description in self.cursor.description
                    ]
                    # Standardize column names after fetching if they might have spaces from DB
                    # However, if we standardize *before* table creation, this is less critical here.
                    # For safety, if table might have been created externally with spaces:
                    # column_names = [col.replace(' ', '_') for col in column_names]

                    df = pd.DataFrame(results, columns=column_names)
                    self.logger.debug(f"Returning {len(df)} results as DataFrame.")
                    return df
                else:
                    self.logger.debug("Returning empty DataFrame (no results).")
                    return pd.DataFrame() # Ensure columns=[] if possible or let pandas handle
            else:
                self.logger.debug(
                    f"Returning {len(results)} results as list of Row objects."
                )
                return results

        except sqlite3.Error as e:
            self.logger.error(
                f"Failed to execute SELECT query. Error: {e}", exc_info=True
            )
            self.logger.error(f"Failed Query: {sql_query}")
            if params: self.logger.error(f"Failed Params: {params}")
            return None


    def table_exists(self, table_name: str) -> bool:
        # ... (no changes needed here)
        self.logger.debug(f"Checking existence of table: {table_name}")
        query = "SELECT name FROM sqlite_master WHERE type='table' AND name=?;"
        try:
            if not self.cursor: raise ConnectionError("DB cursor unavailable.")
            self.cursor.execute(query, (table_name,))
            result = self.cursor.fetchone()
            exists = result is not None
            self.logger.debug(f"Table '{table_name}' exists: {exists}")
            return exists
        except sqlite3.Error as e:
            self.logger.error(f"Error checking table '{table_name}': {e}", exc_info=True)
            return False
        except ConnectionError as ce:
            self.logger.error(f"Cannot check table '{table_name}', cursor unavailable: {ce}")
            raise

    def get_table_columns(self, table_name: str) -> Optional[List[Tuple[str, str]]]:
        # ... (no changes needed here)
        self.logger.debug(f"Fetching columns for table: {table_name}...")
        if not self.table_exists(table_name):
            self.logger.warning(f"Table '{table_name}' does not exist.")
            return None

        query = f'PRAGMA table_info("{table_name}");' # Table name might need quoting if it contains special chars
        if not self.cursor: raise ConnectionError("DB cursor unavailable.")
        try:
            self.cursor.execute(query)
            columns_info = self.cursor.fetchall()
            if not columns_info:
                self.logger.warning(f"No columns found for table '{table_name}'. It might be empty or PRAGMA failed.")
                return [] # Return empty list instead of None if table exists but no columns (unlikely)
            columns = [(col["name"], col["type"]) for col in columns_info]
            self.logger.debug(f"Columns for '{table_name}': {columns}")
            return columns
        except sqlite3.Error as e:
            self.logger.error(f"Failed fetching columns for '{table_name}': {e}", exc_info=True)
            return None


    def _validate_dataframe(
        self,
        df: pd.DataFrame,
        validation_rules: Optional[Dict[str, Callable[[Any], bool]]],
        on_error: ValidationErrorStrategy = "log",
    ) -> pd.DataFrame:
        # ... (no changes needed here, operates on DataFrame as is)
        if not validation_rules or df.empty:
            return df

        valid_rows_mask = pd.Series(True, index=df.index)
        self.logger.info(f"Applying {len(validation_rules)} validation rule(s) to {len(df)} rows...")
        validation_errors_count = 0

        for col_name, validator_func in validation_rules.items():
            if col_name not in df.columns:
                self.logger.warning(f"Validation rule for non-existent column '{col_name}'. Skipping.")
                continue
            self.logger.debug(f"Validating column '{col_name}'...")
            try:
                # Apply validator only to non-NA values, consider NA as passing or failing based on rule design
                col_valid_mask = df[col_name].apply(lambda x: validator_func(x) if pd.notna(x) else True)
                failed_indices = df.index[~col_valid_mask & valid_rows_mask] # Consider only rows still marked valid
                if not failed_indices.empty:
                    num_failed_for_col = len(failed_indices)
                    validation_errors_count += num_failed_for_col
                    self.logger.warning(
                        f"Validation failed for column '{col_name}' on {num_failed_for_col} row(s). "
                        f"Example indices: {failed_indices.tolist()[:5]}" # Log a few examples
                    )
                    if on_error == "fail":
                        raise ValueError(f"Validation failed for column '{col_name}'. Aborting operation.")
                    elif on_error == "skip" or on_error == "log": # if 'log', we log but keep rows unless skip
                        valid_rows_mask &= col_valid_mask # Update overall mask
            except Exception as e:
                self.logger.error(f"Error during validation of column '{col_name}': {e}", exc_info=True)
                if on_error == "fail":
                    raise ValueError(f"Error validating column '{col_name}': {e}") from e
                # If 'skip' or 'log', this column might be problematic, mark all its rows as invalid for safety if desired
                # valid_rows_mask = pd.Series(False, index=df.index) # Or just skip this column's rule

        if validation_errors_count > 0:
            self.logger.warning(f"Total validation failures across all rules: {validation_errors_count}")

        if on_error == "skip":
            df_validated = df[valid_rows_mask].copy()
            self.logger.info(f"Skipped {len(df) - len(df_validated)} invalid rows due to validation failures.")
            return df_validated
        # If 'log' or 'fail' (and not aborted), return original or partially masked df based on intent
        # For now, if 'log', we've logged, return the df that may still contain failed rows (caller beware)
        # If 'fail', it would have raised an exception.
        return df # If on_error is 'log', return original df; it's up to caller


    def _load_dataframe_to_db(
        self,
        df: pd.DataFrame,
        table_name: str,
        # expected_columns: Optional[List[str]] = None, # Less relevant if we standardize and match
        clear_table_first: bool = False,
        insert_strategy: SqlInsertStrategy = "INSERT OR REPLACE",
        validation_rules: Optional[Dict[str, Callable[[Any], bool]]] = None,
        on_validation_error: ValidationErrorStrategy = "log",
    ) -> bool:
        self.logger.info(f"Preparing to load DataFrame into table '{table_name}'.")
        if not self.conn or not self.cursor: # Ensure connection
            self.logger.error("Database not connected. Cannot load DataFrame.")
            raise ConnectionError("Database not connected.")
        if df.empty:
            self.logger.info("DataFrame is empty. No data to load.")
            return True

        # --- Column Standardization happens *before* this method via load_data ---
        # So, df.columns are already standardized (e.g., spaces to underscores)

        df_validated = self._validate_dataframe(df, validation_rules, on_validation_error)

        if df_validated.empty:
            self.logger.info("DataFrame is empty after validation. No data to import.")
            return True

        # SQLite typically handles NaT/NaN as NULL for TEXT/REAL if notna() used correctly
        df_final = df_validated.where(pd.notnull(df_validated), None)
        data_to_insert = [tuple(row) for row in df_final.to_numpy()]

        if not data_to_insert:
            self.logger.info("No valid data remaining after processing. Nothing to insert.")
            return True

        # Table creation and column handling logic:
        # Columns in df_final are now the standardized names.
        # The table will be created with these standardized names if it doesn't exist.
        actual_df_columns = df_final.columns.tolist()

        if clear_table_first and self.table_exists(table_name):
            self.logger.info(f"Clearing data from table '{table_name}'...")
            try:
                self.execute_sql(f'DELETE FROM "{table_name}";')
                self.logger.info(f"Table '{table_name}' cleared (pending commit or rollback by context manager).")
            except Exception as del_e:
                self.logger.error(f"Failed to clear table '{table_name}': {del_e}", exc_info=True)
                return False

        if not self.table_exists(table_name):
            self.logger.info(f"Table '{table_name}' does not exist. Attempting to create with standardized column names.")
            try:
                col_types_list = []
                for col_name in actual_df_columns:
                    # Infer SQL type from DataFrame dtype
                    dtype = df_final[col_name].dtype
                    if pd.api.types.is_integer_dtype(dtype): sql_type = "INTEGER"
                    elif pd.api.types.is_float_dtype(dtype): sql_type = "REAL"
                    elif pd.api.types.is_bool_dtype(dtype): sql_type = "INTEGER" # Store bools as 0/1
                    elif pd.api.types.is_datetime64_any_dtype(dtype): sql_type = "TEXT" # Store datetimes as ISO strings
                    elif pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype): sql_type = "TEXT"
                    else: sql_type = "TEXT" # Default for other types
                    col_types_list.append(f'"{col_name}" {sql_type}') # Quote standardized names
                create_sql = f'CREATE TABLE "{table_name}" ({", ".join(col_types_list)});'
                self.execute_sql(create_sql)
                self.logger.info(f"Table '{table_name}' created with columns: {actual_df_columns}")
            except Exception as ct_e:
                self.logger.error(f"Failed to auto-create table '{table_name}': {ct_e}", exc_info=True)
                return False
        else: # Table exists, ensure DataFrame columns match subset of table columns
            db_columns_info = self.get_table_columns(table_name)
            if db_columns_info is None: # Should not happen if table_exists is true
                self.logger.error(f"Could not get columns for existing table '{table_name}'.")
                return False
            db_column_names = [col[0] for col in db_columns_info] # These should also be standardized if table was created by this manager

            # Filter df_final to only include columns that exist in the DB table
            cols_to_insert_final = [col for col in actual_df_columns if col in db_column_names]
            ignored_df_cols = [col for col in actual_df_columns if col not in db_column_names]
            if ignored_df_cols:
                self.logger.warning(f"Ignoring DataFrame columns not in existing table '{table_name}': {ignored_df_cols}")

            if not cols_to_insert_final:
                self.logger.error(f"No matching columns between DataFrame and existing table '{table_name}'. Aborting.")
                return False

            # Re-filter df_final and data_to_insert for only these columns
            df_final = df_final[cols_to_insert_final]
            data_to_insert = [tuple(row) for row in df_final.to_numpy()]
            actual_df_columns = cols_to_insert_final # Update for INSERT statement

        if not data_to_insert: # Check again if filtering removed all data
            self.logger.info("No data to insert after matching with existing table columns.")
            return True

        placeholders = ", ".join(["?"] * len(actual_df_columns))
        cols_str = ", ".join(f'"{col}"' for col in actual_df_columns) # Quote standardized names
        if insert_strategy not in ["INSERT", "INSERT OR REPLACE", "INSERT OR IGNORE"]:
            self.logger.warning(f"Invalid insert_strategy '{insert_strategy}'. Defaulting to 'INSERT OR REPLACE'.")
            insert_strategy = "INSERT OR REPLACE"
        insert_sql = f'{insert_strategy} INTO "{table_name}" ({cols_str}) VALUES ({placeholders});'

        try:
            self.logger.info(f"Executing bulk insert ({insert_strategy}) with {len(data_to_insert)} rows into '{table_name}'...")
            self.cursor.executemany(insert_sql, data_to_insert)
            self.logger.info(f"Bulk insert for '{table_name}' prepared (pending commit/rollback by context manager).")
            return True
        except sqlite3.Error as db_err:
            self.logger.error(f"Database error during insert into '{table_name}': {db_err}", exc_info=True)
            self.logger.error(f"Failed SQL: {insert_sql}")
            self.logger.error(f"First few data rows (example): {[str(row)[:100] for row in data_to_insert[:2]]}")
            return False


    def _load_hdf5(self, file_path: str, table_name: str, hdf_read_options: Optional[Dict[str, Any]] = None) -> Optional[pd.DataFrame]:
        opts = hdf_read_options or {}
        key_to_use = opts.pop('key', None) # Remove key from opts if present, pandas uses it directly
        self.logger.info(f"Reading HDF5: '{file_path}' (Key hint: {key_to_use or 'Default/First'})")
        try:
            if not key_to_use:
                with pd.HDFStore(file_path, mode='r') as store:
                    keys = store.keys()
                    if not keys:
                        self.logger.error(f"No keys found in HDF5 file '{file_path}'. Cannot read.")
                        return None
                    key_to_use = keys[0] # Default to the first key
                    self.logger.info(f"No HDF5 key explicitly provided. Using first available key: '{key_to_use}'")
            df = pd.read_hdf(file_path, key=key_to_use, **opts)
            self.logger.info(f"Read {len(df)} rows from HDF5 '{file_path}', key '{key_to_use}'.")
            return self._standardize_column_names(df) # Standardize here
        # ... (rest of _load_hdf5, error handling)
        except FileNotFoundError:
            self.logger.error(f"HDF5 file not found: '{file_path}'")
            raise
        except KeyError as ke:
            self.logger.error(f"HDF5 key '{key_to_use}' not found in '{file_path}'. Error: {ke}")
            try:
                with pd.HDFStore(file_path, mode='r') as store:
                    self.logger.info(f"Available keys in '{file_path}': {store.keys()}")
            except Exception as store_err:
                self.logger.error(f"Could not list keys from HDF5 file '{file_path}': {store_err}")
            return None
        except Exception as e:
            self.logger.error(f"Error reading HDF5 file '{file_path}': {e}", exc_info=True)
            return None


    def _load_pickle(self, file_path: str, table_name: str, pickle_read_options: Optional[Dict[str, Any]] = None) -> Optional[pd.DataFrame]:
        opts = pickle_read_options or {}
        self.logger.info(f"Reading Pickle: '{file_path}'")
        try:
            df = pd.read_pickle(file_path, **opts)
            if not isinstance(df, pd.DataFrame):
                self.logger.error(f"Pickle file '{file_path}' did not contain a Pandas DataFrame. Found type: {type(df)}")
                return None
            self.logger.info(f"Read {len(df)} rows from Pickle '{file_path}'.")
            return self._standardize_column_names(df) # Standardize here
        # ... (rest of _load_pickle, error handling)
        except FileNotFoundError:
            self.logger.error(f"Pickle file not found: '{file_path}'")
            raise
        except Exception as e:
            self.logger.error(f"Error reading Pickle file '{file_path}': {e}", exc_info=True)
            return None

    def _load_xml(self, file_path: str, table_name: str, xml_read_options: Optional[Dict[str, Any]] = None) -> Optional[pd.DataFrame]:
        opts = xml_read_options or {}
        self.logger.info(f"Reading XML: '{file_path}' with options {opts}")
        if not opts.get('xpath') and not opts.get('parser') == 'lxml':
            self.logger.warning("Pandas 'read_xml' often benefits from an 'xpath' argument, especially for complex XML. "
                                "Consider providing it in 'load_options': {'xml_read_options': {'xpath': './/your_row_element'}}")
        try:
            df = pd.read_xml(file_path, **opts)
            self.logger.info(f"Read {len(df)} rows from XML '{file_path}'.")
            return self._standardize_column_names(df) # Standardize here
        # ... (rest of _load_xml, error handling)
        except FileNotFoundError:
            self.logger.error(f"XML file not found: '{file_path}'")
            raise
        except ValueError as ve: # Often indicates parsing issues, e.g. malformed XML or incorrect xpath
            self.logger.error(f"ValueError reading XML file '{file_path}'. This might be due to incorrect XML structure, "
                              f"parsing options (like xpath), or an empty file if not handled. Error: {ve}", exc_info=True)
            return None
        except Exception as e:
            self.logger.error(f"Error reading XML file '{file_path}': {e}", exc_info=True)
            return None


    def _load_json(self, file_path: str, table_name: str, json_read_options: Optional[Dict[str, Any]] = None) -> Optional[pd.DataFrame]:
        opts = json_read_options or {}
        self.logger.info(f"Reading JSON: '{file_path}' with options {opts}")
        # Common defaults if not specified by user, pandas has its own defaults too.
        # if 'orient' not in opts: opts['orient'] = 'records'
        # if 'lines' not in opts: opts['lines'] = False
        try:
            df = pd.read_json(file_path, **opts)
            self.logger.info(f"Read {len(df)} rows from JSON '{file_path}'.")
            return self._standardize_column_names(df) # Standardize here
        # ... (rest of _load_json, error handling)
        except FileNotFoundError:
            self.logger.error(f"JSON file not found: '{file_path}'")
            raise
        except ValueError as ve: # Often indicates issues with JSON structure vs 'orient' or other options
            self.logger.error(f"ValueError reading JSON file '{file_path}'. Check JSON format and pandas read_json options "
                              f"(e.g., 'orient', 'lines'). Error: {ve}", exc_info=True)
            return None
        except Exception as e:
            self.logger.error(f"Error reading JSON file '{file_path}': {e}", exc_info=True)
            return None

    def load_data(
        self,
        file_path: str,
        table_name: str,
        file_type: FileType,
        # expected_columns: Optional[List[str]] = None, # Less relevant now with standardization
        clear_table_first: bool = False,
        insert_strategy: SqlInsertStrategy = "INSERT OR REPLACE",
        validation_rules: Optional[Dict[str, Callable[[Any], bool]]] = None,
        on_validation_error: ValidationErrorStrategy = "log",
        load_options: Optional[Dict[str, Any]] = None,
    ) -> bool:
        self.logger.info(
            f"Unified load: '{file_path}' (Type: {file_type.name}) -> Table:'{table_name}'"
        )
        if not self.conn or not self.cursor: # Ensure connected
            self.logger.error("Database not connected for load_data operation.")
            # Attempt to connect if not already? Or rely on context manager.
            # For now, assume it should be connected by `with db_manager:`.
            raise ConnectionError("Database not connected.")

        if not os.path.exists(file_path):
            self.logger.error(f"Data file not found: '{file_path}'")
            return False # Indicate failure clearly
        if "pd" not in globals():
            self.logger.critical("Pandas (pd) library not imported. Cannot process data files.")
            raise ImportError("Pandas library is required for data loading operations.")

        opts = load_options or {}
        df: Optional[pd.DataFrame] = None

        try:
            if file_type == self.SUPPORTED_FILE_TYPES.HDF5:
                df = self._load_hdf5(file_path, table_name, opts.get('hdf_read_options'))
            elif file_type == self.SUPPORTED_FILE_TYPES.PICKLE:
                df = self._load_pickle(file_path, table_name, opts.get('pickle_read_options'))
            elif file_type == self.SUPPORTED_FILE_TYPES.XML:
                df = self._load_xml(file_path, table_name, opts.get('xml_read_options'))
            elif file_type == self.SUPPORTED_FILE_TYPES.JSON:
                df = self._load_json(file_path, table_name, opts.get('json_read_options'))
            elif file_type == self.SUPPORTED_FILE_TYPES.EXCEL:
                excel_opts = opts.get('excel_read_options', {}) # Pass sub-options
                sheet = excel_opts.pop('sheet_name', 0) # Pop sheet_name, pass rest
                try:
                    import openpyxl # Local import check
                except ImportError:
                    self.logger.error("'openpyxl' library is required to read .xlsx files.")
                    raise ImportError("'openpyxl' required for .xlsx files.")
                df_excel = pd.read_excel(file_path, sheet_name=sheet, engine="openpyxl", **excel_opts)
                self.logger.info(f"Read {len(df_excel)} rows from Excel '{file_path}', sheet '{sheet}'.")
                df = self._standardize_column_names(df_excel) # Standardize
            elif file_type == self.SUPPORTED_FILE_TYPES.CSV:
                csv_opts = opts.get('csv_read_options', {}) # Pass sub-options
                default_csv_opts = {"keep_default_na": True, "na_values": ["", "NA", "N/A", "NaN", "null"]} # More robust NA handling
                final_csv_opts = {**default_csv_opts, **csv_opts}
                df_csv = pd.read_csv(file_path, **final_csv_opts)
                self.logger.info(f"Read {len(df_csv)} rows from CSV '{file_path}'.")
                df = self._standardize_column_names(df_csv) # Standardize
            else:
                self.logger.error(f"Unsupported file type provided: {file_type}")
                return False

            if df is None: # Reading failed in helper, or file type not supported properly
                self.logger.error(f"Failed to read DataFrame from '{file_path}' as {file_type.name}.")
                return False
            if df.empty:
                self.logger.info(f"File '{file_path}' resulted in an empty DataFrame. No data to load into table.")
                return True # Not an error, just no data

            # _load_dataframe_to_db handles commit/rollback via context manager on __exit__
            load_status = self._load_dataframe_to_db(
                df, table_name, clear_table_first,
                insert_strategy, validation_rules, on_validation_error
            )
            # No explicit commit/rollback here; context manager handles it.
            if load_status:
                self.logger.info(f"DataFrame from '{file_path}' successfully processed for loading into '{table_name}'.")
            else:
                self.logger.warning(f"Loading DataFrame from '{file_path}' into '{table_name}' reported issues.")
            return load_status

        except FileNotFoundError: # Should be caught by os.path.exists, but good failsafe
            self.logger.error(f"File not found error during load_data: '{file_path}'", exc_info=True)
            return False
        except ImportError as ie: # e.g. missing openpyxl
            self.logger.error(f"Import error during data loading: {ie}", exc_info=True)
            return False
        except pd.errors.EmptyDataError: # For files that are truly empty
            self.logger.info(f"File '{file_path}' is empty (pandas EmptyDataError). No data loaded.")
            return True
        except Exception as e: # Catch-all for other unexpected errors during load
            self.logger.error(f"Unexpected error loading data from '{file_path}': {e}", exc_info=True)
            return False # Indicate failure
    # ... (rest of the DatabaseManager class: import_data_from_excel, csv, export methods, backup, rename)
    # These export/import specific methods would also benefit from column standardization
    # if they interact with DataFrames that have spaces in names.
    # For now, focusing on load_data.