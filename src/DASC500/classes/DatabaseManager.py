import sqlite3
import os
import sys
import shutil
import logging # Added logging
import pandas as pd # Using pandas for Excel import/export
from typing import List, Tuple, Optional, Any, Dict, Union, Literal # For type hinting

# Define valid SQL insert strategies for type hinting and validation
SqlInsertStrategy = Literal["INSERT", "INSERT OR REPLACE", "INSERT OR IGNORE"]

# --- Basic Logger Setup ---
# Configure a default logger format
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class DatabaseManager:
    """
    Manages generic SQLite database operations including connection handling,
    SQL execution, schema introspection, data import/export via Excel,
    and file renaming. Uses Python's logging module.

    Designed for use as a context manager (`with` statement).

    Attributes:
        db_path (str): Path to the SQLite database file.
        excel_path (Optional[str]): Default path to an Excel file for import/export.
        conn (Optional[sqlite3.Connection]): Active SQLite connection.
        cursor (Optional[sqlite3.Cursor]): Cursor for executing SQL.
        logger (logging.Logger): Logger instance for this manager.
    """

    def __init__(self,
                 db_path: str,
                 excel_path: Optional[str] = None,
                 log_file: Optional[str] = None,
                 log_level: int = logging.INFO):
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
        self.excel_path = excel_path
        self.conn = None
        self.cursor = None

        # --- Logger Configuration ---
        # Use class name or a specific name for the logger instance
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(log_level)

        # Prevent adding multiple handlers if init is called again unexpectedly
        if not self.logger.hasHandlers():
            # Console Handler (always add)
            console_handler = logging.StreamHandler(sys.stdout) # Use stdout for info/debug
            console_handler.setFormatter(log_formatter)
            # Optionally filter console handler level differently (e.g., only INFO and above)
            # console_handler.setLevel(logging.INFO)
            self.logger.addHandler(console_handler)

            # File Handler (add if log_file is specified)
            if log_file:
                try:
                    # Ensure log directory exists
                    log_dir = os.path.dirname(log_file)
                    if log_dir and not os.path.exists(log_dir):
                         os.makedirs(log_dir, exist_ok=True)

                    file_handler = logging.FileHandler(log_file, mode='a') # Append mode
                    file_handler.setFormatter(log_formatter)
                    self.logger.addHandler(file_handler)
                    self.logger.info(f"Logging configured to file: {log_file}")
                except Exception as e:
                    self.logger.error(f"Failed to configure file logging to '{log_file}': {e}", exc_info=False) # Avoid traceback in log for this specific error
                    # Continue without file logging

        self.logger.info(f"DatabaseManager initialized for DB: '{self.db_path}'"
                         f"{f' | Default Excel: {self.excel_path}' if self.excel_path else ''}")

        # --- Initial Checks ---
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
             self.logger.error(f"Directory for database '{db_dir}' does not exist.")
             raise FileNotFoundError(f"Directory for database '{db_dir}' does not exist.")


    def connect(self) -> None:
        """
        Establishes the database connection and initializes the cursor.
        Sets the connection's row_factory to sqlite3.Row for dict-like access.

        Raises:
            ConnectionError: If the connection fails.
        """
        if self.conn: # Avoid reconnecting if already connected
            self.logger.debug("Connection already established.")
            return
        try:
            self.logger.info(f"Connecting to database: '{self.db_path}'...")
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
            self.cursor = self.conn.cursor()
            self.logger.info("Database connection established successfully.")
            # Optionally enable foreign key support if needed
            # try:
            #     self.execute_sql("PRAGMA foreign_keys = ON;")
            #     self.logger.debug("PRAGMA foreign_keys = ON executed.")
            # except Exception as fk_e:
            #      self.logger.warning(f"Could not enable foreign key support: {fk_e}")
        except sqlite3.Error as e:
            self.logger.error(f"Failed to connect to database '{self.db_path}'. Error: {e}", exc_info=True)
            self.conn = None
            self.cursor = None
            raise ConnectionError(f"Database connection failed: {e}") from e

    def close(self) -> None:
        """
        Commits any pending changes and closes the database connection if open.
        """
        if self.conn:
            try:
                self.logger.info("Committing final changes (if any) and closing connection...")
                self.conn.commit()
                self.conn.close()
                self.logger.info(f"Database connection to '{self.db_path}' closed.")
            except sqlite3.Error as e:
                self.logger.warning(f"Error during closing sequence (commit/close): {e}", exc_info=True)
            finally:
                self.conn = None
                self.cursor = None
        else:
             self.logger.debug("No active connection to close.")

    def __enter__(self):
        """
        Enters the runtime context (for `with` statement), establishing the connection.
        """
        self.connect() # Ensure connection is active
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exits the runtime context (for `with` statement), ensuring connection closure.
        Logs exception info if one occurred within the context.
        """
        is_exception = exc_type is not None
        if is_exception:
             # Log the exception details before closing
             self.logger.error(f"Exception occurred within context: {exc_type.__name__}({exc_val})", exc_info=(exc_type, exc_val, exc_tb))
        else:
             self.logger.debug("Exiting context manager normally.")

        self.close() # Ensure closure regardless of exceptions
        # Return False to propagate exceptions, True to suppress
        return False

    def execute_sql(self, sql_command: str, params: Optional[Tuple[Any, ...]] = None) -> None:
        """
        Executes a single SQL command (e.g., CREATE, INSERT, UPDATE, DELETE, PRAGMA).
        Does NOT automatically commit changes for DML. Call commit() separately.

        Args:
            sql_command (str): The SQL statement to execute.
            params (Optional[Tuple[Any, ...]]): Optional parameters for the query.

        Raises:
            ConnectionError: If the database connection/cursor is not active.
            sqlite3.Error: If the SQL execution fails. Propagates the error.
        """
        if not self.cursor:
            self.logger.error("execute_sql called but database cursor is unavailable.")
            raise ConnectionError("Database cursor is unavailable. Cannot execute SQL.")
        try:
            self.logger.debug(f"Executing SQL: {sql_command[:150]}{'...' if len(sql_command)>150 else ''} | Params: {params if params else 'None'}")
            if params:
                self.cursor.execute(sql_command, params)
            else:
                self.cursor.execute(sql_command)
            self.logger.debug(f"SQL executed successfully.")
        except sqlite3.Error as e:
            self.logger.error(f"Failed to execute SQL command. Error: {e}", exc_info=True)
            self.logger.error(f"Failed SQL: {sql_command}")
            if params:
                self.logger.error(f"Failed Params: {params}")
            raise # Re-raise the error after logging

    def execute_script(self, sql_script: str) -> None:
        """
        Executes multiple SQL statements contained in a single string.
        Statements should be separated by semicolons (;).
        Does NOT automatically commit changes. Call commit() separately.

        Args:
            sql_script (str): A string containing one or more SQL statements.

        Raises:
            ConnectionError: If the database connection/cursor is not active.
            sqlite3.Error: If any SQL execution within the script fails.
        """
        if not self.cursor:
            self.logger.error("execute_script called but database cursor is unavailable.")
            raise ConnectionError("Database cursor is unavailable. Cannot execute script.")
        try:
            self.logger.info(f"Executing SQL script (length: {len(sql_script)})...")
            self.cursor.executescript(sql_script)
            self.logger.info("SQL script executed successfully.")
        except sqlite3.Error as e:
            self.logger.error(f"Failed to execute SQL script. Error: {e}", exc_info=True)
            # Note: executescript implicitly commits before raising an error on failure.
            raise

    def commit(self) -> None:
        """
        Commits the current database transaction.

        Raises:
            ConnectionError: If the database connection is unavailable.
            sqlite3.Error: If the commit fails.
        """
        if not self.conn:
            self.logger.error("commit called but database connection is unavailable.")
            raise ConnectionError("Database connection is unavailable. Cannot commit.")
        try:
            self.logger.debug("Committing transaction...")
            self.conn.commit()
            self.logger.info("Transaction committed successfully.")
        except sqlite3.Error as e:
            self.logger.error(f"Failed to commit transaction. Error: {e}", exc_info=True)
            raise

    def rollback(self) -> None:
        """
        Rolls back the current database transaction to the last commit point.

        Raises:
            ConnectionError: If the database connection is unavailable.
            sqlite3.Error: If the rollback fails.
        """
        if not self.conn:
            self.logger.error("rollback called but database connection is unavailable.")
            raise ConnectionError("Database connection is unavailable. Cannot rollback.")
        try:
            self.logger.info("Rolling back transaction...")
            self.conn.rollback()
            self.logger.info("Transaction rolled back successfully.")
        except sqlite3.Error as e:
            self.logger.error(f"Failed to rollback transaction. Error: {e}", exc_info=True)
            # Decide if rollback failure should raise an exception
            # raise

    def execute_select_query(self, sql_query: str, params: Optional[Tuple[Any, ...]] = None, return_dataframe: bool = True) -> Optional[Union[pd.DataFrame, List[sqlite3.Row]]]:
        """
        Executes a SELECT query and fetches all results.

        Args:
            sql_query (str): The SELECT SQL query string.
            params (Optional[Tuple[Any, ...]]): Optional parameters for the query.
            return_dataframe (bool): If True (default), returns results as a pandas DataFrame.
                                     If False, returns a list of sqlite3.Row objects.

        Returns:
            Optional[Union[pd.DataFrame, List[sqlite3.Row]]]: Query results,
                an empty DataFrame/list if no rows found, or None on execution error.

        Raises:
            ConnectionError: If the database connection/cursor is not active.
            ImportError: If pandas is needed but not installed.
        """
        if not self.cursor:
            self.logger.error("execute_select_query called but database cursor is unavailable.")
            raise ConnectionError("Database cursor is unavailable. Cannot execute query.")
        try:
            self.logger.debug(f"Executing SELECT: {sql_query[:150]}{'...' if len(sql_query)>150 else ''} | Params: {params if params else 'None'}")
            if params:
                self.cursor.execute(sql_query, params)
            else:
                self.cursor.execute(sql_query)
            results = self.cursor.fetchall()
            self.logger.info(f"SELECT query executed successfully. Found {len(results)} records.")

            if return_dataframe:
                if 'pd' not in globals():
                     self.logger.error("Pandas library is required to return DataFrames but not found.")
                     raise ImportError("Pandas library is required to return DataFrames.")
                if results:
                    column_names = [description[0] for description in self.cursor.description]
                    df = pd.DataFrame(results, columns=column_names)
                    self.logger.debug(f"Returning {len(df)} results as DataFrame.")
                    return df
                else:
                    self.logger.debug("Returning empty DataFrame (no results).")
                    return pd.DataFrame()
            else:
                 self.logger.debug(f"Returning {len(results)} results as list of Row objects.")
                 return results # List of sqlite3.Row objects

        except sqlite3.Error as e:
            self.logger.error(f"Failed to execute SELECT query. Error: {e}", exc_info=True)
            self.logger.error(f"Failed Query: {sql_query}")
            if params:
                self.logger.error(f"Failed Params: {params}")
            return None # Indicate query execution error

    # --- Introspection Methods ---

    def get_table_names(self) -> List[str]:
        """
        Retrieves the names of all user-defined tables in the database.

        Returns:
            List[str]: A list of table names, or an empty list if none found or error.
        """
        self.logger.debug("Fetching table names...")
        query = "SELECT name FROM sqlite_master WHERE type='table';"
        results = self.execute_select_query(query, return_dataframe=False)

        if results is None:
             self.logger.error("Failed to retrieve table names during query execution.")
             return []

        table_names = [row['name'] for row in results]
        self.logger.info(f"Found {len(table_names)} table(s): {table_names}")
        return table_names

    def get_table_columns(self, table_name: str) -> Optional[List[Tuple[str, str]]]:
        """
        Retrieves the column names and declared types for a specific table using PRAGMA.

        Args:
            table_name (str): The name of the table to inspect.

        Returns:
            Optional[List[Tuple[str, str]]]: A list of (column_name, column_type) tuples,
                or None if the table doesn't exist or an error occurs.
        """
        self.logger.debug(f"Fetching columns for table: {table_name}...")
        query = f"PRAGMA table_info(\"{table_name}\");" # Quote table name for safety
        if not self.cursor:
             self.logger.error(f"Cannot fetch columns for '{table_name}', cursor unavailable.")
             raise ConnectionError("Database cursor is unavailable.")
        try:
            self.cursor.execute(query)
            columns_info = self.cursor.fetchall()
            if not columns_info:
                 self.logger.warning(f"No columns found for table '{table_name}' (or table does not exist).")
                 return None
            columns = [(col['name'], col['type']) for col in columns_info]
            self.logger.debug(f"Columns for '{table_name}': {columns}")
            return columns
        except sqlite3.Error as e:
            self.logger.error(f"Failed fetching columns for table '{table_name}': {e}", exc_info=True)
            return None

    def get_all_tables_and_columns(self) -> Dict[str, List[Tuple[str, str]]]:
        """
        Retrieves all user tables and their respective columns and types.

        Returns:
            Dict[str, List[Tuple[str, str]]]: Dictionary mapping table names to lists of
                                              (column_name, column_type) tuples.
        """
        self.logger.info("Fetching all tables and their columns...")
        all_tables_info = {}
        table_names = self.get_table_names()
        if not table_names:
            self.logger.info("No user tables found in the database.")
            return all_tables_info

        for table_name in table_names:
            columns = self.get_table_columns(table_name)
            if columns is not None:
                all_tables_info[table_name] = columns
            else:
                 self.logger.warning(f"Could not retrieve columns for table '{table_name}'.")
        self.logger.info(f"Retrieved column info for {len(all_tables_info)} table(s).")
        return all_tables_info

    def get_relationships(self) -> Dict[str, Dict[str, Any]]:
        """
        Identifies foreign key relationships defined in the database schema
        using `PRAGMA foreign_key_list` and attempts to infer relationship types.

        Note: Relationship inference uses heuristics. Requires foreign_keys=ON.

        Returns:
            Dict[str, Dict[str, Any]]: Dictionary describing the inferred relationships.
        """
        self.logger.info("Identifying foreign key relationships...")
        # Ensure FKs are enabled for this check
        fk_enabled = False
        try:
            # Check current status
            fk_status_cursor = self.conn.cursor() # Use separate cursor for safety?
            fk_status = fk_status_cursor.execute("PRAGMA foreign_keys;").fetchone()
            if fk_status and fk_status[0] == 1:
                 fk_enabled = True
                 self.logger.debug("Foreign key support is already enabled for this connection.")
            else:
                 self.execute_sql("PRAGMA foreign_keys = ON;")
                 fk_enabled = True
                 self.logger.info("Enabled foreign key support for relationship check.")
        except Exception as fk_e:
             self.logger.warning(f"Could not enable/verify foreign key support: {fk_e}. Relationship check may be incomplete.", exc_info=True)

        relationships = {}
        table_names = self.get_table_names()
        if not table_names:
            self.logger.info("No tables found to analyze for relationships.")
            return {}

        foreign_keys_by_table = {}

        if not self.cursor:
             self.logger.error("Cannot get relationships, cursor unavailable.")
             raise ConnectionError("Database cursor is unavailable.")

        self.logger.debug("Fetching foreign key definitions using PRAGMA foreign_key_list...")
        for table_name in table_names:
            query = f"PRAGMA foreign_key_list(\"{table_name}\");" # Quote table name
            try:
                self.cursor.execute(query)
                fks = self.cursor.fetchall()
                if fks:
                    self.logger.debug(f"Found {len(fks)} foreign key(s) originating from table '{table_name}'.")
                    foreign_keys_by_table[table_name] = fks
                    for fk in fks:
                        referenced_table = fk['table']
                        from_col = fk['from']
                        to_col = fk['to']
                        rel_key = (table_name, referenced_table)
                        if rel_key not in relationships:
                            relationships[rel_key] = []
                        relationships[rel_key].append({'from': from_col, 'to': to_col})
            except sqlite3.Error as e:
                self.logger.error(f"Failed fetching foreign keys for table '{table_name}': {e}", exc_info=True)

        # --- Infer Relationship Types (Heuristic based) ---
        self.logger.info("Inferring relationship types (One-to-Many, Many-to-Many)...")
        inferred_relationships = {}
        junction_candidates = set()

        # Identify potential junction tables
        for table, fks in foreign_keys_by_table.items():
            if len(fks) >= 2:
                cols_info = self.get_table_columns(table)
                if not cols_info: continue

                fk_col_names = {fk['from'] for fk in fks}
                # Get primary key columns using PRAGMA
                pk_cols = set()
                try:
                    pk_info = self.cursor.execute(f"PRAGMA table_info(\"{table}\")").fetchall()
                    pk_cols = {col['name'] for col in pk_info if col['pk'] > 0}
                except sqlite3.Error as pk_e:
                    self.logger.warning(f"Could not get PK info for table '{table}' during relationship check: {pk_e}")

                non_key_cols = [col[0] for col in cols_info if col[0] not in fk_col_names and col[0] not in pk_cols]

                if len(non_key_cols) <= 1: # Heuristic: Allow 0 or 1 extra attribute
                    junction_candidates.add(table)
                    self.logger.debug(f"Identified potential M:N Junction Table: '{table}'")

        # Process detected FKs
        processed_for_mn = set()
        for referencing_table, fks in foreign_keys_by_table.items():
            if referencing_table in junction_candidates:
                linked_tables = list({fk['table'] for fk in fks})
                if len(linked_tables) >= 2:
                    table1, table2 = linked_tables[0], linked_tables[1]
                    mn_key = tuple(sorted((table1, table2)))
                    rel_key_str = f"{mn_key[0]} <-> {mn_key[1]}"
                    if rel_key_str not in inferred_relationships:
                        inferred_relationships[rel_key_str] = {
                             'type': 'Many-to-Many', 'junction_table': referencing_table,
                             'details': f"{mn_key[0]} M:N {mn_key[1]} via {referencing_table}"}
                        self.logger.debug(f"Inferred M:N Relationship: {rel_key_str} via {referencing_table}")
                        processed_for_mn.add(referencing_table)

            if referencing_table not in processed_for_mn:
                 for fk in fks:
                     referenced_table = fk['table']
                     rel_key_str = f"{referenced_table} -> {referencing_table}"
                     part_of_mn = any(info.get('junction_table') == referencing_table for info in inferred_relationships.values())
                     if part_of_mn: continue
                     if rel_key_str not in inferred_relationships:
                        direct_rel_key = (referencing_table, referenced_table)
                        inferred_relationships[rel_key_str] = {
                             'type': 'One-to-Many',
                             'details': f"{referenced_table} (One) -> {referencing_table} (Many)",
                             'keys': relationships.get(direct_rel_key, [])}
                        self.logger.debug(f"Inferred 1:N Relationship: {rel_key_str}")

        self.logger.info(f"Relationship inference complete. Found {len(inferred_relationships)} potential relationships.")
        # Log summary
        if inferred_relationships:
            self.logger.info("--- Inferred Relationships Summary ---")
            for rel, info in inferred_relationships.items():
                self.logger.info(f"- Relationship: {rel} | Type: {info['type']} | Details: {info['details']}")
        else:
            self.logger.info("No specific foreign key relationships were inferred.")

        return inferred_relationships

    # --- Data Import/Export Methods (Requires pandas and openpyxl) ---

    def import_data_from_excel(self,
                               table_name: str,
                               excel_file_path: Optional[str] = None,
                               sheet_name: Union[str, int] = 0,
                               expected_columns: Optional[List[str]] = None,
                               clear_table_first: bool = False,
                               insert_strategy: SqlInsertStrategy = "INSERT OR REPLACE") -> bool:
        """ Imports data from Excel into a database table. """
        effective_excel_path = excel_file_path or self.excel_path
        self.logger.info(f"Starting Excel import: '{effective_excel_path}'[Sheet:{sheet_name}] -> Table:'{table_name}'")
        self.logger.info(f"Import options - Strategy: {insert_strategy}, Clear Table First: {clear_table_first}")

        if not self.cursor:
            self.logger.error("Cannot import from Excel, cursor unavailable.")
            raise ConnectionError("Database cursor is unavailable.")
        if not effective_excel_path:
            self.logger.error("No Excel file path specified for import.")
            raise ValueError("No Excel file path specified for import.")
        if not os.path.exists(effective_excel_path):
            self.logger.error(f"Excel file '{effective_excel_path}' not found.")
            raise FileNotFoundError(f"Excel file '{effective_excel_path}' not found.")
        if 'pd' not in globals():
            self.logger.error("Pandas library is required for Excel import but not found.")
            raise ImportError("Pandas library is required for Excel import.")
        try: import openpyxl # Check dependency
        except ImportError:
            self.logger.error("openpyxl library is required for Excel import but not found.")
            raise ImportError("The 'openpyxl' library is required for reading .xlsx files.")

        try:
            df = pd.read_excel(effective_excel_path, sheet_name=sheet_name, engine='openpyxl')
            self.logger.info(f"Read {len(df)} rows from Excel sheet '{sheet_name}'.")
            if df.empty:
                self.logger.info("Excel sheet is empty. No data to import.")
                return True

            # Column Selection/Validation
            actual_excel_columns = df.columns.tolist()
            if expected_columns:
                missing_cols = [col for col in expected_columns if col not in actual_excel_columns]
                if missing_cols:
                    self.logger.error(f"Excel sheet missing required columns: {missing_cols}. Aborting import.")
                    return False
                cols_to_insert = expected_columns
                try: df_processed = df[cols_to_insert].copy()
                except KeyError as e:
                    self.logger.error(f"Column mismatch during selection: {e}. Check 'expected_columns'. Aborting.", exc_info=True)
                    return False
            else:
                self.logger.info("Using all columns found in Excel sheet.")
                cols_to_insert = actual_excel_columns
                df_processed = df.copy()

            # Basic Data Cleaning (NaN -> None)
            df_processed = df_processed.where(pd.notnull(df_processed), None)
            data_to_insert = [tuple(row) for row in df_processed.to_numpy()]

            # Database Insertion
            if clear_table_first:
                self.logger.info(f"Clearing all data from table '{table_name}' before import...")
                delete_sql = f'DELETE FROM "{table_name}";'
                try:
                    self.execute_sql(delete_sql)
                    self.commit()
                    self.logger.info(f"Table '{table_name}' cleared successfully.")
                except Exception as del_e:
                    self.logger.error(f"Failed to clear table '{table_name}': {del_e}", exc_info=True)
                    self.rollback()
                    return False

            # Prepare and execute bulk insert
            placeholders = ", ".join(["?"] * len(cols_to_insert))
            cols_string = ", ".join(f'"{col}"' for col in cols_to_insert)
            if insert_strategy not in ["INSERT", "INSERT OR REPLACE", "INSERT OR IGNORE"]:
                 self.logger.warning(f"Invalid insert_strategy '{insert_strategy}'. Defaulting to 'INSERT OR REPLACE'.")
                 insert_strategy = "INSERT OR REPLACE"
            insert_sql = f"{insert_strategy} INTO \"{table_name}\" ({cols_string}) VALUES ({placeholders});"

            self.logger.info(f"Executing bulk insert ({insert_strategy}) with {len(data_to_insert)} rows into '{table_name}'...")
            self.cursor.executemany(insert_sql, data_to_insert)
            self.commit()
            self.logger.info(f"Bulk insert operation completed for table '{table_name}'.")
            return True

        except FileNotFoundError:
            self.logger.error(f"Excel file '{effective_excel_path}' could not be read (might have been deleted?).", exc_info=True)
            return False
        except pd.errors.EmptyDataError:
            self.logger.info(f"Excel sheet '{sheet_name}' is empty (pandas detected). No data imported.")
            return True
        except ImportError as e:
            self.logger.critical(f"Missing required library for Excel import: {e}", exc_info=True)
            raise
        except Exception as e:
            self.logger.error(f"An error occurred during data import: {type(e).__name__} - {e}", exc_info=True)
            self.rollback()
            return False

    def export_table_to_excel(self,
                              table_name: str,
                              excel_file_path: Optional[str] = None,
                              sheet_name: str = 'Sheet1',
                              sql_query: Optional[str] = None,
                              params: Optional[Tuple[Any, ...]] = None) -> bool:
        """ Exports data from a database table or query to an Excel file. """
        effective_excel_path = excel_file_path or self.excel_path
        self.logger.info(f"Starting data export to Excel '{effective_excel_path}' (Sheet: {sheet_name})...")

        if not self.cursor:
            self.logger.error("Cannot export to Excel, cursor unavailable.")
            raise ConnectionError("Database cursor is unavailable.")
        if not effective_excel_path:
            self.logger.error("No Excel file path specified for export.")
            raise ValueError("No Excel file path specified for export.")
        if 'pd' not in globals():
            self.logger.error("Pandas library is required for Excel export but not found.")
            raise ImportError("Pandas library is required for Excel export.")
        try: import openpyxl # Check dependency
        except ImportError:
            self.logger.error("openpyxl library is required for Excel export but not found.")
            raise ImportError("The 'openpyxl' library is required for writing .xlsx files.")

        try:
            # Determine query
            if sql_query:
                self.logger.info(f"Exporting based on provided SQL query.")
                query_to_run = sql_query
            else:
                self.logger.info(f"Exporting all data from table '{table_name}'.")
                query_to_run = f'SELECT * FROM "{table_name}";'
                params = None

            # Fetch data
            df_to_export = self.execute_select_query(query_to_run, params=params, return_dataframe=True)

            if df_to_export is None:
                self.logger.error("Failed to fetch data from database for export.")
                return False

            self.logger.info(f"Fetched {len(df_to_export)} rows for export.")

            # Export DataFrame to Excel
            self.logger.info(f"Writing data to '{effective_excel_path}', sheet '{sheet_name}'...")
            excel_dir = os.path.dirname(effective_excel_path)
            if excel_dir and not os.path.exists(excel_dir):
                os.makedirs(excel_dir, exist_ok=True)
                self.logger.info(f"Created directory '{excel_dir}' for Excel file.")

            df_to_export.to_excel(effective_excel_path, sheet_name=sheet_name, index=False, engine='openpyxl')
            self.logger.info(f"Data exported successfully to '{effective_excel_path}'.")
            return True

        except ImportError as e:
            self.logger.critical(f"Missing required library for Excel export: {e}", exc_info=True)
            raise
        except Exception as e:
            self.logger.error(f"An error occurred during data export: {type(e).__name__} - {e}", exc_info=True)
            return False

    # --- File Management ---

    def rename_database(self, new_db_path: str) -> bool:
        """
        Renames the database file currently associated with this instance.
        Closes connection first. Updates instance's `db_path` on success.

        Args:
            new_db_path (str): The new path/name for the database file.

        Returns:
            bool: True if renaming was successful, False otherwise.
        """
        original_path = self.db_path
        self.logger.info(f"Attempting to rename database from '{original_path}' to '{new_db_path}'...")

        # Ensure connection is closed
        if self.conn:
            self.logger.info("Closing active connection before renaming...")
            self.close()

        if not os.path.exists(original_path):
            self.logger.error(f"Original database file '{original_path}' does not exist. Cannot rename.")
            return False
        
        if os.path.exists(new_db_path):
            os.remove(new_db_path)

        # Ensure target directory exists
        new_dir = os.path.dirname(new_db_path)
        if new_dir and not os.path.exists(new_dir):
            try:
                os.makedirs(new_dir, exist_ok=True)
                self.logger.info(f"Created directory '{new_dir}' for new database path.")
            except OSError as dir_e:
                self.logger.error(f"Failed to create directory for new path '{new_dir}': {dir_e}", exc_info=True)
                return False

        try:
            self.logger.debug(f"Executing os.rename('{original_path}', '{new_db_path}')...")
            shutil.copyfile(original_path, new_db_path)
            self.logger.info(f"Database successfully renamed to '{new_db_path}'")
            self.db_path = new_db_path # Update internal path ONLY on success
            return True
        except OSError as e:
            self.logger.error(f"Failed to rename database file using os.rename: {e}", exc_info=True)
            return False
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during renaming: {type(e).__name__} - {e}", exc_info=True)
            return False

# End of DatabaseManager class definition
