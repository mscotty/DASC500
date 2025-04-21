import os
import sys
import sqlite3
import logging

# --- Import the updated DatabaseManager ---
# Assumption: The DatabaseManager class is saved in database_manager.py
try:
    from DASC500.classes.DatabaseManager import DatabaseManager
except ImportError:
    print(
        "ERROR: Could not import DatabaseManager. Make sure 'database_manager.py' exists in the same directory.",
        file=sys.stderr,
    )
    sys.exit(1)

from DASC500.utilities.get_top_level_module import get_top_level_module_path

def configure_logger(log_file, debug=True):
    log_level = logging.DEBUG if debug else logging.INFO
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create file handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    # Also log to console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    logger = logging.getLogger("homework_logger")
    logger.setLevel(log_level)

    # Avoid duplicate handlers if script is re-run
    if not logger.handlers:
        logger.addHandler(file_handler)
        # logger.addHandler(console_handler)  # Optional: enable for debug

    return logger, log_level


def main(db_file, new_name=None, log_file="homework2_logger.txt", debug=True):
    if os.path.exists(log_file):
        os.remove(log_file)
        
    script_logger, log_level = configure_logger(log_file, debug)

    script_logger.info("--- Starting Homework Script ---")
    script_logger.info(f"Database file specified: {db_file}")
    if new_name:
        script_logger.info(f"Target rename path: {new_name}")
    if log_file:
        script_logger.info(f"DatabaseManager log file: {log_file}")

    db_manager = (
        None  # Initialize to ensure it's defined for finally block if init fails
    )
    original_db_path = db_file  # Keep track for potential rename later

    try:
        # --- Instantiate DatabaseManager ---
        # Pass the log file path if provided
        db_manager = DatabaseManager(
            db_path=original_db_path, log_file=log_file, log_level=log_level
        )  # Pass script log level

        # --- Perform Tasks using Context Manager ---
        with db_manager:
            script_logger.info("\n--- Task 1: Get Table Names ---")
            table_names = db_manager.get_table_names()
            script_logger.info(f"Tables found: {table_names}")

            script_logger.info("\n--- Task 2: Get Columns for Each Table ---")
            all_cols = db_manager.get_all_tables_and_columns()
            # Log details (already logged by manager, maybe add summary here)
            script_logger.info(f"Retrieved column info for {len(all_cols)} table(s).")
            # Example of accessing details if needed:
            # for table, cols in all_cols.items():
            #    script_logger.info(f"  Table '{table}': {cols}")

            script_logger.info("\n--- Task 3: Identify Relationships ---")
            relationships = db_manager.get_relationships()
            # Summary logged by manager, add confirmation here
            script_logger.info(
                f"Relationship analysis complete. Found {len(relationships)} inferred relationships."
            )

            script_logger.info("\n--- Task 4: Execute Simple Query ---")
            example_query = "SELECT * FROM Patron LIMIT 5;"  # Generic enough for demo
            try:
                query_results_df = db_manager.execute_select_query(
                    example_query, return_dataframe=True
                )
                if query_results_df is not None:
                    script_logger.info(f"Query '{example_query}' executed. Results:")
                    # Print DataFrame nicely using pandas toString
                    try:
                        print(query_results_df.to_string())
                    except Exception:  # Handle potential errors in to_string
                        print(query_results_df)
                else:
                    script_logger.warning(
                        f"Query '{example_query}' failed or returned no results."
                    )
            except sqlite3.Error as query_e:
                script_logger.error(f"Failed to execute example query: {query_e}")
            except ImportError:
                script_logger.warning(
                    "Pandas not found, cannot display query results as table."
                )

        # --- Task 5: Rename Database (after closing connection) ---
        script_logger.info("\n--- Task 5: Rename Database ---")
        if new_name:
            # The connection is closed by the 'with' statement exiting.
            # We can reuse the same manager instance to call rename.
            # The rename method handles closing connection again if needed (idempotent)
            # and updates the manager's internal db_path on success.
            success = db_manager.rename_database(new_name)
            if success:
                script_logger.info(
                    f"Database successfully renamed from '{original_db_path}' to '{new_name}'."
                )
                # Verify existence
                if os.path.exists(new_name):
                    script_logger.info(f"Verified: Renamed file '{new_name}' exists.")
                else:
                    script_logger.error(
                        f"Verification failed: Renamed file '{new_name}' not found!"
                    )
            else:
                script_logger.error(f"Failed to rename database to '{new_name}'.")
        else:
            script_logger.info("No new name provided (--new-name), skipping rename.")

    except FileNotFoundError as e:
        script_logger.critical(f"File not found error: {e}", exc_info=True)
        sys.exit(1)
    except ConnectionError as e:
        script_logger.critical(f"Database connection error: {e}", exc_info=True)
        sys.exit(1)
    except ImportError as e:  # Catch import error for DatabaseManager itself
        script_logger.critical(f"Import error: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        # Catch any other unexpected errors
        script_logger.critical(
            f"An unexpected error occurred: {type(e).__name__} - {e}", exc_info=True
        )
        sys.exit(1)
    finally:
        # Ensure manager connection is closed if an error occurred before/during 'with'
        if db_manager and db_manager.conn:
            script_logger.warning(
                "Attempting final close on DB manager due to potential early exit."
            )
            db_manager.close()
        script_logger.info("--- Homework Script Finished ---")


if __name__ == "__main__":
    db_file = os.path.join(
        get_top_level_module_path(), "../../data/DASC501/unknown2023.db"
    )
    new_name = os.path.join(
        get_top_level_module_path(),
        "../../outputs/DASC501/homework2/auction_data_2023.db",
    )
    log_file = os.path.join(
        get_top_level_module_path(), 
        "../../outputs/DASC501/homework2/logger.txt"
    )
    main(db_file, new_name)
