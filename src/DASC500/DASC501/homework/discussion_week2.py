import sqlite3
import os # Used for the optional cleanup

class SimpleSQLiteManager:
    """
    A simplified approach to basic SQLite operations.
    Demonstrates:
        - connection
        - table creation
        - commit
        - close 
    as methods.
    """
    def __init__(self, db_name):
        """Initializes the manager with the database name."""
        self.db_name = db_name
        self.conn = None
        self.cursor = None
        print(f"INFO: SimpleSQLiteManager initialized for database '{self.db_name}'.")

    def connect_db(self):
        """
        Task 2: Create database file (if needed) and establish connection.
        """
        print(f"Attempting to connect to/create database: '{self.db_name}'...")
        # sqlite3.connect creates the file if it doesn't exist.
        self.conn = sqlite3.connect(self.db_name)
        self.cursor = self.conn.cursor()
        print("SUCCESS: Database connection established and cursor created.")

    def create_table(self, table_name, sql_create_command):
        """
        Task 3: Create an empty table with named columns.
        """
        print(f"Executing SQL to create table '{table_name}'...")
        self.cursor.execute(sql_create_command)
        print(f"SUCCESS: Table '{table_name}' creation command executed.")
        # Note: Table might not be fully visible until committed.

    def commit_changes(self):
        """
        Task 4 (Part 1): Commit the changes.
        """
        print("Committing changes to the database...")
        self.conn.commit()
        print("SUCCESS: Changes committed.")

    def close_db(self):
        """
        Task 4 (Part 2): Close the database connection.
        """
        print("Closing the database connection...")
        self.conn.close()
        print("SUCCESS: Database connection closed.")

# --- Main Execution Logic ---
if __name__ == "__main__":
    DATABASE_FILE = 'DiscussionWeek2.db'
    TABLE_NAME = 'ExampleTable'

    # --- Optional: Clean up existing file ---
    if os.path.exists(DATABASE_FILE):
        print(f"INFO: Deleting existing database file '{DATABASE_FILE}' for a clean run.")
        os.remove(DATABASE_FILE)
    # --------------------------------------

    print("\n--- Starting Simplified SQLite Operations ---")

    # Instantiate the manager
    db_manager = SimpleSQLiteManager(DATABASE_FILE)

    # Call methods in sequence
    db_manager.connect_db()

    # Define the SQL for table creation
    sql_create = f"""
    CREATE TABLE {TABLE_NAME} (
        RecordID INTEGER PRIMARY KEY AUTOINCREMENT,
        ItemName TEXT,
        Quantity INTEGER DEFAULT 0,
        Notes TEXT
    );
    """
    db_manager.create_table(TABLE_NAME, sql_create)

    db_manager.commit_changes()
    db_manager.close_db()

    print("--- Simplified SQLite Operations Finished ---")

    # --- Optional: Verification ---
    if os.path.exists(DATABASE_FILE):
         print(f"\nVERIFICATION: Database file '{DATABASE_FILE}' exists.")
    # ----------------------------