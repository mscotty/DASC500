import pandas as pd
import json
import xml.etree.ElementTree as ET
import h5py
import sqlite3
import io
import os
import requests
from PIL import Image  # Python Imaging Library
import logging

from DASC500.utilities.get_top_level_module import get_top_level_module_path
from DASC500.utilities.print.redirect_print import redirect_print

from DASC500.classes.DatabaseManager import DatabaseManager

INPUT_FOLDER = os.path.join(get_top_level_module_path(), '../../data/DASC501/homework5')
OUTPUT_FOLDER = os.path.join(get_top_level_module_path(), '../../outputs/DASC501/homework5')

redirect_print(os.path.join(OUTPUT_FOLDER, 'homework5_logger.txt'))

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DB_PATH = os.path.join(OUTPUT_FOLDER, "aircraft_maintenance.db")  # Database file in the same directory


def read_data_files(
    h5_file=os.path.join(INPUT_FOLDER, "aircraft_data.h5"),
    pkl_file=os.path.join(INPUT_FOLDER, "aircraft_data.pkl"),
    xml_file=os.path.join(INPUT_FOLDER, "aircraft_data.xml"),
    json_file=os.path.join(INPUT_FOLDER, "aircraft_data.json"),
):
    """
    Reads data from various file formats (h5, pkl, xml, json) using pandas
    or other appropriate libraries.

    Returns:
        tuple: A tuple containing DataFrames (or lists of dictionaries)
               representing data from each file.
    """
    try:
        # Read HDF5 file
        with h5py.File(h5_file, "r") as hf:
            data = {}
            for k in hf:
                data[k] = hf[k][:]
            df_h5 = pd.DataFrame({k: v for k, v in data.items()})
        logging.info(f"Successfully read data from {h5_file}")
    except Exception as e:
        logging.error(f"Error reading {h5_file}: {e}")
        df_h5 = pd.DataFrame()  # Return empty DataFrame in case of error

    try:
        # Read Pickle file
        df_pkl = pd.read_pickle(pkl_file)
        logging.info(f"Successfully read data from {pkl_file}")
    except Exception as e:
        logging.error(f"Error reading {pkl_file}: {e}")
        df_pkl = pd.DataFrame()

    try:
        # Read XML file
        tree = ET.parse(xml_file)
        root = tree.getroot()
        data = []
        for record in root:
            record_data = {}
            for element in record:
                record_data[element.tag] = element.text
            data.append(record_data)
        df_xml = pd.DataFrame(data)
        logging.info(f"Successfully read data from {xml_file}")
    except Exception as e:
        logging.error(f"Error reading {xml_file}: {e}")
        df_xml = pd.DataFrame()

    try:
        # Read JSON file
        with open(json_file, "r") as f:
            data = json.load(f)
        df_json = pd.DataFrame(data)
        logging.info(f"Successfully read data from {json_file}")
    except Exception as e:
        logging.error(f"Error reading {json_file}: {e}")
        df_json = pd.DataFrame()

    return df_h5, df_pkl, df_xml, df_json


def create_aircraft_table(cursor):
    """Creates the 'aircraft_maintenance' table in the database."""

    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS aircraft_maintenance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        aircraft_name TEXT,
        maintenance_date TEXT,
        maintenance_type TEXT,
        cost REAL,
        location TEXT,
        aircraft_image BLOB
    )
    """
    )


def insert_data_into_table(cursor, df, table_name="aircraft_maintenance"):
    """
    Inserts data from a DataFrame into the specified database table.

    Args:
        cursor: Database cursor object.
        df (pd.DataFrame): DataFrame containing data to insert.
        table_name (str, optional): Name of the table.
    """

    for _, row in df.iterrows():
        try:
            # Adjust column names to match the database table
            cursor.execute(
                f"""
            INSERT INTO {table_name} (aircraft_name, maintenance_date, maintenance_type, cost, location)
            VALUES (?, ?, ?, ?, ?)
            """,
                (
                    row.get("aircraft_name")
                    or row.get("name")
                    or row.get("Name"),  # Handle different naming
                    row.get("maintenance_date")
                    or row.get("date")
                    or row.get("Date"),  # Handle different naming
                    row.get("maintenance_type")
                    or row.get("type")
                    or row.get("Type"),  # Handle different naming
                    row.get("cost") or row.get("Cost"),  # Handle different naming
                    row.get("location")
                    or row.get("Location"),  # Handle different naming
                ),
            )
        except sqlite3.Error as e:
            logging.error(f"Error inserting row: {e}")


def add_aircraft_image_column(cursor):
    """Adds the 'aircraft_image' column to the 'aircraft_maintenance' table."""

    cursor.execute(
        """
    ALTER TABLE aircraft_maintenance ADD COLUMN aircraft_image BLOB
    """
    )


def load_images_into_table(cursor):
    """
    Downloads images for specific aircraft, converts them to binary,
    and loads them into the 'aircraft_image' column.
    """

    image_urls = {
        "F-16 Fighting Falcon": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8c/F-16_Fighting_Falcon_-_Don_Muang_2011.jpg/1024px-F-16_Fighting_Falcon_-_Don_Muang_2011.jpg",
        "B-2 Spirit": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/B-2_Spirit_of_Missouri_in_flight.jpg/1024px-B-2_Spirit_of_Missouri_in_flight.jpg",
    }

    for aircraft_name, image_url in image_urls.items():
        try:
            response = requests.get(image_url)
            response.raise_for_status()  # Raise an exception for bad status codes

            img = Image.open(io.BytesIO(response.content))
            img = img.resize((200, 200))  # Resize for storage

            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format="JPEG")  # Save as JPEG
            img_byte_arr = img_byte_arr.getvalue()

            cursor.execute(
                """
                UPDATE aircraft_maintenance SET aircraft_image = ? WHERE aircraft_name = ?
                """,
                (sqlite3.Binary(img_byte_arr), aircraft_name),
            )
            logging.info(f"Successfully loaded image for {aircraft_name}")

        except requests.exceptions.RequestException as e:
            logging.error(f"Error downloading image for {aircraft_name}: {e}")
        except Exception as e:
            logging.error(f"Error processing/inserting image for {aircraft_name}: {e}")


def main():
    """Main function to orchestrate the data loading and database operations."""

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        create_aircraft_table(cursor)

        df_h5, df_pkl, df_xml, df_json = read_data_files()

        # Insert data from each DataFrame
        insert_data_into_table(cursor, df_h5)
        insert_data_into_table(cursor, df_pkl)
        insert_data_into_table(cursor, df_xml)
        insert_data_into_table(cursor, df_json)

        add_aircraft_image_column(cursor)
        load_images_into_table(cursor)

        conn.commit()
        logging.info("All data loaded and processed successfully.")

    except sqlite3.Error as db_err:
        logging.error(f"Database error: {db_err}")
        if conn:
            conn.rollback()  # Rollback in case of database error
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    main()