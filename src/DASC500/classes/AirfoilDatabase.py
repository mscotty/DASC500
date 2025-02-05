from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import os
import time
import requests
import sqlite3

class AirfoilDatabase:
    def __init__(self, db_name="airfoil_data.db", db_dir="."):
        self.db_path = os.path.join(db_dir, db_name) # Path to the database
        os.makedirs(db_dir, exist_ok=True) # Create directory if it doesn't exist.
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self._create_table()

    def _create_table(self):
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS airfoils (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                pointcloud TEXT
            )
        """)
        self.conn.commit()

    def store_airfoil_data(self, name, description, pointcloud):
        try:
            self.cursor.execute("INSERT INTO airfoils (name, description, pointcloud) VALUES (?, ?, ?)", (name, description, pointcloud))
            self.conn.commit()
            print(f"Stored: {name} in database.")
        except sqlite3.IntegrityError:
            print(f"Airfoil {name} already exists in the database.")

    def get_airfoil_data(self, name):
        self.cursor.execute("SELECT description, pointcloud FROM airfoils WHERE name=?", (name,))
        return self.cursor.fetchone()

    def close(self):
        self.conn.close()

def download_dat_files(url, save_dir=".", db_name="airfoil_data.db", db_dir=".", delay=1):
    """
    Downloads all .dat files from the given URL and its subsequent pages,
    parses the data, and stores it in an SQLite database.

    Args:
        url (str): The base URL of the webpage containing the .dat files.
        save_dir (str, optional): The directory to save the downloaded .dat files. Defaults to the current directory (.).
        db_name (str, optional): The name of the SQLite database file. Defaults to "airfoil_data.db".
        db_dir (str, optional): The directory to save the SQLite database. Defaults to the current directory (.).
        delay (int, optional): Delay between requests in seconds. Defaults to 1.
    """

    os.makedirs(save_dir, exist_ok=True)
    airfoil_db = AirfoilDatabase(db_name, db_dir)  # Initialize the database object

    driver = webdriver.Chrome()
    driver.get(url)

    while True:
        WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located((By.TAG_NAME, 'a')))
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')

        dat_links = []
        for link in soup.find_all('a'):
            try:
                if link.has_attr('href') and link['href'].endswith('.dat'):
                    dat_links.append(link['href'])
            except:
                pass

        for dat_link in dat_links:
            dat_url = f"https://m-selig.ae.illinois.edu/ads/{dat_link}"
            try:
                response = requests.get(dat_url, stream=True)
                if response.status_code == 200:
                    dat_filename = os.path.join(save_dir, os.path.basename(dat_link))
                    with open(dat_filename, 'wb') as f:
                        for chunk in response.iter_content(1024):
                            f.write(chunk)
                    print(f"Downloaded: {dat_filename}")

                    name = os.path.splitext(os.path.basename(dat_link))[0]
                    with open(dat_filename, 'r') as f:
                        lines = f.readlines()
                        description = lines[0].strip() if lines else ""
                        pointcloud = "".join(lines[1:]) if len(lines) > 1 else ""
                    airfoil_db.store_airfoil_data(name, description, pointcloud)  # Use the database object

            except Exception as e:
                print(f"Error downloading or processing {dat_url}: {e}")

            time.sleep(delay)

        try:
            next_page_link = driver.find_element(By.LINK_TEXT, "Next")
            next_page_link.click()
        except:
            break

    driver.quit()
    airfoil_db.close()  # Close the database connection

if __name__ == "__main__":
    download_dat_files("https://m-selig.ae.illinois.edu/ads/coord_database.html", save_dir="airfoil_data", db_dir="my_airfoil_database")  # Specify database directory
    print("Finished processing all .dat files.")

    # Example usage of the AirfoilDatabase class:
    airfoil_db = AirfoilDatabase(db_dir="my_airfoil_database") # Same directory
    data = airfoil_db.get_airfoil_data("AG25 Bubble Dancer DLG by Mark Drela")
    if data:
        description, pointcloud = data
        print(f"Description: {description}")
        #print(f"Pointcloud: {pointcloud}") # Pointcloud can be very long.
    airfoil_db.close()