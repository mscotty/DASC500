from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import os
import time
import requests

from DASC500.classes.AirfoilDatabase import AirfoilDatabase
from DASC500.classes.AirfoilSeries import AirfoilSeries

def download_dat_files(url, 
                       save_dir=".", 
                       db_name="airfoil_data.db", 
                       db_dir=".", 
                       overwrite=False,
                       delay=1):
    """
    Downloads all .dat files from the given URL and its subsequent pages,
    parses the data, and stores it in an SQLite database.

    Args:
        url (str): The base URL of the webpage containing the .dat files.
        save_dir (str, optional): The directory to save the downloaded .dat files. Defaults to the current directory (.).
        db_name (str, optional): The name of the SQLite database file. Defaults to "airfoil_data.db".
        db_dir (str, optional): The directory to save the SQLite database. Defaults to the current directory (.).
        overwrite (logical, optional): Dictates if data already present in the specified database should be overwritten. Default to "False".
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
                    airfoil_series = AirfoilSeries.identify_airfoil_series(name)
                    if airfoil_series == AirfoilSeries.OTHER:
                        airfoil_series = AirfoilSeries.identify_airfoil_series(description)
                    with open(dat_filename, 'r') as f:
                        lines = f.readlines()
                        description = lines[0].strip() if lines else ""
                        pointcloud = ""

                        for line in lines[1:]: # Skip the first line
                            line = line.strip() # Remove leading/trailing whitespace
                            if line:  # Check if the line is not empty
                                try:
                                    x, y = map(float, line.split()) # Convert to float and unpack
                                    if abs(x) <= 1.0 and abs(y) <= 1.0: # Check if both coordinates are within [-1, 1]
                                        pointcloud += f"{x} {y}\n" # Add the point only if it's valid
                                except ValueError:
                                    pass # Skip lines that can not be converted to floats or do not have 2 values.
                                    
                    airfoil_db.store_airfoil_data(name, description, pointcloud, airfoil_series, dat_url, overwrite=overwrite)

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
    download_dat_files("https://m-selig.ae.illinois.edu/ads/coord_database.html", 
                       save_dir="airfoil_data", 
                       db_dir="my_airfoil_database",
                       overwrite=True)  # Specify database directory
    print("Finished processing all .dat files.")