from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import os
import time
import requests

def download_dat_files(url, save_dir=".", delay=1):
    """
    Downloads all .dat files from the given URL and its subsequent pages.

    Args:
        url (str): The base URL of the webpage containing the .dat files.
        save_dir (str, optional): The directory to save the downloaded .dat files. 
            Defaults to the current directory (.).
        delay (int, optional): Delay between requests in seconds. Defaults to 1.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Initialize WebDriver (e.g., Chrome)
    driver = webdriver.Chrome()  # Replace with your preferred driver
    driver.get(url)

    while True:
        # Wait for the page to fully load
        WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located((By.TAG_NAME, 'a'))) 

        # Get the page source after JavaScript rendering
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')

        # Find all links to .dat files with better error handling
        dat_links = []
        for link in soup.find_all('a'):
            try:
                if link.has_attr('href') and link['href'].endswith('.dat'):
                    dat_links.append(link['href'])
            except:
                pass 

        # Download each .dat file
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
            except Exception as e:
                print(f"Error downloading {dat_url}: {e}")

        # Introduce a short delay between requests
        time.sleep(delay)

        # Find the link to the next page (if it exists)
        next_page_link = driver.find_element(By.LINK_TEXT, "Next")
        try:
            next_page_link.click()
        except:
            break  # No more pages

    driver.quit()

if __name__ == "__main__":
    download_dat_files("https://m-selig.ae.illinois.edu/ads/coord_database.html", save_dir="airfoil_data")
    print("Finished downloading all .dat files.")