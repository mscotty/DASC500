import datetime
import csv
from typing import List, Tuple
import os

from DASC500.utilities.get_top_level_module import get_top_level_module_path # Used for pathing
from DASC500.utilities.print.redirect_print import redirect_print # Used for logging output

# Problem 1 (1 point)

# Enter your name below
name = "Mitchell Scott"


class DataUtil:
    """
    A utility class for parsing date strings and reading dataset files.
    """

    @staticmethod
    def parse_date_string(date_str: str) -> datetime.date:
        """
        Parses a date string with varying levels of specificity into a datetime.date object.

        Args:
            date_str (str): A string representing a date, which may include year, month, and day
                            separated by dashes. Acceptable formats: "YYYY", "YYYY-MM", "YYYY-MM-DD".

        Returns:
            datetime.date: A date object with missing parts defaulted to 1.
        """
        parts = date_str.split('-')
        year = int(parts[0])
        month = int(parts[1]) if len(parts) > 1 else 1
        day = int(parts[2]) if len(parts) > 2 else 1
        return datetime.date(year, month, day)

    @staticmethod
    def read_dataset_csv_file(file_path: str) -> List[Tuple[int, int, datetime.date]]:
        """
        Reads a CSV file and converts each line into a tuple of two integers and a parsed date.

        Args:
            file_path (str): The path to the input CSV file.

        Returns:
            List[Tuple[int, int, datetime.date]]: A list of tuples containing the first two numeric
                                                  values and a parsed date object.
        """
        dataset = []
        with open(file_path, mode='r', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) != 3:
                    continue  # skip invalid rows
                val1 = int(row[0])
                val2 = int(row[1])
                date = DataUtil.parse_date_string(row[2])
                dataset.append((val1, val2, date))
        return dataset


if __name__ == "__main__":
    # Handle filepathing to input/output folders
    input_file = os.path.join(get_top_level_module_path(), r'..\..\data\DASC510\test.txt')
    output_logger = os.path.join(get_top_level_module_path(), r'..\..\outputs\DASC510\homework3_logger.txt')
    
    # Redirect stdout to a logger file
    redirect_print(output_logger, also_to_stdout=True)
    
    # Problem 2: Test `parse_date_string`
print("Testing parse_date_string...")

d1 = DataUtil.parse_date_string("2002")
assert isinstance(d1, datetime.date)
assert d1.year == 2002
assert d1.month == 1
assert d1.day == 1
print("Test 1 passed: '2002' ->", d1)

d2 = DataUtil.parse_date_string("2021-03")
assert isinstance(d2, datetime.date)
assert d2.year == 2021
assert d2.month == 3
assert d2.day == 1
print("Test 2 passed: '2021-03' ->", d2)

d3 = DataUtil.parse_date_string("2020-3-7")
assert isinstance(d3, datetime.date)
assert d3.year == 2020
assert d3.month == 3
assert d3.day == 7
print("Test 3 passed: '2020-3-7' ->", d3)

d4 = DataUtil.parse_date_string("2019-12-25")
assert isinstance(d4, datetime.date)
assert d4.year == 2019
assert d4.month == 12
assert d4.day == 25
print("Test 4 passed: '2019-12-25' ->", d4)

print("All parse_date_string tests passed.\n")


# Problem 3: Test `read_dataset_csv_file`
print("Testing read_dataset_csv_file...")

test_dataset = DataUtil.read_dataset_csv_file(input_file)
assert len(test_dataset) == 4

for idx, item in enumerate(test_dataset):
    assert isinstance(item, tuple)
    assert isinstance(item[0], (float, int))
    assert isinstance(item[1], (float, int))
    assert isinstance(item[2], datetime.date)
    print(f"Row {idx + 1} passed: {item}")

print("All read_dataset_csv_file tests passed.")
