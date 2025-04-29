import sys
import os

def redirect_print(filepath, also_to_stdout=False):
    """
    Redirects all subsequent print statements to the specified file, using UTF-8 encoding.

    Args:
        filepath (str): The path to the file where print statements will be written.
        also_to_stdout (bool, optional): If True, print statements will also be
                                        displayed in the command window. Defaults to False.
    """
    original_stdout = sys.stdout
    try:
        file = open(filepath, 'w', encoding='utf-8')  # Specify UTF-8 encoding here
    except Exception as e:
        print(f"Error opening file for redirection: {e}")
        return  # Exit if the file cannot be opened

    class DualOutput:
        def __init__(self, file, stdout):
            self.file = file
            self.stdout = stdout

        def write(self, data):
            self.file.write(data)
            if self.stdout:
                self.stdout.write(data)
                self.stdout.flush()

        def flush(self):
            self.file.flush()
            if self.stdout:
                self.stdout.flush()

        def close(self):
            self.file.close()
            if self.stdout:
                self.stdout = None  # Avoid closing standard output

    if also_to_stdout:
        sys.stdout = DualOutput(file, original_stdout)
    else:
        sys.stdout = file

def restore_print():
    """
    Restores print statements to their default behavior (command window).
    """
    if hasattr(sys.stdout, 'close'):  # Check if sys.stdout has a close method
        sys.stdout.close()
    sys.stdout = sys.__stdout__