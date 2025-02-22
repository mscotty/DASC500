import os
import subprocess
import tempfile
import pandas as pd
import numpy as np

import sqlite3

from DASC500.xfoil.fix_airfoil_data import *

class XFoilRunner:
    def __init__(self, 
                 xfoil_path="xfoil", 
                 polar_dir="D:\Mitchell\School\polar_data", 
                 log_failed_runs=True):
        self.xfoil_path = xfoil_path
        self.polar_dir = polar_dir
        self.log_failed_runs = log_failed_runs
        self.failed_runs_log = "failed_runs.txt"
        os.makedirs(self.polar_dir, exist_ok=True)
    
    def _initialize_database(self):
        with sqlite3.connect(self.database_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    airfoil TEXT,
                    mach REAL,
                    reynolds REAL,
                    alpha REAL,
                    cl REAL,
                    cd REAL,
                    cm REAL
                )
            ''')
            conn.commit()
    
    def run_xfoil(self, 
                  airfoil_name, 
                  pointcloud_str, 
                  alpha_start=10, 
                  alpha_end=15, 
                  alpha_increment=1, 
                  Re=100000,
                  Mach=0.1):
        """Runs XFoil for a given airfoil and pointcloud, generating a polar."""
        # Run checks on the pointcloud provided
        rows = pointcloud_str.split('\n')
        rows = [x for x in rows if x.strip()]
        pointcloud_np = np.array([np.fromstring(row, dtype=float, sep=' ') for row in rows])
        pointcloud_np = reorder_airfoil_data(pointcloud_np)
        pointcloud_reorder = ""
        for row in pointcloud_np:
            pointcloud_reorder += " ".join(f"{val:.6f}" for val in row) + "\n"

        # Use a temporary file for the airfoil coordinates
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dat", mode="w") as temp_airfoil_file:
            temp_airfoil_file.write(pointcloud_reorder)
            airfoil_file = temp_airfoil_file.name

        airfoil_name_plain = os.path.split(airfoil_name)[1]
        polar_file = os.path.join(self.polar_dir, f"{airfoil_name_plain}.pol")

        try:
            if os.path.isfile(polar_file):
                os.remove(polar_file)
                
            command = [self.xfoil_path, "-no_gui"]

            with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
                xfoil_commands = f"""
                    load {airfoil_file}
                    {airfoil_name_plain}
                    pane
                    ppar
                    n 240
                    t 1

                    
                    oper
                    re {Re}
                    m {Mach}
                    iter 200
                    pacc
                    {polar_file}
                    
                    aseq {alpha_start} {alpha_end} {alpha_increment}
                    pacc
                    
                    quit
                """

                stdout, stderr = process.communicate(input=xfoil_commands)

                if process.returncode != 0:
                    print(f"XFoil error for {airfoil_name}: {stderr}")
                    return None  # Indicate failure

                # Read the polar data with pandas
                try:
                    polar_df = pd.read_csv(polar_file, 
                                           skiprows=12, 
                                           delim_whitespace=True, 
                                           names=["alpha", "cl", "cd", "cm"])
                    print(f"XFoil ran successfully for {airfoil_name}")
                    return polar_df
                except pd.errors.EmptyDataError:
                    print(f"Polar file is empty for {airfoil_name}")
                    self._log_failed_run(airfoil_name, Mach, Re, alpha_start, alpha_end, alpha_increment)
                    return None
                except FileNotFoundError:
                    print(f"Polar file not found for {polar_file}")
                    return None

        except FileNotFoundError:
            print(f"XFoil executable not found at {self.xfoil_path}")
            self._log_failed_run(airfoil_name, Mach, Re, alpha_start, alpha_end, alpha_increment)
            return None
        except Exception as e:
            print(f"An error occurred while running XFoil: {e}")
            self._log_failed_run(airfoil_name, Mach, Re, alpha_start, alpha_end, alpha_increment)
            return None
        finally:
            # Cleanup
            if os.path.exists(airfoil_file):
                os.remove(airfoil_file)

    def run_xfoil_multi(self, 
                        airfoil_name, 
                        pointcloud_str, 
                        reynolds_list=[10000, 50000, 100000], 
                        mach_list=[0.01, 0.1, 0.5], 
                        alpha_start=0, 
                        alpha_end=10, 
                        alpha_increment=2):
        """Runs XFoil for multiple Reynolds numbers, Mach numbers, and angle of attack ranges."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dat", mode="w") as temp_airfoil_file:
            temp_airfoil_file.write(pointcloud_str)
            airfoil_file = temp_airfoil_file.name
        
        results = []
        
        for Re in reynolds_list:
            for Mach in mach_list:
                polar_file = os.path.join(self.polar_dir, f"{airfoil_name}_Re{Re}_M{Mach}.pol")
                if os.path.isfile(polar_file):
                    os.remove(polar_file)
                
                command = [self.xfoil_path]
                
                with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
                    xfoil_commands = f"""
                        load {airfoil_file}
                        {airfoil_name}
                        pane
                        oper
                        visc {Re}
                        m {Mach}
                        iter 200
                        pacc
                        {polar_file}
                        
                        aseq {alpha_start} {alpha_end} {alpha_increment}
                        pacc
                        
                        quit
                    """
                    stdout, stderr = process.communicate(input=xfoil_commands)
                    
                    if process.returncode != 0:
                        print(f"XFoil error for {airfoil_name} at Re={Re}, Mach={Mach}: {stderr}")
                        continue
                    
                    try:
                        print(f'Polar file at: {polar_file}')
                        polar_df = pd.read_csv(polar_file, skiprows=12, delim_whitespace=True, names=["alpha", "cl", "cd", "cm"])
                        polar_df["Re"] = Re
                        polar_df["Mach"] = Mach
                        results.append(polar_df)
                    except (pd.errors.EmptyDataError, FileNotFoundError):
                        print(f"Polar file not found or empty for {airfoil_name} at Re={Re}, Mach={Mach}")
        
        if os.path.exists(airfoil_file):
            os.remove(airfoil_file)
        
        return pd.concat(results, ignore_index=True) if results else None
    
    def log_failed_run(self, airfoil, mach, reynolds, alpha):
        if self.log_failed_runs:
            with open(self.failed_runs_log, "a") as log_file:
                log_file.write(f"{airfoil}, {mach}, {reynolds}, {alpha}\n")
        
