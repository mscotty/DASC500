import os
import math
import sqlite3
import matplotlib.pyplot as plt
import concurrent.futures
import seaborn as sns
import pandas as pd
import numpy as np

from DASC500.formulas.airfoil.compute_aspect_ratio import calculate_aspect_ratio
from DASC500.formulas.airfoil.compute_LE_radius import leading_edge_radius
from DASC500.formulas.airfoil.compute_TE_angle import trailing_edge_angle
from DASC500.formulas.airfoil.compute_thickness_camber import compute_thickness_camber
from DASC500.formulas.airfoil.compute_span import calculate_span
from DASC500.formulas.airfoil.compute_thickness_to_chord_ratio import thickness_to_chord_ratio

from DASC500.xfoil.fix_airfoil_data import *

from DASC500.classes.XFoilRunner import XFoilRunner

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
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS aero_coeffs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                reynolds_number REAL NOT NULL,
                mach REAL NOT NULL,
                alpha REAL NOT NULL,
                cl REAL,
                cd REAL,
                cm REAL,
                FOREIGN KEY (name) REFERENCES airfoils(name),
                UNIQUE (name, reynolds_number, mach, alpha)
            )
        """)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS airfoil_geometry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                max_thickness REAL,
                max_camber REAL,
                chord_length REAL,
                span REAL,
                aspect_ratio REAL,
                leading_edge_radius REAL,
                trailing_edge_angle REAL,
                thickness_to_chord_ratio REAL,
                thickness_distribution TEXT,
                camber_distribution TEXT,
                normalized_chord TEXT,
                FOREIGN KEY (name) REFERENCES airfoils(name)
            )
        """)
        self.conn.commit()
    
    def create_aero_coeffs_table(self):
        """Creates the table for storing aerodynamic coefficients if it doesn't exist."""
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS aero_coeffs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                reynolds_number REAL NOT NULL,
                mach REAL NOT NULL,
                alpha REAL NOT NULL,
                cl REAL,
                cd REAL,
                cm REAL,
                FOREIGN KEY (name) REFERENCES airfoils(name),
                UNIQUE (name, reynolds_number, mach, alpha)  -- Ensure unique combinations
            )
        """)
        self.conn.commit()
    
    def create_geometry_table(self):
        # with sqlite3.connect(self.db_path) as conn:
        #     cursor = conn.cursor()
        #     cursor.execute("""
        #         CREATE TABLE IF NOT EXISTS airfoil_geometry (
        #             id INTEGER PRIMARY KEY AUTOINCREMENT,
        #             name TEXT UNIQUE NOT NULL,
        #             max_thickness REAL,
        #             max_camber REAL,
        #             chord_length REAL,
        #             span REAL,
        #             aspect_ratio REAL,
        #             leading_edge_radius REAL,
        #             trailing_edge_angle REAL,
        #             thickness_to_chord_ratio REAL,
        #             FOREIGN KEY (name) REFERENCES airfoils(name)
        #         )
        #     """)
        #     conn.commit()
            # try:
            #     cursor.execute("ALTER TABLE airfoil_geometry ADD COLUMN thickness_distribution TEXT")
            #     print("Added column thickness_distribution")
            # except sqlite3.OperationalError:
            #     print("Column thickness_distribution already exists")

            # try:
            #     cursor.execute("ALTER TABLE airfoil_geometry ADD COLUMN camber_distribution TEXT")
            #     print("Added column camber_distribution")
            # except sqlite3.OperationalError:
            #     print("Column camber_distribution already exists")

            # try:
            #     cursor.execute("ALTER TABLE airfoil_geometry ADD COLUMN normalized_chord TEXT")
            #     print("Added column normalized_chord")
            # except sqlite3.OperationalError:
            #     print("Column normalized_chord already exists")
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS airfoil_geometry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            max_thickness REAL,
            max_camber REAL,
            chord_length REAL,
            span REAL,
            aspect_ratio REAL,
            leading_edge_radius REAL,
            trailing_edge_angle REAL,
            thickness_to_chord_ratio REAL,
            thickness_distribution TEXT,
            camber_distribution TEXT,
            normalized_chord TEXT,
            FOREIGN KEY (name) REFERENCES airfoils(name)
            )
        """)
        try:
            self.cursor.execute("ALTER TABLE airfoil_geometry ADD COLUMN thickness_distribution TEXT")
            print("Added column thickness_distribution")
        except sqlite3.OperationalError:
            print("Column thickness_distribution already exists")

        try:
            self.cursor.execute("ALTER TABLE airfoil_geometry ADD COLUMN camber_distribution TEXT")
            print("Added column camber_distribution")
        except sqlite3.OperationalError:
            print("Column camber_distribution already exists")

        try:
            self.cursor.execute("ALTER TABLE airfoil_geometry ADD COLUMN normalized_chord TEXT")
            print("Added column normalized_chord")
        except sqlite3.OperationalError:
            print("Column normalized_chord already exists")
        
        self.conn.commit()
            
        
    def store_airfoil_data(self, name, description, pointcloud, overwrite=False):
        try:
            if overwrite:
                self.cursor.execute("REPLACE INTO airfoils (name, description, pointcloud) VALUES (?, ?, ?)", (name, description, pointcloud))
            else:
                self.cursor.execute("INSERT INTO airfoils (name, description, pointcloud) VALUES (?, ?, ?)", (name, description, pointcloud))
            self.conn.commit()
            print(f"Stored: {name} in database.")
        except sqlite3.IntegrityError:
            if overwrite:
                print(f"Updated: {name} in database.") # If it exists and overwrite is True, it should update it.
            else:
                print(f"Airfoil {name} already exists in the database. Use overwrite=True to update.")

    def store_aero_coeffs(self, name, reynolds_number, mach, alpha, cl, cd, cm):
        """Stores aerodynamic coefficients for a given airfoil, Reynolds number, Mach number, and AoA.
        
        Ensures values are non-empty and valid before inserting.
        """
        # Validate inputs
        if None in [name, reynolds_number, mach, alpha, cl, cd, cm]:
            print(f"Skipping entry for {name} (Re={reynolds_number}, Mach={mach}, alpha={alpha}) due to missing values.")
            return
        
        if any(math.isnan(x) for x in [cl, cd, cm]):
            print(f"Skipping entry for {name} (Re={reynolds_number}, Mach={mach}, alpha={alpha}) due to NaN values.")
            return

        try:
            self.cursor.execute("""
                INSERT OR REPLACE INTO aero_coeffs (name, reynolds_number, mach, alpha, cl, cd, cm)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (name, reynolds_number, mach, alpha, cl, cd, cm))  
            self.conn.commit()
            print(f"Stored aero coeffs for {name} (Re={reynolds_number}, Mach={mach}, alpha={alpha})")
        except Exception as e:
            print(f"Error storing aero coeffs for {name}: {e}")
            self.conn.rollback()
    
    def cleanup_all_pointclouds(self):
        """Cleans up the point clouds of all airfoils in the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name, pointcloud FROM airfoils")
            airfoils = cursor.fetchall()

            for name, pointcloud_str in airfoils:
                rows = pointcloud_str.split('\n')
                rows = [x for x in rows if x.strip()]
                pointcloud_np = np.array([np.fromstring(row, dtype=float, sep=' ') for row in rows])
                pointcloud_np = reorder_airfoil_data(pointcloud_np)
                pointcloud_reorder = ""
                for row in pointcloud_np:
                    pointcloud_reorder += " ".join(f"{val:.6f}" for val in row) + "\n"

                # Update the database with the cleaned point cloud
                cursor.execute("UPDATE airfoils SET pointcloud = ? WHERE name = ?", (pointcloud_reorder.strip(), name))
                conn.commit()
                print(f"Cleaned point cloud for {name}")

    def get_airfoil_data(self, name):
        self.cursor.execute("SELECT description, pointcloud FROM airfoils WHERE name=?", (name,))
        return self.cursor.fetchone()
    
    def plot_airfoil(self, name):
        """Plots the airfoil using its pointcloud data."""
        data = self.get_airfoil_data(name)
        if data:
            description, pointcloud_str = data
            try:
                # Parse the pointcloud string into x and y coordinates
                points = [line.split() for line in pointcloud_str.strip().split('\n')]
                x = [float(p[0]) for p in points if len(p) == 2]  # Handle potential errors
                y = [float(p[1]) for p in points if len(p) == 2]

                if x and y: # Make sure x and y are not empty before plotting
                    plt.figure(figsize=(8, 6))  # Adjust figure size as needed
                    plt.plot(x, y)
                    plt.xlabel("X Coordinate")
                    plt.ylabel("Y Coordinate")
                    plt.title(f"Airfoil: {name}")
                    plt.grid(True)
                    plt.axis('equal') # Important for airfoil plots!
                    plt.show()
                else:
                    print(f"No valid point cloud data found for {name}")

            except (ValueError, IndexError) as e:
                print(f"Error parsing point cloud data for {name}: {e}")
        else:
            print(f"Airfoil {name} not found in the database.")
    
    def compute_geometry_metrics(self):
        # with sqlite3.connect(self.db_path) as conn:
        #     cursor = conn.cursor()
        #     cursor.execute("SELECT name, pointcloud FROM airfoils")
        #     airfoils = cursor.fetchall()

        #     for name, pointcloud in airfoils:
        #         rows = pointcloud.split('\n')
        #         rows = [x for x in rows if x.strip()]
        #         points = np.array([np.fromstring(row, dtype=float, sep=' ') for row in rows])
        #         points = reorder_airfoil_data(points)

        #         x_coords, thickness, camber = compute_thickness_camber(points)
        #         LE_radius = leading_edge_radius(points)
        #         TE_angle = trailing_edge_angle(points)
        #         chord_length = max(x_coords) - min(x_coords)
        #         t_to_c = thickness_to_chord_ratio(thickness, chord_length)
        #         span = calculate_span(points)
        #         aspect_ratio = aspect_ratio(span, chord_length)
        #         max_thickness = max(thickness)
        #         max_camber = max(camber)
                
                # Calculate normalized chord
                # normalized_chord = np.linspace(0, 1, len(thickness))

                # # Store distribution data as comma-separated strings
                # thickness_dist_str = ",".join(map(str, thickness))
                # camber_dist_str = ",".join(map(str, camber))
                # normalized_chord_str = ",".join(map(str, normalized_chord))

        #         cursor.execute("""
            #     INSERT OR REPLACE INTO airfoil_geometry (name, max_thickness, max_camber, leading_edge_radius, trailing_edge_angle, chord_length, thickness_to_chord_ratio, total_length, thickness_distribution, camber_distribution, normalized_chord)
            #     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            # """, (name, max_thickness, max_camber, LE_radius, TE_angle, chord_length, t_to_c, span, thickness_dist_str, camber_dist_str, normalized_chord_str))
            # conn.commit()
        #         print(f"Geometry metrics computed and stored for {name}")
        
        self.cursor.execute("SELECT name, pointcloud FROM airfoils")
        airfoils = self.cursor.fetchall()
        
        for name, pointcloud in airfoils:
            rows = pointcloud.split('\n')
            rows = [x for x in rows if x.strip()]
            points = np.array([np.fromstring(row, dtype=float, sep=' ') for row in rows])
            points = reorder_airfoil_data(points)
            
            x_coords, thickness, camber = compute_thickness_camber(points)
            LE_radius = leading_edge_radius(points)
            TE_angle = trailing_edge_angle(points)
            chord_length = max(x_coords) - min(x_coords)
            t_to_c = thickness_to_chord_ratio(thickness, chord_length)
            span = calculate_span(points)
            aspect_ratio = calculate_aspect_ratio(span, chord_length)
            max_thickness = max(thickness)
            max_camber = max(camber)
            
            # Calculate normalized chord
            normalized_chord = np.linspace(0, 1, len(thickness))

            # Store distribution data as comma-separated strings
            thickness_dist_str = ",".join(map(str, thickness))
            camber_dist_str = ",".join(map(str, camber))
            normalized_chord_str = ",".join(map(str, normalized_chord))

            self.cursor.execute("""
                INSERT OR REPLACE INTO airfoil_geometry (name, max_thickness, max_camber, leading_edge_radius, trailing_edge_angle, chord_length, thickness_to_chord_ratio, span, thickness_distribution, camber_distribution, normalized_chord)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (name, max_thickness, max_camber, LE_radius, TE_angle, chord_length, t_to_c, span, thickness_dist_str, camber_dist_str, normalized_chord_str))
            self.conn.commit()
            print(f"Geometry metrics computed and stored for {name}")
    
    def find_airfoils_by_geometry(self, parameter, target_value, tolerance, tolerance_type="absolute"):
        """
        Finds airfoils based on a specified geometric parameter, target value, and tolerance.

        Args:
            parameter (str): The geometric parameter to search for (e.g., "max_thickness", "chord_length").
            target_value (float): The target value for the parameter.
            tolerance (float): The tolerance for the search.
            tolerance_type (str): "absolute" or "percentage".
        """
        valid_parameters = ["max_thickness", "max_camber", "leading_edge_radius",
                            "trailing_edge_angle", "chord_length", "thickness_to_chord_ratio", "total_length"]

        if parameter not in valid_parameters:
            print(f"Invalid parameter. Choose from: {', '.join(valid_parameters)}")
            return []

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            if tolerance_type == "absolute":
                lower_bound = target_value - tolerance
                upper_bound = target_value + tolerance
            elif tolerance_type == "percentage":
                lower_bound = target_value * (1 - tolerance / 100.0)
                upper_bound = target_value * (1 + tolerance / 100.0)
            else:
                print("Invalid tolerance_type. Choose 'absolute' or 'percentage'.")
                return []

            query = f"SELECT name FROM airfoil_geometry WHERE {parameter} BETWEEN ? AND ?"
            cursor.execute(query, (lower_bound, upper_bound))
            results = cursor.fetchall()

            airfoil_names = [row[0] for row in results]
            if airfoil_names:
                print(f"Airfoils matching {parameter} = {target_value} ({tolerance} {tolerance_type}):")
                for name in airfoil_names:
                    print(f"- {name}")
                return airfoil_names
            else:
                print(f"No airfoils found matching {parameter} = {target_value} ({tolerance} {tolerance_type}).")
                return []
    
    def plot_leading_edge_radius(self, parameter="chord_length"):
        """Plots leading-edge radius against a specified parameter."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT name, leading_edge_radius, {parameter} FROM airfoil_geometry")
            results = cursor.fetchall()

        if results:
            names, radii, params = zip(*results)
            plt.figure(figsize=(8, 6))
            plt.scatter(params, radii)
            plt.xlabel(parameter)
            plt.ylabel("Leading Edge Radius")
            plt.title("Leading Edge Radius vs. " + parameter)
            plt.grid(True)
            plt.show()

    def plot_trailing_edge_angle(self, parameter="chord_length"):
        """Plots trailing-edge angle against a specified parameter."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT name, trailing_edge_angle, {parameter} FROM airfoil_geometry")
            results = cursor.fetchall()

        if results:
            names, angles, params = zip(*results)
            plt.figure(figsize=(8, 6))
            plt.scatter(params, angles)
            plt.xlabel(parameter)
            plt.ylabel("Trailing Edge Angle")
            plt.title("Trailing Edge Angle vs. " + parameter)
            plt.grid(True)
            plt.show()

    def plot_geometry_correlations(self):
        """Plots correlations between geometric parameters using a heatmap."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT max_thickness, max_camber, leading_edge_radius, trailing_edge_angle, chord_length, thickness_to_chord_ratio, span, aspect_ratio FROM airfoil_geometry")
            results = cursor.fetchall()

        if results:
            df = pd.DataFrame(results, columns=["max_thickness", "max_camber", "leading_edge_radius", "trailing_edge_angle", "chord_length", "thickness_to_chord_ratio", "span", "aspect_ratio"])
            correlation_matrix = df.corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
            plt.title("Geometric Parameter Correlations")
            plt.show()
    
    def run_all_airfoils(self, 
                         xfoil_runner, 
                         max_workers=4, 
                         reynolds_list=[10000, 50000],
                         mach_list=[0.1, 0.2],
                         **varargin):
        """
        Runs all airfoils in parallel through XFoil for multiple Mach, Reynolds numbers, and AoA values.
        
        Uses multiprocessing for faster execution.

        Args:
            xfoil_runner (XFoilRunner): Instance of XFoilRunner to execute simulations.
            mach_list (list): List of Mach numbers to simulate.
            reynolds_list (list): List of Reynolds numbers.
            alpha_range (tuple): (start_alpha, end_alpha, alpha_increment)
            max_workers (int): Number of parallel processes to run.
        """
        self.cursor.execute("SELECT name, pointcloud FROM airfoils")
        airfoils = self.cursor.fetchall()

        tasks = []
        for name, pointcloud in airfoils:
            for Re in reynolds_list:
                for Mach in mach_list:
                    tasks.append((name, pointcloud, Re, Mach))

        def run_task(task):
            name, pointcloud, Re, Mach = task
            print(f"Running {name} at Re={Re}, Mach={Mach}...")
            polar_df = xfoil_runner.run_xfoil(name, pointcloud, Re=Re, Mach=Mach, **varargin)

            if polar_df is None or polar_df.empty:
                print(f"Skipping {name} at Re={Re}, Mach={Mach} due to empty results.")
                return
            
            for _, row in polar_df.iterrows():
                self.store_aero_coeffs(name, Re, Mach, row["alpha"], row["cl"], row["cd"], row["cm"])

        # Use multiprocessing to run simulations in parallel
        """with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            executor.map(run_task, tasks)"""
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(run_task, tasks)

        print("Finished processing all airfoils.")
    
    def get_aero_coeffs(self, name):
        """Returns all aerodynamic coefficients for a given airfoil."""
        self.cursor.execute("SELECT reynolds_number, mach, alpha, cl, cd, cm FROM aero_coeffs WHERE name=?", (name,))
        rows = self.cursor.fetchall()
        return pd.DataFrame(rows, columns=["Re", "Mach", "Alpha", "Cl", "Cd", "Cm"]) if rows else None

    def get_aero_coeffs_by_conditions(self, name, Re, Mach):
        """Returns aerodynamic coefficients for a given airfoil at a specific Re and Mach."""
        self.cursor.execute("""
            SELECT alpha, cl, cd, cm FROM aero_coeffs 
            WHERE name=? AND reynolds_number=? AND mach=?
        """, (name, Re, Mach))
        rows = self.cursor.fetchall()
        return pd.DataFrame(rows, columns=["Alpha", "Cl", "Cd", "Cm"]) if rows else None

    def plot_polar(self, name, Re, Mach):
        """Plots Cl vs. Cd (Lift-Drag Polar) for the given airfoil."""
        df = self.get_aero_coeffs_by_conditions(name, Re, Mach)
        if df is None or df.empty:
            print(f"No aerodynamic data found for {name} (Re={Re}, Mach={Mach})")
            return
        
        plt.figure(figsize=(8, 6))
        plt.plot(df["Cd"], df["Cl"], marker='o', linestyle='-')
        plt.xlabel("Cd (Drag Coefficient)")
        plt.ylabel("Cl (Lift Coefficient)")
        plt.title(f"Lift-Drag Polar for {name} (Re={Re}, Mach={Mach})")
        plt.grid()
        plt.show()

    def plot_coeff_vs_alpha(self, name, coeff="Cl", Re=None, Mach=None):
        """Plots Cl, Cd, or Cm vs. Alpha for a given airfoil at specified conditions."""
        if coeff not in ["Cl", "Cd", "Cm"]:
            print("Invalid coefficient. Choose 'Cl', 'Cd', or 'Cm'.")
            return

        df = self.get_aero_coeffs_by_conditions(name, Re, Mach)
        if df is None or df.empty:
            print(f"No aerodynamic data found for {name} (Re={Re}, Mach={Mach})")
            return
        
        plt.figure(figsize=(8, 6))
        plt.plot(df["Alpha"], df[coeff], marker='o', linestyle='-')
        plt.xlabel("Angle of Attack (α)")
        plt.ylabel(coeff)
        plt.title(f"{coeff} vs. Alpha for {name} (Re={Re}, Mach={Mach})")
        plt.grid()
        plt.show()

    def clear_database(self):
        self.cursor.execute("DELETE FROM airfoils")
        self.conn.commit()
        print("Database cleared.")

    def close(self):
        self.conn.close()


if __name__ == "__main__":
    # Example usage of the AirfoilDatabase class:
    """airfoil_db = AirfoilDatabase(db_dir="my_airfoil_database") # Same directory
    airfoil_name = "bacnlf"
    xfoil = XFoilRunner("D:/Mitchell/software/CFD/xfoil.exe")
    data = airfoil_db.get_airfoil_data(airfoil_name)
    if data:
        description, pointcloud = data
        print(f"Description: {description}")
        print(f"Pointcloud: {pointcloud}") # Pointcloud can be very long.
        #data = airfoil_db.plot_airfoil(airfoil_name)
        data = xfoil.run_xfoil(os.path.join(r"D:\Mitchell\School\2025 Winter\DASC500\github\DASC500\airfoil_data", airfoil_name), pointcloud)
        #print(data)
    airfoil_db.close()"""
    
    db = AirfoilDatabase(db_dir="my_airfoil_database")
    #db.create_geometry_table()
    #db.compute_geometry_metrics()
    print(db.find_airfoils_by_geometry('thickness_to_chord_ratio', 0.2, 0.05))
    
    
    """xfoil = XFoilRunner("D:/Mitchell/software/CFD/xfoil.exe")

    mach_list = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # Define Mach numbers
    reynolds_list = [1000, 5000, 10000, 25000, 50000, 100000, 150000, 200000, 300000, 500000, 750000, 1000000, 1500000, 2000000]  # Define Reynolds numbers
    alpha_range = (-5, 15, 1)  # Define angle of attack range

    db.run_all_airfoils(xfoil, max_workers=12, mach_list=mach_list, reynolds_list=reynolds_list, alpha_start=-20, alpha_end=20, alpha_increment=4)"""
    db.close()
    

    
    
