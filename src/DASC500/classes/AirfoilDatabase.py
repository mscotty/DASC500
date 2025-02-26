import os
import sqlite3
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import concurrent.futures
import threading
import seaborn as sns
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon

from DASC500.plotting.plot_histogram import plot_histogram

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
        # self.conn = sqlite3.connect(self.db_path)
        # self.cursor = self.conn.cursor()
        self.write_lock = threading.Lock() #Add the lock.
        self._enable_wal()
        self._create_table()
    
    def _enable_wal(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")

    def _create_table(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS airfoils (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    description TEXT,
                    pointcloud TEXT,
                    airfoil_series TEXT,
                    source TEXT
                )
            """)
            cursor.execute("""
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
            cursor.execute("""
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
            conn.commit()
        
    def store_airfoil_data(self, name, description, pointcloud, airfoil_series, source, overwrite=False):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                if overwrite:
                    cursor.execute("REPLACE INTO airfoils (name, description, pointcloud, airfoil_series, source) VALUES (?, ?, ?, ?, ?)", (name, description, pointcloud, airfoil_series.value, source))
                else:
                    cursor.execute("INSERT INTO airfoils (name, description, pointcloud, airfoil_series, source) VALUES (?, ?, ?, ?, ?)", (name, description, pointcloud, airfoil_series.value, source))
                conn.commit()
                print(f"Stored: {name} in database.")
        except sqlite3.IntegrityError:
            if overwrite:
                print(f"Updated: {name} in database.")
            else:
                print(f"Airfoil {name} already exists in the database. Use overwrite=True to update.")

    def get_airfoil_data(self, name):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT description, pointcloud FROM airfoils WHERE name=?", (name,))
            return cursor.fetchone()
    
    def get_airfoil_dataframe(self):
        """Returns a Pandas DataFrame with airfoil names, series, and number of points."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name, airfoil_series, pointcloud FROM airfoils")
            results = cursor.fetchall()

        data = []
        for row in results:
            name, series, pointcloud = row
            num_points = len(pointcloud.strip().split('\n')) if pointcloud else 0
            data.append({
                'Name': name,
                'Series': series,
                'Num_Points': num_points
            })

        return pd.DataFrame(data)

    def get_airfoil_geometry_dataframe(self):
        """Retrieves airfoil geometry data from the database and returns it as a Pandas DataFrame."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT 
                        id, name, max_thickness, max_camber, chord_length, span, 
                        aspect_ratio, leading_edge_radius, trailing_edge_angle, 
                        thickness_to_chord_ratio 
                    FROM airfoil_geometry
                """)
                results = cursor.fetchall()

            if not results:
                return pd.DataFrame()  # Return an empty DataFrame if no data found

            column_names = [description[0] for description in cursor.description]
            df = pd.DataFrame(results, columns=column_names)
            return df

        except sqlite3.Error as e:
            print(f"SQLite error: {e}")
            return pd.DataFrame() #Return empty dataframe on error.
        except Exception as e:
            print(f"An error occurred: {e}")
            return pd.DataFrame() #Return empty dataframe on error.
    
    # def cleanup_all_pointclouds(self):
    #     """Cleans up the point clouds of all airfoils in the database."""
    #     with sqlite3.connect(self.db_path) as conn:
    #         cursor = conn.cursor()
    #         cursor.execute("SELECT name, pointcloud FROM airfoils")
    #         airfoils = cursor.fetchall()

    #         for name, pointcloud_str in airfoils:
    #             """rows = pointcloud_str.split('\n')
    #             rows = [x for x in rows if x.strip()]
    #             pointcloud_np = np.array([np.fromstring(row, dtype=float, sep=' ') for row in rows])"""
    #             pointcloud_np = self._pointcloud_to_numpy(pointcloud_str)
    #             pointcloud_np = reorder_airfoil_data(pointcloud_np)
    #             pointcloud_reorder = ""
    #             for row in pointcloud_np:
    #                 pointcloud_reorder += " ".join(f"{val:.6f}" for val in row) + "\n"

    #             # Update the database with the cleaned point cloud
    #             cursor.execute("UPDATE airfoils SET pointcloud = ? WHERE name = ?", (pointcloud_reorder.strip(), name))
    #             conn.commit()
    #             print(f"Cleaned point cloud for {name}")
    
    def _pointcloud_to_numpy(self, pointcloud_str):
        """Converts a pointcloud string to a NumPy array."""
        if not pointcloud_str:
            return np.array([])
        rows = pointcloud_str.strip().split('\n')
        rows = [x.strip() for x in rows if x.strip()]
        try:
            return np.array([np.fromstring(row, dtype=float, sep=' ') for row in rows])
        except ValueError:
            return np.array([])
    
    def check_pointcloud_outliers(self, name, threshold=3.0):
        """Checks for outliers in the pointcloud of a given airfoil."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT pointcloud FROM airfoils WHERE name = ?", (name,))
            result = cursor.fetchone()

            if result and result[0]:
                pointcloud_np = self._pointcloud_to_numpy(result[0])
                if pointcloud_np.size == 0:
                    return False, "Empty pointcloud"
                x = pointcloud_np[:, 0]
                y = pointcloud_np[:, 1]

                # Calculate z-scores for x and y coordinates
                z_x = np.abs((x - np.mean(x)) / np.std(x))
                z_y = np.abs((y - np.mean(y)) / np.std(y))

                # Identify outliers based on z-score threshold
                outlier_indices = np.where((z_x > threshold) | (z_y > threshold))[0]

                if len(outlier_indices) > 0:
                    outlier_points = pointcloud_np[outlier_indices]
                    return True, outlier_points
                else:
                    return False, None
            else:
                return False, "Airfoil not found"

    def check_all_pointcloud_outliers(self, threshold=3.0):
        """Checks all airfoils in the database for outliers."""
        outliers_found = {}

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM airfoils")
            airfoils = cursor.fetchall()

            for airfoil in airfoils:
                name = airfoil[0]
                has_outliers, outliers = self.check_pointcloud_outliers(name, threshold)
                if has_outliers:
                    outliers_found[name] = outliers

        if outliers_found:
            print("Airfoils with outliers:")
            for name, outliers in outliers_found.items():
                print(f"- {name}:")
                for point in outliers:
                    print(f"  {point}")
        else:
            print("No outliers found in any airfoils.")
        return outliers_found

    def check_self_intersection(self, pointcloud_np):
        """Checks if a pointcloud (NumPy array) has self-intersections."""
        if len(pointcloud_np) < 3:
            return False

        try:
            polygon = Polygon(pointcloud_np)
            return not polygon.is_valid or not polygon.is_simple
        except Exception as e:
            print(f"Shapely Error: {e}")
            return True

        except Exception as e:
            print(f"ConvexHull Error: {e}")
            return True

    def check_self_closing(self, pointcloud_np):
        """Checks if a pointcloud (NumPy array) is self-closing."""
        if len(pointcloud_np) < 3:
            return False  # Not enough points for a closed shape
        return np.allclose(pointcloud_np[0], pointcloud_np[-1])

    def check_airfoil_validity(self):
        """Checks all airfoils in the database for self-intersections and self-closing."""
        invalid_intersections = []
        non_closing = []

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name, pointcloud FROM airfoils")
            airfoils = cursor.fetchall()

            for name, pointcloud_str in airfoils:
                pointcloud_np = self._pointcloud_to_numpy(pointcloud_str)
                if pointcloud_np.size > 0:
                    if self.check_self_intersection(pointcloud_np):
                        invalid_intersections.append(name)
                    if not self.check_self_closing(pointcloud_np):
                        non_closing.append(name)

        if invalid_intersections:
            print("Airfoils with self-intersections:")
            for name in invalid_intersections:
                print(f"- {name}")
            print(f"Total Number of Airfoils that failed: {len(invalid_intersections)}")
        else:
            print("No self-intersections found.")

        if non_closing:
            print("\nAirfoils that are not self-closing:")
            for name in non_closing:
                print(f"- {name}")
            print(f"Total Number of Airfoils that failed: {len(non_closing)}")
        else:
            print("\nAll airfoils are self-closing.")

    def fix_all_airfoils(self):
        """Reorders and closes all airfoils in the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name, pointcloud FROM airfoils")
            airfoils = cursor.fetchall()
            for name, pointcloud in airfoils:
                pointcloud_np = self._pointcloud_to_numpy(pointcloud)
                self_intersection = self.check_self_intersection(pointcloud_np)
                self_closing = self.check_self_closing(pointcloud_np)
                if self_intersection:
                    pointcloud_np = reorder_airfoil_data(pointcloud_np)
                    print(f"Reordered {name}")
                if not self_closing:
                    pointcloud_np = close_airfoil_data(pointcloud_np)
                    print(f"Closed {name}")
                pointcloud_str = '\n'.join([' '.join(map(str, row)) for row in pointcloud_np])
                if self_intersection or not self_closing:
                    cursor.execute("UPDATE airfoils SET pointcloud = ? WHERE name = ?", (pointcloud_str, name))
                    conn.commit()
    
    def plot_airfoil_series_pie(self, output_dir=None, output_name=None):
        """Fetches airfoil series data from the database and plots a pie chart."""

        # Connect to database and retrieve all airfoil_series values
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT airfoil_series FROM airfoils")
            series_list = [row[0] for row in cursor.fetchall() if row[0]]

        if not series_list:
            print("No airfoil series data found in the database.")
            return

        # Count occurrences of each airfoil series
        series_counts = Counter(series_list)

        # Extract labels and counts for pie chart
        labels = list(series_counts.keys())
        counts = list(series_counts.values())

        # Plot pie chart
        plt.figure(figsize=(8, 8))  # Adjust figure size as needed
        plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140)
        plt.title("Airfoil Series Distribution")

        # Save the pie chart
        if output_dir is not None or output_name is not None:
            plt.savefig(os.path.join(output_dir if output_dir is not None else '', output_name if output_name is not None else 'airfoil_series_pie.png'))
        else:
            plt.show() # Display the pie chart

    def plot_airfoil(self, 
                     name, 
                     ax=None,
                     output_dir=None, 
                     output_name=None):
        """Plots the airfoil using its point cloud data, with markers for individual points."""
        data = self.get_airfoil_data(name)
        if data:
            description, pointcloud_str = data
            pointcloud_np = self._pointcloud_to_numpy(pointcloud_str)
            x = pointcloud_np[:,0]
            y = pointcloud_np[:,1]

            if ax is None:
                fig, ax = plt.subplots()

            ax.plot(x, y, label=f"{name} : {len(x)}", linestyle='-', marker='o', markersize=4)
            ax.set_title(f"Airfoil: {name}")
            ax.set_xlabel("X Coordinate")
            ax.set_ylabel("Y Coordinate")
            ax.grid(True)
            ax.axis("equal")
            ax.legend(loc='upper left')
            if ax is None:
                if output_dir is None:
                    plt.show()
                else:
                    if output_name is None:
                        output_name = name + '.png'
                    plt.savefig(os.path.join(output_dir, output_name))
            else:
                return ax
        else:
            print(f"Airfoil {name} not found in the database.")
    
    def plot_multiple_airfoils(self, 
                               names, 
                               output_dir=None, 
                               output_name=None):
        """Plots multiple airfoils on the same figure."""
        plt.figure(figsize=(10, 7))

        for name in names:
            data = self.get_airfoil_data(name)
            if data:
                description, pointcloud_str = data
                try:
                    points = [line.split() for line in pointcloud_str.strip().split('\n')]
                    x = [float(p[0]) for p in points if len(p) == 2]
                    y = [float(p[1]) for p in points if len(p) == 2]

                    if x and y:
                        plt.plot(x, y, label=name, linestyle='-', marker='o', markersize=3)  # Markers added
                    else:
                        print(f"No valid point cloud data found for {name}")

                except (ValueError, IndexError) as e:
                    print(f"Error parsing point cloud data for {name}: {e}")
            else:
                print(f"Airfoil {name} not found in the database.")

        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.title("Airfoil Comparison")
        plt.grid(True)
        plt.axis('equal')
        plt.legend()
        if output_dir is None:
            plt.show()
        else:
            if output_name is None:
                output_name = ' vs '.join(names) + '.png'
            plt.savefig(os.path.join(output_dir, output_name))
    
    def compute_geometry_metrics(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name, pointcloud FROM airfoils")
            airfoils = cursor.fetchall()

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

                cursor.execute("""
                    INSERT OR REPLACE INTO airfoil_geometry (name, max_thickness, max_camber, leading_edge_radius, trailing_edge_angle, chord_length, thickness_to_chord_ratio, span, aspect_ratio, thickness_distribution, camber_distribution, normalized_chord)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (name, max_thickness, max_camber, LE_radius, TE_angle, chord_length, t_to_c, span, aspect_ratio, thickness_dist_str, camber_dist_str, normalized_chord_str))
                conn.commit()
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
    
    def store_aero_coeffs(self, name, reynolds_number, mach, alpha, cl, cd, cm):
        try:
            with self.write_lock: #Acquire the Lock
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT OR IGNORE INTO aero_coeffs (name, reynolds_number, mach, alpha, cl, cd, cm)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (name, reynolds_number, mach, alpha, cl, cd, cm))
                    conn.commit()
                    print(f"Stored aero coeffs for {name} (Re={reynolds_number}, Mach={mach}, alpha={alpha})")
        except sqlite3.Error as e:
            print(f"SQLite error storing aero coeffs for {name}: {e}")
            print(f"Error details: {e.__class__.__name__}, {e}")
    
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
        def run_task(task):
            name, pointcloud, Re, Mach = task
            polar_df = xfoil_runner.run_xfoil(name, pointcloud, Re=Re, Mach=Mach, **varargin)
            if polar_df is not None and not polar_df.empty:
                for _, row in polar_df.iterrows():
                    self.store_aero_coeffs(name, Re, Mach, row["alpha"], row["cl"], row["cd"], row["cm"])

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            tasks = []
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name, pointcloud FROM airfoils")
                airfoils = cursor.fetchall()
            for name, pointcloud in airfoils:
                for Re in reynolds_list:
                    for Mach in mach_list:
                        tasks.append((name, pointcloud, Re, Mach))
            executor.map(run_task, tasks)

        print("Finished processing all airfoils.")
    
    def get_aero_coeffs(self, name, Re=None, Mach=None):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM aero_coeffs WHERE name = ?"
            params = [name]
            if Re is not None:
                query += " AND reynolds_number = ?"
                params.append(Re)
            if Mach is not None:
                query += " AND mach = ?"
                params.append(Mach)
            cursor.execute(query, tuple(params))
            return cursor.fetchall()

    def plot_polar(self, name, Re, Mach):
        df = self.get_aero_coeffs(name, Re, Mach)
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
        if coeff not in ["Cl", "Cd", "Cm"]:
            print("Invalid coefficient. Choose 'Cl', 'Cd', or 'Cm'.")
            return

        df = self.get_aero_coeffs(name, Re, Mach)
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
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM airfoils")
            cursor.execute("DELETE FROM aero_coeffs")
            conn.commit()
            print("Database cleared.")

    def close(self):
        # We are not opening the connection in init anymore.
        pass


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
    db.compute_geometry_metrics()
    # db.check_airfoil_validity()
    # db.fix_all_airfoils()
    # db.check_airfoil_validity()
    db.close()
    

    
    
