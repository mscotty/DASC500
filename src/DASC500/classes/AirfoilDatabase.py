import os
import sqlite3
import matplotlib.pyplot as plt
import concurrent.futures
import threading

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
                    pointcloud TEXT
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
            conn.commit()
        
    def store_airfoil_data(self, name, description, pointcloud, overwrite=False):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                if overwrite:
                    cursor.execute("REPLACE INTO airfoils (name, description, pointcloud) VALUES (?, ?, ?)", (name, description, pointcloud))
                else:
                    cursor.execute("INSERT INTO airfoils (name, description, pointcloud) VALUES (?, ?, ?)", (name, description, pointcloud))
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
    xfoil = XFoilRunner("D:/Mitchell/software/CFD/xfoil.exe")

    #mach_list = [0.2]  # Define Mach numbers
    #reynolds_list = [10000]  # Define Reynolds numbers
    alpha_start=0
    alpha_end=8
    alpha_increment=4
    mach_list = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # Define Mach numbers
    reynolds_list = [1000, 5000, 10000, 25000, 50000, 100000, 150000, 200000, 300000, 500000, 750000, 1000000, 1500000, 2000000]  # Define Reynolds numbers
    #alpha_range = (-5, 15, 1)  # Define angle of attack range

    db.run_all_airfoils(xfoil, max_workers=1, mach_list=mach_list, reynolds_list=reynolds_list, alpha_start=-20, alpha_end=20, alpha_increment=4)
    # db.close()
    

    
    
