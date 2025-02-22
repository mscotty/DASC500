import sqlite3


class XFoilDatabase:
    def __init__(self, database_path="xfoil_results.db"):
        self.database_path = database_path
        self._initialize_database()

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
    
    def store_results(self, airfoil, mach, reynolds, alpha, cl, cd, cm):
        if None in (cl, cd, cm):
            print(f"Skipping storage for {airfoil}, invalid aerodynamic coefficients.")
            return
        
        with sqlite3.connect(self.database_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO results (airfoil, mach, reynolds, alpha, cl, cd, cm)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (airfoil, mach, reynolds, alpha, cl, cd, cm))
            conn.commit()
        print(f"Stored results for {airfoil}: CL={cl}, CD={cd}, CM={cm}")
    
    def get_results(self, airfoil, reynolds=None, mach=None):
        with sqlite3.connect(self.database_path) as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM results WHERE airfoil = ?"
            params = [airfoil]
            if reynolds is not None:
                query += " AND reynolds = ?"
                params.append(reynolds)
            if mach is not None:
                query += " AND mach = ?"
                params.append(mach)
            cursor.execute(query, tuple(params))
            return cursor.fetchall()
    
    def clear_results(self):
        with sqlite3.connect(self.database_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM results")
            conn.commit()
        print("Cleared all stored XFoil results.")
        
