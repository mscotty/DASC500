from DASC500.classes.AirfoilDatabase import AirfoilDatabase



db = AirfoilDatabase(db_dir="my_airfoil_database")
df = db.get_aero_coeffs('aquilasm')
print(df)