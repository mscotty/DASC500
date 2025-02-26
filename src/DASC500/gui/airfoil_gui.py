import sys
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QListWidget, QLabel, QGridLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from DASC500.classes.AirfoilDatabase import AirfoilDatabase

class AirfoilViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.airfoil_db_instance = AirfoilDatabase(db_dir="my_airfoil_database")
        self.initUI()
        self.load_airfoil_names()

    def initUI(self):
        self.setWindowTitle("Airfoil Viewer")
        self.setGeometry(100, 100, 800, 600)

        self.layout = QGridLayout()

        self.listWidget = QListWidget()
        self.listWidget.itemClicked.connect(self.plot_airfoil)
        self.layout.addWidget(self.listWidget, 0, 0, 2, 1) # Span 2 rows

        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas, 0, 1, 1, 2) # Span 2 columns

        self.geometry_labels = {
            "Max Thickness:": QLabel(""),
            "Max Camber:": QLabel(""),
            "Chord Length:": QLabel(""),
            "Span:": QLabel(""),
            "Aspect Ratio:": QLabel(""),
            "Leading Edge Radius:": QLabel(""),
            "Trailing Edge Angle:": QLabel(""),
            "Thickness/Chord Ratio:": QLabel("")
        }

        row = 1
        col = 1
        for label_text, label_widget in self.geometry_labels.items():
            self.layout.addWidget(QLabel(label_text), row, col)
            self.layout.addWidget(label_widget, row, col + 1)
            row += 1

        self.setLayout(self.layout)

    def load_airfoil_names(self):
        with sqlite3.connect(self.airfoil_db_instance.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM airfoils")
            airfoils = cursor.fetchall()
        for airfoil in airfoils:
            self.listWidget.addItem(airfoil[0])

    def plot_airfoil(self, item):
        airfoil_name = item.text()
        self.ax.clear()
        self.airfoil_db_instance.plot_airfoil(airfoil_name, self.ax)
        self.canvas.draw()
        self.display_geometry_data(airfoil_name)

    def display_geometry_data(self, airfoil_name):
        with sqlite3.connect(self.airfoil_db_instance.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT max_thickness, max_camber, chord_length, span, aspect_ratio, 
                       leading_edge_radius, trailing_edge_angle, thickness_to_chord_ratio
                FROM airfoil_geometry
                WHERE name = ?
            """, (airfoil_name,))
            geometry_data = cursor.fetchone()

        if geometry_data:
            labels = list(self.geometry_labels.values())
            for i, value in enumerate(geometry_data):
                #labels[i].setText(str(value))
                labels[i].setText(f"{value:.5f}")
        else:
            for label in self.geometry_labels.values():
                label.setText("N/A")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = AirfoilViewer()
    viewer.show()
    sys.exit(app.exec())