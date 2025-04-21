import sys
import os
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QListWidget, QLabel,
                             QGridLayout, QTabWidget, QLineEdit, QPushButton, QComboBox,
                             QScrollArea, QWidget, QFileDialog, QRadioButton, QGroupBox,
                             QMenuBar, QMenu, QMessageBox, QDialog, QFormLayout, QDoubleSpinBox,
                             QCheckBox)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from DASC500.classes.AirfoilDatabase import AirfoilDatabase
from DASC500.classes.AirfoilSeries import AirfoilSeries

from DASC500.xfoil.fix_airfoil_data import *


class AirfoilViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.airfoil_db_instance = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Airfoil Viewer")
        self.setGeometry(100, 100, 1000, 800)

        self.layout = QVBoxLayout()

        self.setup_menu()
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)

        self.setup_start_tab()
        self.setup_viewer_tab()
        self.setup_geometry_search_tab()
        self.setup_compare_tab()
        self.setup_xfoil_tab()
        self.setup_xfoil_results_search_tab()
        self.setup_xfoil_plot()

        self.setLayout(self.layout)
        self.disable_other_tabs()

    def setup_menu(self):
        menubar = QMenuBar()
        file_menu = QMenu("File", self)
        menubar.addMenu(file_menu)

        open_action = file_menu.addAction("Open Database")
        open_action.triggered.connect(self.browse_database)

        save_action = file_menu.addAction("Save Database")
        save_action.triggered.connect(self.save_database)

        save_as_action = file_menu.addAction("Save Database As")
        save_as_action.triggered.connect(self.save_database_as)

        clear_db_action = file_menu.addAction("Clear Database")
        clear_db_action.triggered.connect(self.clear_database)

        self.layout.setMenuBar(menubar)

    def save_database(self):
        """Saves the current database."""
        if self.airfoil_db_instance:
            try:
                self.airfoil_db_instance.save_database()
                QMessageBox.information(self, "Database Saved", "Database saved successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error Saving Database", f"An error occurred: {e}")
        else:
            QMessageBox.warning(self, "Save Database", "No database is currently open.")

    def save_database_as(self):
        """Saves the current database to a new file."""
        if self.airfoil_db_instance:
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Airfoil Database As", "", "SQLite Database (*.db)")
            if file_path:
                try:
                    self.airfoil_db_instance.save_database_as(file_path)
                    QMessageBox.information(self, "Database Saved As", "Database saved successfully.")
                except Exception as e:
                    QMessageBox.critical(self, "Error Saving Database", f"An error occurred: {e}")
        else:
            QMessageBox.warning(self, "Save Database As", "No database is currently open.")

    def clear_database(self):
        """Clears the current database."""
        if self.airfoil_db_instance:
            try:
                self.airfoil_db_instance.clear_database()
                QMessageBox.information(self, "Database Cleared", "Database cleared successfully.")
                self.disable_other_tabs()
                self.populate_airfoil_lists()
            except Exception as e:
                QMessageBox.critical(self, "Error Clearing Database", f"An error occurred: {e}")
        else:
            QMessageBox.warning(self, "Clear Database", "No database is currently open.")

    def populate_airfoil_lists(self):
        """Populates all airfoil lists in the UI."""
        self.xfoil_airfoil_list.clear()
        # Add other lists here when the other tabs are created.
        if self.airfoil_db_instance:
            try:
                with sqlite3.connect(self.airfoil_db_instance.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM airfoils")
                    results = cursor.fetchall()
                    for row in results:
                        self.listWidget.addItem(row[0])
                        self.xfoil_airfoil_list.addItem(row[0])
            except sqlite3.Error as e:
                QMessageBox.critical(self, "Database Error", f"Error populating airfoil list: {e}")

    def setup_start_tab(self):
        start_tab = QWidget()
        start_layout = QVBoxLayout()

        # Database Selection
        db_group = QGroupBox("Database Selection")
        db_layout = QVBoxLayout()

        self.db_path_edit = QLineEdit()
        db_layout.addWidget(self.db_path_edit)

        db_browse_button = QPushButton("Browse")
        db_browse_button.clicked.connect(self.browse_database)
        db_layout.addWidget(db_browse_button)

        db_group.setLayout(db_layout)
        start_layout.addWidget(db_group)

        # File Import
        file_group = QGroupBox("Import Airfoils")
        file_layout = QVBoxLayout()

        self.file_path_edit = QLineEdit()
        file_layout.addWidget(self.file_path_edit)

        file_browse_button = QPushButton("Browse")
        file_browse_button.clicked.connect(self.browse_file)
        file_layout.addWidget(file_browse_button)

        self.overwrite_checkbox = QRadioButton("Overwrite Existing")
        file_layout.addWidget(self.overwrite_checkbox)

        import_button = QPushButton("Import")
        import_button.clicked.connect(self.import_file)
        file_layout.addWidget(import_button)

        file_group.setLayout(file_layout)
        start_layout.addWidget(file_group)

        # Individual Airfoil Addition
        airfoil_group = QGroupBox("Add Individual Airfoil")
        airfoil_layout = QGridLayout()

        airfoil_layout.addWidget(QLabel("Name:"), 0, 0)
        self.airfoil_name_edit = QLineEdit()
        airfoil_layout.addWidget(self.airfoil_name_edit, 0, 1)

        airfoil_layout.addWidget(QLabel("Description:"), 1, 0)
        self.airfoil_desc_edit = QLineEdit()
        airfoil_layout.addWidget(self.airfoil_desc_edit, 1, 1)

        airfoil_layout.addWidget(QLabel("Airfoil Series:"), 2, 0)
        self.airfoil_series_combo = QComboBox()
        for series in AirfoilSeries:
            self.airfoil_series_combo.addItem(series.value)
        airfoil_layout.addWidget(self.airfoil_series_combo, 2, 1)

        airfoil_layout.addWidget(QLabel("Source:"), 3, 0)
        self.airfoil_source_edit = QLineEdit()
        airfoil_layout.addWidget(self.airfoil_source_edit, 3, 1)

        airfoil_layout.addWidget(QLabel("Point Cloud File:"), 4, 0)
        self.airfoil_pointcloud_edit = QLineEdit()
        airfoil_layout.addWidget(self.airfoil_pointcloud_edit, 4, 1)

        pointcloud_browse_button = QPushButton("Browse")
        pointcloud_browse_button.clicked.connect(self.browse_pointcloud)
        airfoil_layout.addWidget(pointcloud_browse_button, 4, 2)

        add_airfoil_button = QPushButton("Add Airfoil")
        add_airfoil_button.clicked.connect(self.add_individual_airfoil)
        airfoil_layout.addWidget(add_airfoil_button, 5, 0, 1, 3)

        airfoil_group.setLayout(airfoil_layout)
        start_layout.addWidget(airfoil_group)

        # Load Database Button
        load_button = QPushButton("Load Database")
        load_button.clicked.connect(self.load_database)
        start_layout.addWidget(load_button)

        start_tab.setLayout(start_layout)
        self.tabs.addTab(start_tab, "Start")

    def browse_database(self):
        """Opens a file dialog to select a database."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Airfoil Database", "", "SQLite Database (*.db)")
        if file_path:
            try:
                self.airfoil_db_instance = AirfoilDatabase(file_path)
                self.enable_other_tabs()
                self.populate_airfoil_lists()  # Ensure this is called
                QMessageBox.information(self, "Database Opened", "Database opened successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error Opening Database", f"An error occurred: {e}")

    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File", "", "CSV Files (*.csv);;JSON Files (*.json);;All Files (*.*)")
        if file_path:
            self.file_path_edit.setText(file_path)

    def import_file(self):
        file_path = self.file_path_edit.text()
        if not file_path:
            return

        overwrite = self.overwrite_checkbox.isChecked()
        _, file_extension = os.path.splitext(file_path)
        file_extension = file_extension.lower()

        if file_extension == '.csv':
            self.airfoil_db_instance.add_airfoils_from_csv(file_path, overwrite)
        elif file_extension == '.json':
            self.airfoil_db_instance.add_airfoils_from_json(file_path, overwrite)
        else:
            QMessageBox.warning(self, "Import File", "Unsupported file type. Please select a CSV or JSON file.")
            return

    def load_database(self):
        db_path = self.db_path_edit.text()
        if not db_path:
            return

        self.airfoil_db_instance = AirfoilDatabase(db_name=os.path.basename(db_path), db_dir=os.path.dirname(db_path))
        self.enable_other_tabs()
        self.load_airfoil_names()
    
    def load_airfoil_names(self):
        self.listWidget.clear()
        self.xfoil_airfoil_list.clear() #Added this line to clear xfoil list as well.
        if self.airfoil_db_instance:
            with sqlite3.connect(self.airfoil_db_instance.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM airfoils")
                airfoils = cursor.fetchall()
            for airfoil in airfoils:
                self.listWidget.addItem(airfoil[0])
                self.xfoil_airfoil_list.addItem(airfoil[0]) #Populate xfoil list as well.
            self.viewer_name_edit.setEnabled(True)
            self.viewer_desc_edit.setEnabled(True)
            self.viewer_series_combo.setEnabled(True)
            self.viewer_source_edit.setEnabled(True)

    def browse_pointcloud(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Point Cloud File", "", "Text Files (*.txt);;All Files (*.*)")
        if file_path:
            self.airfoil_pointcloud_edit.setText(file_path)

    def add_individual_airfoil(self):
        if not self.airfoil_db_instance:
            QMessageBox.warning(self, "Add Airfoil", "Please load a database first.")
            return

        name = self.airfoil_name_edit.text()
        description = self.airfoil_desc_edit.text()
        series = self.airfoil_series_combo.currentText() #Get the value from the combo box.
        source = self.airfoil_source_edit.text()
        pointcloud_file = self.airfoil_pointcloud_edit.text()

        if not name or not pointcloud_file:
            QMessageBox.warning(self, "Add Airfoil", "Name and Point Cloud File are required.")
            return

        try:
            with open(pointcloud_file, 'r') as file:
                pointcloud = file.read()
            self.airfoil_db_instance.store_airfoil_data(name, description, pointcloud, series, source)
            self.load_airfoil_names()
            QMessageBox.information(self, "Add Airfoil", f"Airfoil {name} added successfully.")
        except FileNotFoundError:
            QMessageBox.warning(self, "Add Airfoil", f"Point Cloud File not found: {pointcloud_file}")

    def setup_viewer_tab(self):
        viewer_tab = QWidget()
        viewer_layout = QGridLayout()

        self.listWidget = QListWidget()
        self.listWidget.itemClicked.connect(self.plot_airfoil)
        viewer_layout.addWidget(self.listWidget, 0, 0, 2, 1)

        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        viewer_layout.addWidget(self.canvas, 0, 1, 1, 3)

        # Airfoil Geometry and Info
        info_group = QGroupBox("Airfoil Geometry and Info")
        info_layout = QGridLayout()

        # Geometry Labels
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

        row = 0
        col = 0
        for label_text, label_widget in self.geometry_labels.items():
            info_layout.addWidget(QLabel(label_text), row, col)
            info_layout.addWidget(label_widget, row, col + 1)
            row += 1
            if row == 4:
                row = 0
                col += 2

        # Airfoil Info Edits
        info_layout.addWidget(QLabel("Name:"), 0, 4)
        self.viewer_name_edit = QLineEdit()
        info_layout.addWidget(self.viewer_name_edit, 0, 5)

        info_layout.addWidget(QLabel("Description:"), 1, 4)
        self.viewer_desc_edit = QLineEdit()
        info_layout.addWidget(self.viewer_desc_edit, 1, 5)

        info_layout.addWidget(QLabel("Airfoil Series:"), 2, 4)
        self.viewer_series_combo = QComboBox()
        for series in AirfoilSeries:
            self.viewer_series_combo.addItem(series.value)
        info_layout.addWidget(self.viewer_series_combo, 2, 5)

        info_layout.addWidget(QLabel("Source:"), 3, 4)
        self.viewer_source_edit = QLineEdit()
        info_layout.addWidget(self.viewer_source_edit, 3, 5)

        self.update_info_button = QPushButton("Update Info")
        self.update_info_button.clicked.connect(self.update_airfoil_info)
        info_layout.addWidget(self.update_info_button, 4, 4, 1, 2)

        info_group.setLayout(info_layout)
        viewer_layout.addWidget(info_group, 1, 1, 1, 3)

        viewer_tab.setLayout(viewer_layout)
        self.tabs.addTab(viewer_tab, "Viewer")
    
    def update_airfoil_info(self):
        selected_item = self.listWidget.currentItem()
        if not selected_item:
            QMessageBox.warning(self, "Update Info", "Please select an airfoil.")
            return

        name = selected_item.text()
        new_name = self.viewer_name_edit.text()
        description = self.viewer_desc_edit.text()
        series = self.viewer_series_combo.currentText()
        source = self.viewer_source_edit.text()

        self.airfoil_db_instance.update_airfoil_info(name, new_name, description, series, source)
        selected_item.setText(new_name)
        self.load_airfoil_names()
        QMessageBox.information(self, "Update Info", "Airfoil info updated.")
    
    def plot_airfoil(self, item):
        airfoil_name = item.text()
        self.ax.clear()
        self.airfoil_db_instance.plot_airfoil(airfoil_name, self.ax)
        self.canvas.draw()
        self.display_geometry_data(airfoil_name)
        self.populate_info_edits(airfoil_name)

    def populate_info_edits(self, airfoil_name):
        data = self.airfoil_db_instance.get_airfoil_data(airfoil_name)
        if data and len(data) == 4:
            description, pointcloud, series, source = data
            self.viewer_name_edit.setText(airfoil_name)
            self.viewer_desc_edit.setText(description)
            self.viewer_series_combo.setCurrentText(series)
            self.viewer_source_edit.setText(source)

            # Populate Geometry Data
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
                    if value is not None:
                        labels[i].setText("{:.3f}".format(value))
                    else:
                        labels[i].setText(str(value))
            else:
                for label in self.geometry_labels.values():
                    label.setText("N/A")

        else:
            self.viewer_name_edit.clear()
            self.viewer_desc_edit.clear()
            self.viewer_series_combo.setCurrentIndex(0)
            self.viewer_source_edit.clear()
            for label in self.geometry_labels.values():
                label.setText("N/A")

    def setup_geometry_search_tab(self):
        search_tab = QWidget()
        search_layout = QGridLayout()  # Use QGridLayout for better layout

        self.search_params = {}
        self.search_fields = ["max_thickness", "max_camber", "chord_length", "span",
                               "aspect_ratio", "leading_edge_radius", "trailing_edge_angle",
                               "thickness_to_chord_ratio"]

        row = 0
        for field in self.search_fields:
            search_layout.addWidget(QLabel(field.replace("_", " ").title()), row, 0)
            self.search_params[field] = QLineEdit()
            search_layout.addWidget(self.search_params[field], row, 1)
            row += 1

        # Buttons
        self.search_button = QPushButton("Search")
        self.search_button.clicked.connect(self.search_airfoils)
        search_layout.addWidget(self.search_button, row, 0, 1, 2)
        row += 1
        
        self.clear_criteria_button = QPushButton("Clear Criteria")
        self.clear_criteria_button.clicked.connect(self.clear_search_criteria)
        search_layout.addWidget(self.clear_criteria_button, row, 0, 1, 2)
        row += 1
        
        # Results List
        self.search_results_list = QListWidget()
        self.search_results_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        search_layout.addWidget(self.search_results_list, 0, 2, row, 1)  # Span rows, column 2

        # List Clear Button
        self.clear_list_button = QPushButton("Clear List")
        self.clear_list_button.clicked.connect(self.clear_search_list)
        search_layout.addWidget(self.clear_list_button, row, 2)
        row += 1

        # Add Matplotlib figure and canvas
        self.search_fig, self.search_ax = plt.subplots(figsize=(10, 7))
        self.search_canvas = FigureCanvas(self.search_fig)
        search_layout.addWidget(self.search_canvas, 0, 3, row, 2)  # Span rows, columns 3 and 4

        # Plot Buttons
        self.clear_plot_button = QPushButton("Clear Plot")
        self.clear_plot_button.clicked.connect(self.clear_search_plot)
        search_layout.addWidget(self.clear_plot_button, row, 3)

        self.plot_selected_button = QPushButton("Plot Selected")
        self.plot_selected_button.clicked.connect(self.plot_selected_airfoils_search_tab)
        search_layout.addWidget(self.plot_selected_button, row, 4)

        search_tab.setLayout(search_layout)
        self.tabs.addTab(search_tab, "Geometry Search")

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
                labels[i].setText(str(value))
        else:
            for label in self.geometry_labels.values():
                label.setText("N/A")

    def search_airfoils(self):
        search_criteria = {}
        for field, line_edit in self.search_params.items():
            value = line_edit.text()
            if value:
                try:
                    search_criteria[field] = float(value)
                except ValueError:
                    print(f"Invalid input for {field}")
                    return

        # Clear previous results
        self.search_results_list.clear()

        # Iterate through search criteria and find matching airfoils
        all_results = set()  # Use a set to avoid duplicates

        for field, target_value in search_criteria.items():
            # Set default tolerance and tolerance_type
            tolerance = 0.1  # You can adjust this default tolerance
            tolerance_type = "absolute"  # Or "percentage"

            # Call find_airfoils_by_geometry and add results to the set
            airfoil_names = self.airfoil_db_instance.find_airfoils_by_geometry(
                field, target_value, tolerance, tolerance_type
            )
            all_results.update(airfoil_names)

        # Populate the list widget with the combined results
        for name in sorted(list(all_results)):
            self.search_results_list.addItem(name)

    def plot_selected_airfoils_search_tab(self):
        selected_items = self.search_results_list.selectedItems()
        names = [item.text() for item in selected_items]
        if names:
            self.search_ax.clear()  # Clear the existing plot
            for name in names:
                self.airfoil_db_instance.add_airfoil_to_plot(name, self.search_ax, linestyle='-', marker='o', markersize=3)
            self.search_ax.set_xlabel("X Coordinate")
            self.search_ax.set_ylabel("Y Coordinate")
            self.search_ax.set_title("Selected Airfoil Comparison")
            self.search_ax.grid(True)
            self.search_ax.axis('equal')
            self.search_ax.legend()
            self.search_canvas.draw()
    
    def clear_search_criteria(self):
        for line_edit in self.search_params.values():
            line_edit.clear()

    def clear_search_list(self):
        self.search_results_list.clear()

    def clear_search_plot(self):
        self.search_ax.clear()
        self.search_canvas.draw()
    
    def disable_other_tabs(self):
        """Disables other tabs until a database is opened."""
        for i in range(1, self.tabs.count()):
            self.tabs.setTabEnabled(i, False)

    def enable_other_tabs(self):
        """Enables other tabs after a database is opened."""
        for i in range(1, self.tabs.count()):
            self.tabs.setTabEnabled(i, True)
    
    def setup_compare_tab(self):
        compare_tab = QWidget()
        compare_layout = QVBoxLayout()

        # Point Cloud Selection
        pointcloud_group = QGroupBox("Point Cloud Selection")
        pointcloud_layout = QGridLayout()

        pointcloud_layout.addWidget(QLabel("Point Cloud File:"), 0, 0)
        self.compare_pointcloud_edit = QLineEdit()
        pointcloud_layout.addWidget(self.compare_pointcloud_edit, 0, 1)

        pointcloud_browse_button = QPushButton("Browse")
        pointcloud_browse_button.clicked.connect(self.browse_compare_pointcloud)
        pointcloud_layout.addWidget(pointcloud_browse_button, 0, 2)

        pointcloud_group.setLayout(pointcloud_layout)
        compare_layout.addWidget(pointcloud_group)

        # Plotting
        self.compare_fig, self.compare_ax = plt.subplots(figsize=(10, 7))
        self.compare_canvas = FigureCanvas(self.compare_fig)
        self.compare_toolbar = NavigationToolbar(self.compare_canvas, self)
        compare_layout.addWidget(self.compare_toolbar)
        compare_layout.addWidget(self.compare_canvas)

        # Compare Button
        compare_button = QPushButton("Compare")
        compare_button.clicked.connect(self.compare_airfoils)
        compare_layout.addWidget(compare_button)

        compare_tab.setLayout(compare_layout)
        self.tabs.addTab(compare_tab, "Compare")

        self.compare_canvas.mpl_connect('button_press_event', self.on_compare_click)
        self.compare_pointcloud_points = None

    def browse_compare_pointcloud(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Point Cloud File", "", "Text Files (*.txt);;All Files (*.*)")
        if file_path:
            self.compare_pointcloud_edit.setText(file_path)
            self.load_compare_pointcloud(file_path)

    def load_compare_pointcloud(self, file_path):
        try:
            with open(file_path, 'r') as file:
                pointcloud_str = file.read()
            points = [line.split() for line in pointcloud_str.strip().split('\n')]
            points = np.array([[float(p[0]), float(p[1])] for p in points if len(p) == 2])
            normalized_points = normalize_pointcloud(points)
            self.compare_pointcloud_points = normalized_points

            self.compare_ax.clear()
            self.compare_ax.plot(normalized_points[:, 0], normalized_points[:, 1], 'o-', label="Input Point Cloud")
            self.compare_ax.set_xlabel("X Coordinate")
            self.compare_ax.set_ylabel("Y Coordinate")
            self.compare_ax.grid(True)
            self.compare_ax.axis('equal')
            self.compare_ax.legend()
            self.compare_canvas.draw()

        except FileNotFoundError:
            QMessageBox.warning(self, "Load Point Cloud", "File not found.")
        except Exception as e:
            QMessageBox.warning(self, "Load Point Cloud", f"Error loading point cloud: {e}")

    def compare_airfoils(self):
        if self.compare_pointcloud_points is None:
            QMessageBox.warning(self, "Compare Airfoils", "Please load a point cloud first.")
            return

        pointcloud_str = '\n'.join([' '.join(map(str, point)) for point in self.compare_pointcloud_points])
        matches = self.airfoil_db_instance.find_best_matching_airfoils(pointcloud_str)

        if matches:
            best_match_name, _ = matches[0]
            self.airfoil_db_instance.add_airfoil_to_plot(best_match_name, self.compare_ax, linestyle='--', label=f"Best Match: {best_match_name}")
            self.compare_ax.legend()
            self.compare_canvas.draw()
        else:
            QMessageBox.information(self, "Compare Airfoils", "No matching airfoils found.")

    def on_compare_click(self, event):
        if event.inaxes == self.compare_ax and event.button == 1 and self.compare_pointcloud_points is not None:
            x, y = event.xdata, event.ydata
            distances = np.linalg.norm(self.compare_pointcloud_points - np.array([x, y]), axis=1)
            closest_point_index = np.argmin(distances)

            dialog = PointEditDialog(self.compare_pointcloud_points[closest_point_index, 0],
                                     self.compare_pointcloud_points[closest_point_index, 1], self)
            if dialog.exec():
                new_x, new_y = dialog.get_values()
                self.compare_pointcloud_points[closest_point_index, 0] = new_x
                self.compare_pointcloud_points[closest_point_index, 1] = new_y
                self.compare_ax.clear()
                self.compare_ax.plot(self.compare_pointcloud_points[:, 0], self.compare_pointcloud_points[:, 1], 'o-', label="Input Point Cloud")
                self.compare_ax.set_xlabel("X Coordinate")
                self.compare_ax.set_ylabel("Y Coordinate")
                self.compare_ax.grid(True)
                self.compare_ax.axis('equal')
                self.compare_ax.legend()
                self.compare_canvas.draw()
    
    def setup_xfoil_tab(self):
        """Sets up the XFOIL Results tab."""
        xfoil_tab = QWidget()
        xfoil_layout = QVBoxLayout()

        # Airfoil Selection
        airfoil_group = QGroupBox("Airfoil Selection")
        airfoil_layout = QVBoxLayout()
        self.xfoil_airfoil_list = QListWidget()
        self.xfoil_airfoil_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.xfoil_airfoil_list.itemSelectionChanged.connect(self.populate_reynolds_mach)
        airfoil_layout.addWidget(self.xfoil_airfoil_list)
        airfoil_group.setLayout(airfoil_layout)
        xfoil_layout.addWidget(airfoil_group)

        # Populate the airfoil list from the database
        if self.airfoil_db_instance:
            try:
                with sqlite3.connect(self.airfoil_db_instance.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM airfoils")
                    results = cursor.fetchall()
                    for row in results:
                        self.xfoil_airfoil_list.addItem(row[0])
            except sqlite3.Error as e:
                QMessageBox.critical(self, "Database Error", f"Error populating airfoil list: {e}")
        else:
            QMessageBox.warning(self, "Airfoil Selection", "Airfoil Database Instance is not set.")

        # Reynolds/Mach Selection
        reynolds_mach_group = QGroupBox("Reynolds/Mach Selection")
        reynolds_mach_layout = QGridLayout()
        reynolds_mach_layout.addWidget(QLabel("Reynolds Number:"), 0, 0)
        self.xfoil_reynolds_combo = QComboBox()
        reynolds_mach_layout.addWidget(self.xfoil_reynolds_combo, 0, 1)
        reynolds_mach_layout.addWidget(QLabel("Mach Number:"), 1, 0)
        self.xfoil_mach_combo = QComboBox()
        reynolds_mach_layout.addWidget(self.xfoil_mach_combo, 1, 1)
        reynolds_mach_group.setLayout(reynolds_mach_layout)
        xfoil_layout.addWidget(reynolds_mach_group)

        # Coefficient Selection
        coefficient_group = QGroupBox("Coefficient Selection")
        coefficient_layout = QVBoxLayout()
        self.xfoil_cl_check = QCheckBox("Cl")
        self.xfoil_cd_check = QCheckBox("Cd")
        self.xfoil_cm_check = QCheckBox("Cm")
        coefficient_layout.addWidget(self.xfoil_cl_check)
        coefficient_layout.addWidget(self.xfoil_cd_check)
        coefficient_layout.addWidget(self.xfoil_cm_check)
        coefficient_group.setLayout(coefficient_layout)
        xfoil_layout.addWidget(coefficient_group)

        # Plot Button
        self.xfoil_plot_button = QPushButton("Plot")
        self.xfoil_plot_button.clicked.connect(self.plot_xfoil_results)
        xfoil_layout.addWidget(self.xfoil_plot_button)

        # Plotting Area
        self.xfoil_fig, self.xfoil_ax = plt.subplots(figsize=(10, 7))
        self.xfoil_canvas = FigureCanvas(self.xfoil_fig)
        self.xfoil_toolbar = NavigationToolbar(self.xfoil_canvas, self)
        xfoil_layout.addWidget(self.xfoil_toolbar)
        xfoil_layout.addWidget(self.xfoil_canvas)

        xfoil_tab.setLayout(xfoil_layout)
        self.tabs.addTab(xfoil_tab, "XFOIL Results")

    def populate_reynolds_mach(self):
        """Populates the Reynolds and Mach combo boxes."""
        selected_items = self.xfoil_airfoil_list.selectedItems()
        if not selected_items:
            return

        airfoil_name = selected_items[0].text()
        self.xfoil_reynolds_combo.clear()
        self.xfoil_mach_combo.clear()

        if self.airfoil_db_instance:
            try:
                with sqlite3.connect(self.airfoil_db_instance.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT DISTINCT reynolds_number, mach FROM aero_coeffs WHERE name = ?", (airfoil_name,))
                    results = cursor.fetchall()

                    reynolds_set = set()
                    mach_set = set()

                    for reynolds, mach in results:
                        if reynolds is not None:
                            reynolds_set.add(reynolds)
                        if mach is not None:
                            mach_set.add(mach)

                    for reynolds in sorted(reynolds_set):
                        self.xfoil_reynolds_combo.addItem(str(reynolds))
                    for mach in sorted(mach_set):
                        self.xfoil_mach_combo.addItem(str(mach))

            except sqlite3.Error as e:
                QMessageBox.critical(self, "Database Error", f"Error populating Reynolds/Mach: {e}")

    def plot_xfoil_results(self):
        """Plots the XFOIL results based on user selections."""
        selected_items = self.xfoil_airfoil_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Plot XFOIL Results", "Please select an airfoil.")
            return

        airfoil_name = selected_items[0].text()
        reynolds = float(self.xfoil_reynolds_combo.currentText()) if self.xfoil_reynolds_combo.currentText() else None
        mach = float(self.xfoil_mach_combo.currentText()) if self.xfoil_mach_combo.currentText() else None
        coefficients = [c for c, check in [("cl", self.xfoil_cl_check), ("cd", self.xfoil_cd_check), ("cm", self.xfoil_cm_check)] if check.isChecked()]

        if not coefficients:
            QMessageBox.warning(self, "Plot XFOIL Results", "Please select coefficients.")
            return

        self.xfoil_ax.clear()
        for coeff in coefficients:
            x_values, y_values = [], []
            data = self.airfoil_db_instance.get_aero_coeffs(airfoil_name, reynolds, mach)
            if data:
                for row in data:
                    x_values.append(row[4])  # Assuming alpha is at index 4
                    if coeff == "cl":
                        y_values.append(row[5])  # Assuming cl is at index 5
                    elif coeff == "cd":
                        y_values.append(row[6])  # Assuming cd is at index 6
                    elif coeff == "cm":
                        y_values.append(row[7])  # Assuming cm is at index 7
                self.xfoil_ax.plot(x_values, y_values, label=f"{airfoil_name} - {coeff}")

        self.xfoil_ax.set_xlabel("Alpha")
        self.xfoil_ax.set_ylabel("Coefficient Value")
        self.xfoil_ax.set_title(f"XFOIL Results: {airfoil_name} (Re={reynolds}, Mach={mach})")
        self.xfoil_ax.legend()
        self.xfoil_canvas.draw()

    def setup_xfoil_results_search_tab(self):
        """Sets up the XFOIL results search tab."""
        xfoil_search_tab = QWidget()
        xfoil_search_layout = QGridLayout()

        # Input fields for XFOIL search parameters
        xfoil_search_layout.addWidget(QLabel("Parameter:"), 0, 0)
        self.xfoil_parameter_combo = QComboBox()
        self.xfoil_parameter_combo.addItems(["reynolds", "alpha", "cl", "cd", "cm"])
        xfoil_search_layout.addWidget(self.xfoil_parameter_combo, 0, 1)

        xfoil_search_layout.addWidget(QLabel("Target Value:"), 1, 0)
        self.xfoil_target_value_edit = QLineEdit()
        xfoil_search_layout.addWidget(self.xfoil_target_value_edit, 1, 1)

        xfoil_search_layout.addWidget(QLabel("Tolerance:"), 2, 0)
        self.xfoil_tolerance_edit = QLineEdit()
        xfoil_search_layout.addWidget(self.xfoil_tolerance_edit, 2, 1)

        xfoil_search_layout.addWidget(QLabel("Tolerance Type:"), 3, 0)
        self.xfoil_tolerance_type_combo = QComboBox()
        self.xfoil_tolerance_type_combo.addItems(["absolute", "percentage"])
        xfoil_search_layout.addWidget(self.xfoil_tolerance_type_combo, 3, 1)

        # Search button
        self.xfoil_search_button = QPushButton("Search XFOIL Results")
        self.xfoil_search_button.clicked.connect(self.perform_xfoil_results_search)
        xfoil_search_layout.addWidget(self.xfoil_search_button, 4, 0, 1, 2)

        # Results list
        self.xfoil_search_results_list = QListWidget()
        self.xfoil_search_results_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        xfoil_search_layout.addWidget(self.xfoil_search_results_list, 0, 2, 5, 1)

        # Plot area
        self.xfoil_search_fig, self.xfoil_search_ax = plt.subplots(figsize=(10, 7))
        self.xfoil_search_canvas = FigureCanvas(self.xfoil_search_fig)
        xfoil_search_layout.addWidget(self.xfoil_search_canvas, 0, 3, 5, 2)

        # Plot buttons
        self.xfoil_clear_plot_button = QPushButton("Clear Plot")
        self.xfoil_clear_plot_button.clicked.connect(self.clear_xfoil_search_plot)
        xfoil_search_layout.addWidget(self.xfoil_clear_plot_button, 5, 3)

        self.xfoil_plot_selected_button = QPushButton("Plot Selected")
        self.xfoil_plot_selected_button.clicked.connect(self.plot_selected_airfoils_xfoil_search_tab)
        xfoil_search_layout.addWidget(self.xfoil_plot_selected_button, 5, 4)

        xfoil_search_tab.setLayout(xfoil_search_layout)
        self.tabs.addTab(xfoil_search_tab, "XFOIL Results Search")

    def perform_xfoil_results_search(self):
        """Performs the XFOIL results search."""
        parameter = self.xfoil_parameter_combo.currentText()
        try:
            target_value = float(self.xfoil_target_value_edit.text())
            tolerance = float(self.xfoil_tolerance_edit.text())
        except ValueError:
            QMessageBox.warning(self, "XFOIL Results Search", "Invalid input for target value or tolerance.")
            return

        tolerance_type = self.xfoil_tolerance_type_combo.currentText()

        results = self.airfoil_db_instance.find_airfoils_by_xfoil_results(
            parameter, target_value, tolerance, tolerance_type
        )

        self.xfoil_search_results_list.clear()
        if results:
            self.xfoil_search_results_list.addItems(results)
        else:
            QMessageBox.information(self, "XFOIL Results Search", "No matching airfoils found.")

    def clear_xfoil_search_plot(self):
        """Clears the XFOIL search plot."""
        self.xfoil_search_ax.clear()
        self.xfoil_search_canvas.draw()

    def plot_selected_airfoils_xfoil_search_tab(self):
        """Plots the selected airfoils from the search results list."""
        selected_items = self.xfoil_search_results_list.selectedItems()
        names = [item.text() for item in selected_items]
        if names:
            self.xfoil_search_ax.clear()
            for name in names:
                self.airfoil_db_instance.add_airfoil_to_plot(name, self.xfoil_search_ax, linestyle='-', marker='o', markersize=3)
            self.xfoil_search_ax.set_xlabel("X Coordinate")
            self.xfoil_search_ax.set_ylabel("Y Coordinate")
            self.xfoil_search_ax.set_title("Selected Airfoil Comparison")
            self.xfoil_search_ax.grid(True)
            self.xfoil_search_ax.axis('equal')
            self.xfoil_search_ax.legend()
            self.xfoil_search_canvas.draw()

    def setup_xfoil_plot(self):
        """Sets up the XFOIL plot tab."""
        xfoil_plot_tab = QWidget()
        xfoil_plot_layout = QVBoxLayout()

        # Airfoil selection (reuse existing xfoil_airfoil_list)
        airfoil_group = QGroupBox("Airfoil Selection")
        airfoil_layout = QVBoxLayout()
        airfoil_layout.addWidget(self.xfoil_airfoil_list)  # Reuse existing list
        airfoil_group.setLayout(airfoil_layout)
        xfoil_plot_layout.addWidget(airfoil_group)

        # Reynolds and alpha input
        input_layout = QGridLayout()
        input_layout.addWidget(QLabel("Reynolds Number:"), 0, 0)
        self.xfoil_plot_reynolds_edit = QLineEdit()
        input_layout.addWidget(self.xfoil_plot_reynolds_edit, 0, 1)

        input_layout.addWidget(QLabel("Alpha:"), 1, 0)
        self.xfoil_plot_alpha_edit = QLineEdit()
        input_layout.addWidget(self.xfoil_plot_alpha_edit, 1, 1)
        xfoil_plot_layout.addLayout(input_layout)

        # Plot button
        self.xfoil_plot_button = QPushButton("Plot XFOIL Data")
        self.xfoil_plot_button.clicked.connect(self.perform_xfoil_plot)
        xfoil_plot_layout.addWidget(self.xfoil_plot_button)

        # Plot area
        self.xfoil_plot_fig, self.xfoil_plot_ax = plt.subplots(figsize=(10, 7))
        self.xfoil_plot_canvas = FigureCanvas(self.xfoil_plot_fig)
        xfoil_plot_layout.addWidget(self.xfoil_plot_canvas)

        xfoil_plot_tab.setLayout(xfoil_plot_layout)
        self.tabs.addTab(xfoil_plot_tab, "XFOIL Plot")

    def perform_xfoil_plot(self):
        """Performs the XFOIL plot."""
        try:
            reynolds = float(self.xfoil_plot_reynolds_edit.text())
            alpha = float(self.xfoil_plot_alpha_edit.text())
        except ValueError:
            QMessageBox.warning(self, "XFOIL Plot", "Invalid input for Reynolds number or alpha.")
            return

        selected_items = self.xfoil_airfoil_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "XFOIL Plot", "Please select an airfoil.")
            return

        airfoil_name = selected_items[0].text()

        data = self.airfoil_db_instance.get_xfoil_data(airfoil_name, reynolds, alpha)
        if data:
            self.xfoil_plot_ax.clear()

            # Extract and plot CL, CD, CM
            alpha_values = [d["alpha"] for d in data]
            cl_values = [d["cl"] for d in data]
            cd_values = [d["cd"] for d in data]
            cm_values = [d["cm"] for d in data]

            self.xfoil_plot_ax.plot(alpha_values, cl_values, label="CL")
            self.xfoil_plot_ax.plot(alpha_values, cd_values, label="CD")
            self.xfoil_plot_ax.plot(alpha_values, cm_values, label="CM")

            self.xfoil_plot_ax.set_xlabel("Alpha (degrees)")
            self.xfoil_plot_ax.set_ylabel("Coefficient Value")
            self.xfoil_plot_ax.grid(True)
            self.xfoil_plot_ax.legend()
            self.xfoil_plot_canvas.draw()
        else:
            QMessageBox.information(self, "XFOIL Plot", f"No XFOIL data found for {airfoil_name} at Reynolds={reynolds}, Alpha={alpha}.")


class PointEditDialog(QDialog):
    def __init__(self, x, y, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Point Coordinates")
        layout = QFormLayout()

        self.x_spinbox = QDoubleSpinBox()
        self.x_spinbox.setValue(x)
        layout.addRow("X:", self.x_spinbox)

        self.y_spinbox = QDoubleSpinBox()
        self.y_spinbox.setValue(y)
        layout.addRow("Y:", self.y_spinbox)

        buttons = QPushButton("OK")
        buttons.clicked.connect(self.accept)
        layout.addRow(buttons)

        self.setLayout(layout)

    def get_values(self):
        return self.x_spinbox.value(), self.y_spinbox.value()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = AirfoilViewer()
    viewer.show()
    sys.exit(app.exec())
