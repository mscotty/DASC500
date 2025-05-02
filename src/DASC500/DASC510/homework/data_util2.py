import os
import csv
from DASC500.utilities.get_top_level_module import get_top_level_module_path

measure_file = os.path.join(
    get_top_level_module_path(), "../../data/DASC510/homework4/length_measurements.csv"
)


class UnitOfMeasurement:
    def __init__(
        self,
        name,
        shorthand,
        scaler_to_base=1,
        offset_to_base=0,
        other_acceptable_labels=None,
    ):
        self.name = name
        self.shorthand = shorthand
        self.scaler_to_base = scaler_to_base
        self.offset_to_base = offset_to_base
        self.other_acceptable_labels = other_acceptable_labels

    def convert_value_to_base(self, value):
        return self.scaler_to_base * value + self.offset_to_base

    def convert_value_from_base(self, value):
        return (1.0 / self.scaler_to_base) * value - self.offset_to_base

    def format_value(self, value):
        return "{} {}".format(value, self.shorthand)

    def __str__(self):
        labels = (
            f" (also: {', '.join(self.other_acceptable_labels)})"
            if self.other_acceptable_labels
            else ""
        )
        return f"Unit: {self.name} ({self.shorthand}){labels}, Scaler: {self.scaler_to_base}, Offset: {self.offset_to_base}"


class UnitOfMeasurementProvider:
    def __init__(self):
        self.default_unit: UnitOfMeasurement
        self.unit_label_map = {}

    def get_base_unit(self) -> UnitOfMeasurement:
        return self.default_unit

    def lookup_unit_for_unit_label(self, unit_label):
        try:
            return self.unit_label_map[unit_label]
        except KeyError:
            raise ValueError(f"Unit label '{unit_label}' not found.")

    def register_unit(self, unit_of_measurement, make_default=False):
        if make_default:
            self.default_unit = unit_of_measurement

        self.unit_label_map[unit_of_measurement.name] = unit_of_measurement
        self.unit_label_map[unit_of_measurement.shorthand] = unit_of_measurement
        if unit_of_measurement.other_acceptable_labels is not None:
            for other_label in unit_of_measurement.other_acceptable_labels:
                self.unit_label_map[other_label] = unit_of_measurement

    def load_from_csv_file(self, unit_of_measurement_file_path):
        with open(unit_of_measurement_file_path) as csv_file:
            reader = csv.reader(csv_file, delimiter=",")
            header = next(reader, None)  # Read the header row
            if not header or len(header) < 5:
                raise ValueError(
                    f"Invalid CSV file format: '{unit_of_measurement_file_path}'. Expected at least 5 columns (name, shorthand, scaler_to_base, offset_to_base, other_acceptable_label)."
                )

            first = True
            for row_number, row in enumerate(
                reader, start=2
            ):  # Start row count at 2 (after header)
                if len(row) < 5:
                    raise ValueError(
                        f"Missing data in CSV file '{unit_of_measurement_file_path}' at row {row_number}. Expected at least 5 columns."
                    )
                try:
                    name = row[0]
                    shorthand = row[1]
                    scaler_to_base = float(row[2])
                    offset_to_base = float(row[3])
                    other_acceptable_label_str = row[4].strip()
                    other_acceptable_labels = (
                        tuple(
                            label.strip()
                            for label in other_acceptable_label_str.split(";")
                            if label.strip()
                        )
                        if other_acceptable_label_str
                        else None
                    )

                    unit = UnitOfMeasurement(
                        name,
                        shorthand,
                        scaler_to_base,
                        offset_to_base,
                        other_acceptable_labels=other_acceptable_labels,
                    )
                    self.register_unit(unit, first)
                    first = False
                except ValueError as e:
                    raise ValueError(
                        f"Invalid data in CSV file '{unit_of_measurement_file_path}' at row {row_number}: {e}"
                    )
                except Exception as e:
                    raise Exception(
                        f"An unexpected error occurred while processing CSV file '{unit_of_measurement_file_path}' at row {row_number}: {e}"
                    )


class Measurement:
    def __init__(self, value, unit_of_measurement: UnitOfMeasurement):
        self.value = value
        self.unit_of_measurement = unit_of_measurement

    def __add__(self, rhs):
        base_unit = unit_of_measurement_provider.get_base_unit()
        lhs_value = self.unit_of_measurement.convert_value_to_base(self.value)
        rhs_value = rhs.unit_of_measurement.convert_value_to_base(rhs.value)
        return Measurement(lhs_value + rhs_value, base_unit)

    def __sub__(self, rhs):
        base_unit = unit_of_measurement_provider.get_base_unit()
        lhs_value = self.unit_of_measurement.convert_value_to_base(self.value)
        rhs_value = rhs.unit_of_measurement.convert_value_to_base(rhs.value)
        return Measurement(lhs_value - rhs_value, base_unit)

    def __mul__(self, rhs):
        return Measurement(self.value * rhs, self.unit_of_measurement)

    def __rmul__(self, lhs):
        return Measurement(self.value * lhs, self.unit_of_measurement)

    def __str__(self):
        return self.unit_of_measurement.format_value(self.value)

    def convert_units(self, new_unit_of_measurement: UnitOfMeasurement):
        base_value = self.unit_of_measurement.convert_value_to_base(self.value)
        self.value = new_unit_of_measurement.convert_value_from_base(base_value)
        self.unit_of_measurement = new_unit_of_measurement

    @staticmethod
    def Parse(value: str):
        parts = value.split(" ")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid measurement format: '{value}'. Expected 'value unit'."
            )
        try:
            n_value = float(parts[0])
        except ValueError:
            raise ValueError(
                f"Invalid value in measurement: '{parts[0]}'. Must be a number."
            )
        unit_label = parts[1]
        try:
            unit = unit_of_measurement_provider.lookup_unit_for_unit_label(unit_label)
        except KeyError:
            raise ValueError(f"Unknown unit: '{unit_label}'.")
        return Measurement(n_value, unit)


# load current units of measure
unit_of_measurement_provider = UnitOfMeasurementProvider()
unit_of_measurement_provider.load_from_csv_file(measure_file)

# Two examples of utilizing the classes

yard = Measurement.Parse("3 ft")
# print("One yard is {}".format(yard))

meters_unit = unit_of_measurement_provider.lookup_unit_for_unit_label("m")
print("Meters unit: {}".format(meters_unit))

yard = Measurement.Parse("3 ft")
yard.convert_units(meters_unit)
# print("Another way of saying one yard is {}".format(yard))
