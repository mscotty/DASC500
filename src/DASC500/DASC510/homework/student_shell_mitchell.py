import os

from DASC500.DASC510.homework import data_util2
from DASC500.utilities.print.redirect_print import redirect_print
from DASC500.utilities.get_top_level_module import get_top_level_module_path

output_logger = os.path.join(
    get_top_level_module_path(), "../../outputs/DASC510/homework4/output_logger.txt"
)
redirect_print(output_logger, also_to_stdout=True)

# 1(1 point) Assign a variable named name a string object with text representing your name.
name = "Mitchell Scott"
print(name)

# 2(2 points) Assign a variable measurement_1 as the returned object of the Measurement's Parse static method when passed “730.3 ft”
measurement_1 = data_util2.Measurement.Parse("730.3 ft")
print(measurement_1)

# 3(2 points) Assign a variable measurement_2 as the returned object of the Measurement's Parse static method when passed “1.5 km”
measurement_2 = data_util2.Measurement.Parse("1.5 km")
print(measurement_2)

# 4(2 points) Assign a variable measurement_3 as the result of adding measurement_1 and measurement_2 together
# using the ‘+’ operator
measurement_3 = measurement_1 + measurement_2
print(measurement_3)

# 5(2 points) Assign a variable measurement_4 as the result of multiplying measurement_1 by 3
# using the multiplication, ‘*’, operator
measurement_4 = measurement_1 * 3
print(measurement_4)

# 6(2 points) Assign a variable measurement_5 as the result of subtracting measurement_1 from measurement_2
# using the subtraction, ‘-’, operator
measurement_5 = measurement_2 - measurement_1
print(measurement_5)

# 7(2 points) Use the unit_of_measurement_provider variable within the data_util2 module to lookup the
# UnitOfMeasurement object representing meters. Assign the returned UnitOfMeasurement object to a variable meters_unit
meters_unit = data_util2.unit_of_measurement_provider.lookup_unit_for_unit_label("m")
print(meters_unit)

# 8(2 points) Assign a variable measurement_6 as the returned object of the Measurement's Parse static
# method when passed “53.4 ft”. Then call the convert_units method on this Measurement object passing
# in the UnitOfMeasurement object representing meters, meters_unit.
measurement_6 = data_util2.Measurement.Parse("53.4 ft")
measurement_6.convert_units(meters_unit)
print(measurement_6)

# 9(2 points) Use the unit_of_measurement_provider variable within the data_util2 module to lookup
# the UnitOfMeasurement object representing inches. Assign the returned UnitOfMeasurement  object
# to a variable inches_unit
inches_unit = data_util2.unit_of_measurement_provider.lookup_unit_for_unit_label("in")
print(inches_unit)

# 10(2 points) Assign a variable measurement_7 as the returned object of the Measurement's Parse static
# method when passed “42.0 ft”. Then call the convert_units method on this Measurement object passing
# in the UnitOfMeasurement object representing inches, inches_unit.
measurement_7 = data_util2.Measurement.Parse("42.0 ft")
measurement_7.convert_units(inches_unit)
print(measurement_7)
