import os

from DASC500.classes.DataAnalysis import DataAnalysis
from DASC500.utilities.get_top_level_module import get_top_level_module_path

data = DataAnalysis(file=os.path.join(get_top_level_module_path(), '../..', "data", "DASC512", 'Lessons', 'Lesson5', "cars.csv"))
data.print_stats()