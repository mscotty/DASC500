import os

from DASC500.classes.DataAnalysis import DataAnalysis
from DASC500.utilities.get_top_level_module import get_top_level_module_path

data_filename = 'Logan.csv'
folder = get_top_level_module_path()
data_file = os.path.join(folder,  "../../data/homework2", data_filename)
output_folder = os.path.join(folder, "../../outputs/homework2")

data_obj = DataAnalysis(data_file)
data_obj.plot_stacked_bar_chart_horizontal(x_column=None, y_column='City', output_dir=output_folder, x_axis_name='Percentage of Time (%)')
data_obj.plot_clustered_bar_chart_horizontal(x_column=None, y_column='City', output_dir=output_folder, x_axis_name='Percentage of Time (%)')
data_obj.plot_individual_bar_charts(x_column='City', y_column=None, output_dir=output_folder, y_axis_name='Percentage of Time(%)')
data_obj.plot_line_chart(x_column='City', y_column='Idle', output_dir=output_folder)
data_obj.plot_heatmap(output_dir=output_folder, title_name='Heatmap of Values', x_axis_name='Categories', y_axis_name='Cities')
data_obj.plot_radar_chart(output_dir=output_folder, title_name='Radar Chart of Categories by City')