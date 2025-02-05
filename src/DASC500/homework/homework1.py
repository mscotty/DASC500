import os

from DASC500.classes.DataAnalysis import DataAnalysis
from DASC500.utilities.get_top_level_module import get_top_level_module_path


file_names = ['CommuteTimes.csv', 'JoblessRate.csv']
folder = get_top_level_module_path()
data_folder = os.path.join(folder,  "../../data/homework1/")
output_folder = os.path.join(folder, "../../outputs/homework1")
bin_options = ['Freedman-Diaconis', 'Square Root', 'Sturges']
obj = {file_name: {} for file_name in file_names}
for file_name in file_names:
    data_obj = DataAnalysis(os.path.join(data_folder, file_name))
    output_file = os.path.join(output_folder, os.path.splitext(file_name)[0] + '_stats.txt')
    data_obj.print_stats(output_file)
    for bin_option in bin_options:
        data_obj.plot_histograms_per_col(binning_method=bin_option, output_dir=output_folder, use_bin_width=True)
    if file_name == "JoblessRate.csv":
        print(data_obj.calculate_pearson_corr_coeff('Jobless_rate', 'Delinquent_Loans'))
    obj[file_name] = data_obj

