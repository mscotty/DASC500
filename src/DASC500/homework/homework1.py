import os

from DASC500.classes.DataAnalysis import DataAnalysis


file_names = ['CommuteTimes.csv', 'JoblessRate.csv']
folder = r'D:\Mitchell\School\2025 Winter\DASC500\Homework\1'
bin_options = ['Freedman-Diaconis', 'Square Root', 'Sturges']
obj = {file_name: {} for file_name in file_names}
for file_name in file_names:
    data_obj = DataAnalysis(os.path.join(folder, file_name))
    data_obj.print_stats()
    for bin_option in bin_options:
        data_obj.plot_histograms(binning_method=bin_option, output_dir=folder, use_bin_width=True)
    if file_name == "JoblessRate.csv":
        print(data_obj.calculate_pearson_corr_coeff('Jobless_rate', 'Delinquent_Loans'))
    obj[file_name] = data_obj

