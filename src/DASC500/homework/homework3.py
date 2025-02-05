import os

from DASC500.utilities.get_top_level_module import get_top_level_module_path
from DASC500.classes.DataAnalysis import DataAnalysis
from DASC500.formulas.statistics.calculate_mean import calculate_mean
from DASC500.formulas.statistics.calculate_sample_variance import calculate_sample_variance

folder = get_top_level_module_path()
output_folder = os.path.join(folder, "../../outputs/homework3")
output_logger = os.path.join(output_folder, 'homework3_results.txt')
file = os.path.join(folder, '../../data/homework3/Advertising.csv')
data_obj = DataAnalysis(file)

def problem_1():
    output_str = (
            f"Problem 1:\n"
            f"See .../classes/DataAnalysis.py\n"
        )
    with open(output_logger, 'w+') as f:
        f.write(output_str)

def problem_2():
    
    sample_size = 30
    random_seed = 1
    data_obj.downsample_dataframe(sample_size, random_seed)
    data_obj.plot_histograms_per_col(output_dir=output_folder)
    output_str = (
        f"\nProblem 2:\n"
        f"DataAnalysis object downsampled (random seed: {random_seed})\n"
        f"goal: {sample_size}\n"
        f"actual: {len(data_obj.df)}\n"
        f"Histogram plots created here:\n{output_folder}"
    )
    with open(output_logger, 'a+') as f:
        f.write(output_str)

def problem_3():
    output_str = f'\nProblem 3:\n'
    data_obj.calculate_stats()
    means = []
    samp_var = []
    for key in data_obj.num_headers.keys():
        if 'Unnamed' in key:
            continue
        means.append(calculate_mean(data_obj.df[key]))
        samp_var.append(calculate_sample_variance(data_obj.df[key]))
        output_str += f"{key}:\n"
        output_str += f"Manual Results:\n"
        output_str += f"Mean={means[-1]} [sales in thousands of items]\n"
        output_str += f"Sample Variance={samp_var[-1]} [sales in thousands of items]^2\n"
        output_str += f"Implicit Function Results:\n"
        output_str += f"Mean={data_obj.num_headers[key]['mean']} [sales in thousands of items]\n"
        output_str += f"Sample Variance={data_obj.num_headers[key]['sample_variance']} [sales in thousands of items]^2\n"
    with open(output_logger, 'a+') as f:
        f.write(output_str)
    
def problem_4():
    confidence_percent = 0.95
    data_obj.confidence_intervals(confidence_percent)
    with open(output_logger, 'a+') as f:
        f.write('\nProblem 4:\n')
    data_obj.print_confidence_intervals(output_logger)

def problem_5():
    print('Problem 5:')

def problem_6():
    print('Problem 6:')

def problem_7():
    print('Problem 7:')

def problem_8():
    print('Problem 8:')

if __name__ == "__main__":
    problem_1()
    problem_2()
    problem_3()
    problem_4()