import os

import numpy as np
import matplotlib.pyplot as plt

from DASC500.utilities.get_top_level_module import get_top_level_module_path
from DASC500.classes.DataAnalysis import DataAnalysis
from DASC500.formulas.statistics.calculate_mean import calculate_mean
from DASC500.formulas.statistics.calculate_sample_variance import calculate_sample_variance

folder = get_top_level_module_path()
output_folder = os.path.join(folder, "../../outputs/homework3")
output_logger = os.path.join(output_folder, 'homework3_results.txt')
file = os.path.join(folder, '../../data/homework3/Advertising.csv')
data_obj = DataAnalysis(file)
important_col_names = data_obj.df.select_dtypes(include=[np.number])
important_col_names = [val for val in important_col_names if 'Unnamed' not in val]

# Store results from both problem 6 and problem 7
means_problem_6 = {col: [] for col in important_col_names}
var_problem_6 = {col: [] for col in important_col_names}
means_problem_7 = {col: [] for col in important_col_names}
var_problem_7 = {col: [] for col in important_col_names}

def problem_6_and_7_helper(runs=3, 
                           sample_size=30, 
                           confidence_percent=0.95, 
                           alpha=0.05, 
                           pop_mean=30.55,
                           is_problem_6=True):
    """Runs multiple iterations of sampling, calculating stats, and plotting trends."""
    
    run_indices = list(range(1, runs + 1))  # X-axis ticks (integer run numbers)

    for idx in range(runs):
        data_obj.downsample_dataframe(sample_size)
        data_obj.calculate_stats()

        with open(output_logger, 'a+') as f:
            f.write(f"\nRepeating the confidence interval and hypothesis #{idx + 1}\n")

        data_obj.confidence_intervals(confidence_percent)
        data_obj.print_confidence_intervals(output_logger, col_names=important_col_names)

        for column in important_col_names:
            mean_val = data_obj.num_headers[column]['mean']
            var_val = data_obj.num_headers[column]['sample_variance']

            # Store results in the appropriate dictionary
            if is_problem_6:
                means_problem_6[column].append(mean_val)
                var_problem_6[column].append(var_val)
            else:
                means_problem_7[column].append(mean_val)
                var_problem_7[column].append(var_val)

            units = "thousands of items" if 'Sales' in column else "thousands of dollars"
            output_str = f"\nFor {column}:\n"
            output_str += f"Mean={mean_val} [{units}]\n"
            output_str += f"Sample Variance={var_val} [{units}]^2\n"

            with open(output_logger, 'a+', encoding="utf-8") as f:
                f.write(output_str)

def plot_results():
    """Plots mean and variance from both problems 6 and 7 for comparison."""
    for column in important_col_names:
        plt.figure(figsize=(10, 4))

        # Plot Mean
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(means_problem_6[column]) + 1), means_problem_6[column], 
                 marker='o', linestyle='-', label='Problem 6 Mean', color='b')
        plt.plot(range(1, len(means_problem_7[column]) + 1), means_problem_7[column], 
                 marker='s', linestyle='--', label='Problem 7 Mean', color='g')
        plt.xlabel('Run #')
        plt.ylabel('Calculated Mean')
        plt.title(f'Mean Evolution for {column}')
        plt.xticks(range(1, max(len(means_problem_6[column]), len(means_problem_7[column])) + 1))  # Integer x-axis
        plt.legend()
        plt.grid(True)

        # Plot Variance
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(var_problem_6[column]) + 1), var_problem_6[column], 
                 marker='o', linestyle='-', label='Problem 6 Variance', color='b')
        plt.plot(range(1, len(var_problem_7[column]) + 1), var_problem_7[column], 
                 marker='s', linestyle='--', label='Problem 7 Variance', color='g')
        plt.xlabel('Run #')
        plt.ylabel('Sample Variance')
        plt.title(f'Variance Evolution for {column}')
        plt.xticks(range(1, max(len(var_problem_6[column]), len(var_problem_7[column])) + 1))  # Integer x-axis
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f'{column}_mean_variance_comparison.png'))
        plt.close()  # Close figure after saving

def problem_1():
    output_str = (
            f"Problem 1:\n"
            f"See .../classes/DataAnalysis.py\n"
        )
    with open(output_logger, 'w+', encoding="utf-8") as f:
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
        f"Histogram plots created here:\n{output_folder}\n"
    )
    with open(output_logger, 'a+') as f:
        f.write(output_str)

def problem_3():
    output_str = f"\nProblem 3:\n"
    data_obj.calculate_stats()
    means = []
    samp_var = []
    for key in data_obj.num_headers.keys():
        if 'Unnamed' in key:
            continue
        elif 'Sales' in key:
            units = "thousands of items"
        else:
            units = "thousands of dollars"
        means.append(calculate_mean(data_obj.df[key]))
        samp_var.append(calculate_sample_variance(data_obj.df[key]))
        output_str += f"\n{key}:\n"
        output_str += f"Manual Results:\n"
        output_str += f"Mean={means[-1]} [{units}]\n"
        output_str += f"Sample Variance={samp_var[-1]} [{units}]^2\n"
        output_str += f"Implicit Function Results:\n"
        output_str += f"Mean={data_obj.num_headers[key]['mean']} [{units}]\n"
        output_str += f"Sample Variance={data_obj.num_headers[key]['sample_variance']} [{units}]^2\n"
    with open(output_logger, 'a+') as f:
        f.write(output_str)
    
def problem_4():
    confidence_percent = 0.95
    data_obj.confidence_intervals(confidence_percent)
    with open(output_logger, 'a+') as f:
        f.write('\nProblem 4:\n')
    data_obj.print_confidence_intervals(output_logger, col_names=important_col_names)

def problem_5():
    with open(output_logger, 'a+') as f:
        f.write(f'\nProblem 5:\n')
    
    alpha = 0.05
    pop_mean = 30.55
    for column in data_obj.df.select_dtypes(include=[np.number]):
        if 'Unnamed' in column:
            continue
        result = data_obj.hypothesis_test(column, alpha=alpha, mu_0=pop_mean)
        with open(output_logger, 'a+', encoding="utf-8") as f:
            f.write(f'For {column}:\n')
            for key, value in result.items():
                f.write(f"{key}: {value}\n")

def problem_6():
    with open(output_logger, 'a+') as f:
        f.write(f'\nProblem 6:\n')
    
    problem_6_and_7_helper(runs=3, 
                           sample_size=30, 
                           confidence_percent=0.95, 
                           alpha=0.05, 
                           pop_mean=30.55,
                           is_problem_6=True)

def problem_7():
    with open(output_logger, 'a+') as f:
        f.write(f'\nProblem 7:\n')
    
    problem_6_and_7_helper(runs=3, 
                           sample_size=100, 
                           confidence_percent=0.95, 
                           alpha=0.05, 
                           pop_mean=30.55,
                           is_problem_6=False)

def problem_8():
    plot_results()  # Generate and save plots after problem 7 completes
    with open(output_logger, 'a+') as f:
        f.write(f'\nProblem 8:\n')
        f.write(f'For assistance, please reference the plots generated here:\n{output_folder}\n')

if __name__ == "__main__":
    problem_1()
    problem_2()
    problem_3()
    problem_4()
    problem_5()
    problem_6()
    problem_7()
    problem_8()