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

def problem_6_and_7_helper(runs=3, 
                           sample_size=30, 
                           confidence_percent=0.95, 
                           alpha=0.05, 
                           pop_mean=30.55,):
    """Runs multiple iterations of sampling, calculating stats, and plotting trends."""
    
    run_indices = list(range(1, runs + 1))  # X-axis ticks (integer run numbers)
    output = {}
    
    for idx in range(runs):
        data_obj.downsample_dataframe(sample_size)
        data_obj.calculate_stats()

        with open(output_logger, 'a+') as f:
            f.write(f"\nRepeating the confidence interval and hypothesis #{idx + 1}\n")

        data_obj.confidence_intervals(confidence_percent)
        data_obj.print_confidence_intervals(output_logger, col_names=important_col_names)

        
        for column in important_col_names:
            if column not in output.keys():
                output[column] = {
                    'mean': [], 
                    'sample_variance': [], 
                    'conf_int_mean':[], 
                    'test_stat': []
                }
            mean_val = data_obj.num_headers[column]['mean']
            var_val = data_obj.num_headers[column]['sample_variance']

            # Store results in the appropriate dictionary
            output[column]['mean'].append(mean_val)
            output[column]['sample_variance'].append(var_val)
            output[column]['conf_int_mean'].append(data_obj.conf_interval[column]['mean_CI'])

            units = "thousands of items" if 'Sales' in column else "thousands of dollars"
            output_str = f"\nFor {column}:\n"
            output_str += f"Mean={mean_val} [{units}]\n"
            output_str += f"Sample Variance={var_val} [{units}]^2\n"

            with open(output_logger, 'a+', encoding="utf-8") as f:
                f.write(output_str)
            
            result = data_obj.hypothesis_test(column, alpha=alpha, mu_0=pop_mean)
            output[column]['test_stat'].append(result['Test Statistic (W)'])
            with open(output_logger, 'a+', encoding="utf-8") as f:
                for key, value in result.items():
                    f.write(f"{key}: {value}\n")
    
    return output

"""
def plot_results(dict_in, 
                 keys_in=None, 
                 rows=None, 
                 cols=None, 
                 output_folder=None,
                 filename_prefix="plot",
                 fig=None):
    Plots multiple series from a dictionary in a grid layout.
    
    Arguments:
    - dict_in: Dictionary where keys are data labels and values are lists of values over iterations.
    - keys_in: Subset of keys to plot (default: all keys in dict_in).
    - rows, cols: Grid dimensions (default: auto-calculated based on keys_in).
    - filename_prefix: Prefix for saved plot filenames.
    
    
    if keys_in is None:
        keys_in = list(dict_in.keys())

    num_keys = len(keys_in)

    # Auto-determine rows and cols if not provided
    if rows is None and cols is None:
        rows = int(num_keys ** 0.5)  # Square root approximation
        cols = (num_keys // rows) + (num_keys % rows > 0)  # Ensure all keys fit

    if fig is None:
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        axes = np.array(axes)  # Ensure it's an array even for 1 subplot
        axes = axes.flatten()  # Flatten in case of single row/col cases

    for ind, key in enumerate(keys_in):
        if key not in dict_in or not dict_in[key]:
            continue  # Skip if data is missing

        ax = axes[ind]  # Get subplot axis

        # Get run indices for x-axis (assumes all lists in dict_in[key] are same length)
        runs = list(range(1, len(dict_in[key]) + 1))

        ax.plot(runs, dict_in[key], marker='o', linestyle='-', label=f'{key}', color='b')
        ax.set_xlabel('Run #')
        ax.set_ylabel(key)
        ax.set_title(f'Evolution of {key}')
        ax.set_xticks(runs)  # Ensure x-ticks are integers
        ax.legend()
        ax.grid(True)

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    if output_folder is not None:
        save_path = os.path.join(output_folder, f"{filename_prefix}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Plot saved: {save_path}")
    else:
        return plt"""

def plot_results(output_dict, 
                 feature_name, 
                 output_folder, 
                 filename_prefix="plot"):
    """Plots mean, variance, confidence intervals, and test statistics for a given feature.
    
    Arguments:
    - output_dict: Dictionary where keys are sample sizes (e.g., '15 Samples') and values are stats dicts.
    - feature_name: Name of the feature (e.g., 'TV', 'Radio', etc.) to plot.
    - output_folder: Directory to save the plots.
    - filename_prefix: Prefix for saved plot filenames.
    """

    sample_sizes = list(output_dict.keys())
    colors = ['b', 'g', 'r', 'm']  # Ensure unique colors per sample size
    linestyles = ['-', '-', '-', '--']  # Dashed for n=200
    markers = ['o', 's', '^', 'd']  # Different markers for distinction
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    x_vals = list(range(1, max(len(output_dict[s][feature_name]['mean']) for s in sample_sizes) + 1))
    
    for idx, sample_size in enumerate(sample_sizes):
        data = output_dict[sample_size][feature_name]
        linestyle = '--' if sample_size == '200 Samples' else '-'
        color = colors[idx]
        marker = markers[idx]
        sample_label = f"{sample_size}" 
        
        # Mean Plot
        axes[0].plot(x_vals, np.repeat(data['mean'], len(x_vals)) if sample_size == '200 Samples' else data['mean'], 
                     marker=marker, linestyle=linestyle, color=color, label=sample_label)
        
        # Variance Plot
        axes[1].plot(x_vals, np.repeat(data['sample_variance'], len(x_vals)) if sample_size == '200 Samples' else data['sample_variance'], 
                     marker=marker, linestyle=linestyle, color=color)
        
        # Confidence Interval Plot (both upper and lower bounds)
        """lower, upper = zip(*data['conf_int_mean'])
        axes[2].plot(x_vals, np.repeat(lower, len(x_vals)) if sample_size == '200 Samples' else lower, 
                     marker=marker, linestyle=linestyle, color=color)
        axes[2].plot(x_vals, np.repeat(upper, len(x_vals)) if sample_size == '200 Samples' else upper, 
                     marker=marker, linestyle=linestyle, color=color)
        axes[2].plot(x_vals, np.repeat(data['mean'], len(x_vals)) if sample_size == '200 Samples' else data['mean'], 
                     linestyle='--', color=color, alpha=0.5)  # Dashed mean line"""
        lower_CI, upper_CI = zip(*data['conf_int_mean'])  # Extract lower/upper bounds
        axes[2].fill_between(x_vals, lower_CI, upper_CI, color=color, alpha=0.2)  # Shaded region for CI
        axes[2].plot(x_vals, np.repeat(data['mean'], len(x_vals)) if sample_size == '200 Samples' else data['mean'], linestyle='dashed', color=color)  # Dashed line for mean

        
        # Test Statistic Plot
        axes[3].plot(x_vals, np.repeat(data['test_stat'], len(x_vals)) if sample_size == '200 Samples' else data['test_stat'], 
                     marker=marker, linestyle=linestyle, color=color)
        
    titles = ["Mean Evolution", "Variance Evolution", "Confidence Interval", "Test Statistic Evolution"]
    y_labels = ["Mean", "Sample Variance", "Confidence Interval", "Test Statistic"]
    
    for ax, title, ylabel in zip(axes, titles, y_labels):
        ax.set_title(f"{feature_name} - {title}")
        ax.set_xlabel("Run #")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x_vals)
        ax.grid(True)
    
    axes[0].legend()  # Single legend on mean plot
    plt.tight_layout()
    
    if output_folder:
        plt.savefig(f"{output_folder}/{filename_prefix}_{feature_name}.png")
        plt.close()
    else:
        plt.show()

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
    bin_options = ['Freedman-Diaconis', 'Square Root', 'Sturges']
    for key in data_obj.num_headers.keys():
        if 'Unnamed' in key:
            continue
        elif 'Sales' in key:
            units = "thousands of items"
        else:
            units = "thousands of dollars"
        for bin_option in bin_options:
            data_obj.plot_histograms_per_col(key_in=key,
                                             binning_method=bin_option, 
                                             output_dir=output_folder, 
                                             use_bin_width=True,
                                             x_axis_units=units)
    #data_obj.plot_histograms_per_col(output_dir=output_folder)
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
    
    output = problem_6_and_7_helper(runs=3, 
                                    sample_size=30, 
                                    confidence_percent=0.95, 
                                    alpha=0.05, 
                                    pop_mean=30.55)
    
    return output

def problem_7():
    with open(output_logger, 'a+') as f:
        f.write(f'\nProblem 7:\n')
    
    output = problem_6_and_7_helper(runs=3, 
                                    sample_size=100, 
                                    confidence_percent=0.95, 
                                    alpha=0.05, 
                                    pop_mean=30.55)
    
    return output

def problem_8(output):
    for feature in ['TV', 'Newspaper', 'Radio', 'Sales']:
        plot_results(output_dict=output, 
                     feature_name=feature, 
                     output_folder=output_folder, 
                     filename_prefix="problem8")
    
    with open(output_logger, 'a+') as f:
        f.write(f'\nProblem 8:\n')
        f.write(f'For assistance, please reference the plots generated here:\n{output_folder}\n')

if __name__ == "__main__":
    problem_1()
    problem_2()
    problem_3()
    problem_4()
    problem_5()
    output = {}
    output['15 Samples'] = problem_6_and_7_helper(runs=3, 
                                    sample_size=15, 
                                    confidence_percent=0.95, 
                                    alpha=0.05, 
                                    pop_mean=30.55)
    output['30 Samples'] = problem_6()
    output['100 Samples'] = problem_7()
    output['200 Samples'] = problem_6_and_7_helper(runs=1, 
                                    sample_size=200, 
                                    confidence_percent=0.95, 
                                    alpha=0.05, 
                                    pop_mean=30.55)
    problem_8(output)