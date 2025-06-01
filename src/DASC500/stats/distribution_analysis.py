import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import logging
from statsmodels.graphics.gofplots import qqplot


def test_distribution_fit(data, dist_name='norm', alpha=0.05):
    """
    Tests if data fits a specified distribution using appropriate statistical tests.
    
    Parameters:
    -----------
    data : array-like
        Data to test
    dist_name : str, default='norm'
        Distribution to test against ('norm', 'uniform', 'expon', etc.)
    alpha : float, default=0.05
        Significance level
        
    Returns:
    --------
    dict
        Dictionary with test results
    """
    # Drop NaN values
    clean_data = np.array(data.dropna())
    
    if len(clean_data) < 3:
        return {
            'test_name': None,
            'statistic': None,
            'p_value': None,
            'fits_distribution': None
        }
    
    # Select appropriate test based on distribution
    if dist_name == 'norm':
        # For normal distribution: Shapiro-Wilk test
        if len(clean_data) < 5000:  # Shapiro-Wilk works best for n < 5000
            stat, p_value = stats.shapiro(clean_data)
            test_name = 'Shapiro-Wilk'
        else:
            # For larger samples: D'Agostino's K^2 test
            stat, p_value = stats.normaltest(clean_data)
            test_name = "D'Agostino's K^2"
    elif dist_name == 'uniform':
        # For uniform distribution: Kolmogorov-Smirnov test
        stat, p_value = stats.kstest(clean_data, 'uniform', 
                                    args=(clean_data.min(), clean_data.max() - clean_data.min()))
        test_name = 'Kolmogorov-Smirnov'
    elif dist_name == 'expon':
        # For exponential distribution: Kolmogorov-Smirnov test
        stat, p_value = stats.kstest(clean_data, 'expon', 
                                    args=(0, clean_data.mean()))
        test_name = 'Kolmogorov-Smirnov'
    else:
        # Default to Anderson-Darling test
        result = stats.anderson(clean_data, dist_name)
        stat = result.statistic
        # Find the critical value for our alpha
        critical_values = result.critical_values
        significance_levels = [15, 10, 5, 2.5, 1]  # Default levels in stats.anderson
        
        # Find the closest significance level to our alpha
        closest_idx = np.abs(np.array(significance_levels) - alpha*100).argmin()
        critical_value = critical_values[closest_idx]
        
        # Determine if distribution fits based on critical value
        p_value = None  # Anderson test doesn't return p-value
        fits_distribution = stat < critical_value
        
        return {
            'test_name': 'Anderson-Darling',
            'statistic': stat,
            'critical_value': critical_value,
            'significance_level': significance_levels[closest_idx]/100,
            'fits_distribution': fits_distribution
        }
    
    # Determine if distribution fits based on p-value
    fits_distribution = p_value > alpha
    
    return {
        'test_name': test_name,
        'statistic': stat,
        'p_value': p_value,
        'fits_distribution': fits_distribution
    }


def create_distribution_histogram(data, column_name, dist_name='norm', bins=30, 
                                 fit_curve=True, output_dir=None, show_plot=True):
    """
    Creates a histogram with optional distribution fit curve.
    
    Parameters:
    -----------
    data : array-like
        Data to plot
    column_name : str
        Name of the column/variable
    dist_name : str, default='norm'
        Distribution to fit ('norm', 'uniform', 'expon', etc.)
    bins : int, default=30
        Number of bins for histogram
    fit_curve : bool, default=True
        Whether to overlay a fitted distribution curve
    output_dir : str, optional
        Directory to save the plot
    show_plot : bool, default=True
        Whether to display the plot
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    # Drop NaN values
    clean_data = data.dropna()
    
    if len(clean_data) < 2:
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histogram with KDE
    sns.histplot(clean_data, bins=bins, kde=True, ax=ax)
    
    # Fit and plot distribution if requested
    if fit_curve:
        x = np.linspace(clean_data.min(), clean_data.max(), 1000)
        
        if dist_name == 'norm':
            # Fit normal distribution
            mu, sigma = stats.norm.fit(clean_data)
            y = stats.norm.pdf(x, mu, sigma)
            label = f'Normal (μ={mu:.2f}, σ={sigma:.2f})'
        elif dist_name == 'uniform':
            # Fit uniform distribution
            a, b = clean_data.min(), clean_data.max()
            y = stats.uniform.pdf(x, a, b-a)
            label = f'Uniform ({a:.2f}, {b:.2f})'
        elif dist_name == 'expon':
            # Fit exponential distribution
            loc, scale = stats.expon.fit(clean_data)
            y = stats.expon.pdf(x, loc, scale)
            label = f'Exponential (λ={1/scale:.4f})'
        else:
            # Try to fit the specified distribution
            try:
                dist = getattr(stats, dist_name)
                params = dist.fit(clean_data)
                y = dist.pdf(x, *params)
                label = f'{dist_name.capitalize()} fit'
            except:
                fit_curve = False
        
        if fit_curve:
            ax.plot(x, y, 'r-', linewidth=2, label=label)
            ax.legend()
    
    # Set labels and title
    ax.set_title(f'Distribution of {column_name}')
    ax.set_xlabel(column_name)
    ax.set_ylabel('Frequency')
    
    # Save plot if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        safe_col_name = column_name.replace('/', '_').replace('\\', '_')
        plot_path = os.path.join(output_dir, f'dist_hist_{safe_col_name}.png')
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    # Show or close plot
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def create_qq_plot(data, column_name, dist_name='norm', output_dir=None, show_plot=True):
    """
    Creates a Q-Q plot to assess if data follows a specified distribution.
    
    Parameters:
    -----------
    data : array-like
        Data to plot
    column_name : str
        Name of the column/variable
    dist_name : str, default='norm'
        Distribution to test against ('norm', 'uniform', 'expon', etc.)
    output_dir : str, optional
        Directory to save the plot
    show_plot : bool, default=True
        Whether to display the plot
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    # Drop NaN values
    clean_data = data.dropna()
    
    if len(clean_data) < 3:
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Create Q-Q plot
    if dist_name == 'norm':
        # Use statsmodels for better normal Q-Q plot
        qqplot(clean_data, line='s', ax=ax)
        ax.set_title(f'Normal Q-Q Plot for {column_name}')
    else:
        # For other distributions, use scipy's probplot
        try:
            stats.probplot(clean_data, dist=dist_name, plot=ax)
            ax.set_title(f'{dist_name.capitalize()} Q-Q Plot for {column_name}')
        except:
            # Fallback to normal Q-Q plot
            stats.probplot(clean_data, dist='norm', plot=ax)
            ax.set_title(f'Normal Q-Q Plot for {column_name} (fallback)')
    
    # Save plot if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        safe_col_name = column_name.replace('/', '_').replace('\\', '_')
        plot_path = os.path.join(output_dir, f'qq_plot_{safe_col_name}.png')
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    # Show or close plot
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def create_probability_plot(data, column_name, output_dir=None, show_plot=True):
    """
    Creates a probability plot (ECDF) to visualize the empirical distribution.
    
    Parameters:
    -----------
    data : array-like
        Data to plot
    column_name : str
        Name of the column/variable
    output_dir : str, optional
        Directory to save the plot
    show_plot : bool, default=True
        Whether to display the plot
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    # Drop NaN values
    clean_data = data.dropna()
    
    if len(clean_data) < 2:
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create ECDF plot
    sorted_data = np.sort(clean_data)
    ecdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    
    ax.step(sorted_data, ecdf, where='post', label='ECDF')
    
    # Add normal CDF for comparison
    mu, sigma = np.mean(clean_data), np.std(clean_data)
    x = np.linspace(sorted_data.min(), sorted_data.max(), 1000)
    ax.plot(x, stats.norm.cdf(x, mu, sigma), 'r-', label=f'Normal CDF (μ={mu:.2f}, σ={sigma:.2f})')
    
    # Set labels and title
    ax.set_title(f'Empirical Cumulative Distribution for {column_name}')
    ax.set_xlabel(column_name)
    ax.set_ylabel('Cumulative Probability')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Save plot if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        safe_col_name = column_name.replace('/', '_').replace('\\', '_')
        plot_path = os.path.join(output_dir, f'ecdf_{safe_col_name}.png')
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    # Show or close plot
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def compare_distributions(data, column_name, dist_list=['norm', 'uniform', 'expon', 'lognorm'], 
                         output_dir=None, show_plot=True):
    """
    Compares how well data fits multiple distributions.
    
    Parameters:
    -----------
    data : array-like
        Data to analyze
    column_name : str
        Name of the column/variable
    dist_list : list of str, default=['norm', 'uniform', 'expon', 'lognorm']
        Distributions to compare
    output_dir : str, optional
        Directory to save the plot
    show_plot : bool, default=True
        Whether to display the plot
        
    Returns:
    --------
    tuple
        (figure, best_fit_distribution)
    """
    # Drop NaN values
    clean_data = data.dropna()
    
    if len(clean_data) < 3:
        return None, None
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot histogram
    hist_kwargs = {'alpha': 0.4, 'density': True, 'bins': 'auto'}
    ax.hist(clean_data, **hist_kwargs, label='Data')
    
    # Create x range for plotting
    x = np.linspace(clean_data.min(), clean_data.max(), 1000)
    
    # Test and plot each distribution
    results = []
    for dist_name in dist_list:
        try:
            # Fit distribution
            dist = getattr(stats, dist_name)
            params = dist.fit(clean_data)
            
            # Plot PDF
            pdf = dist.pdf(x, *params)
            ax.plot(x, pdf, label=f'{dist_name.capitalize()}')
            
            # Test fit
            test_result = test_distribution_fit(clean_data, dist_name)
            
            # Add results
            if 'p_value' in test_result:
                results.append({
                    'distribution': dist_name,
                    'test': test_result['test_name'],
                    'statistic': test_result['statistic'],
                    'p_value': test_result['p_value'],
                    'fits_distribution': test_result['fits_distribution']
                })
            else:
                results.append({
                    'distribution': dist_name,
                    'test': test_result['test_name'],
                    'statistic': test_result['statistic'],
                    'critical_value': test_result.get('critical_value'),
                    'fits_distribution': test_result['fits_distribution']
                })
        except Exception as e:
            # Skip distributions that fail to fit
            pass
    
    # Set labels and title
    ax.set_title(f'Distribution Comparison for {column_name}')
    ax.set_xlabel(column_name)
    ax.set_ylabel('Probability Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save plot if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        safe_col_name = column_name.replace('/', '_').replace('\\', '_')
        plot_path = os.path.join(output_dir, f'dist_comparison_{safe_col_name}.png')
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    # Show or close plot
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    # Determine best fit distribution
    if results:
        results_df = pd.DataFrame(results)
        
                # For tests with p-values, higher is better
        p_value_results = results_df[results_df['p_value'].notna()]
        if not p_value_results.empty:
            best_fit = p_value_results.loc[p_value_results['p_value'].idxmax()]
            best_fit_dist = best_fit['distribution']
        else:
            # For tests without p-values (like Anderson-Darling), lower statistic is better
            best_fit = results_df.loc[results_df['statistic'].idxmin()]
            best_fit_dist = best_fit['distribution']
    else:
        best_fit_dist = None
    
    return fig, best_fit_dist


def calculate_distribution_moments(data):
    """
    Calculates the four moments of a distribution: mean, variance, skewness, and kurtosis.
    
    Parameters:
    -----------
    data : array-like
        Data to analyze
        
    Returns:
    --------
    dict
        Dictionary with the four moments
    """
    # Drop NaN values
    clean_data = data.dropna()
    
    if len(clean_data) < 3:
        return {
            'mean': np.nan,
            'variance': np.nan,
            'skewness': np.nan,
            'kurtosis': np.nan
        }
    
    # Calculate moments
    mean = np.mean(clean_data)
    variance = np.var(clean_data)
    skewness = stats.skew(clean_data)
    kurtosis = stats.kurtosis(clean_data)  # Fisher's definition (normal = 0)
    
    return {
        'mean': mean,
        'variance': variance,
        'skewness': skewness,
        'kurtosis': kurtosis
    }


def analyze_distributions(df, columns=None, distributions=['norm', 'uniform', 'expon', 'lognorm'],
                         output_dir=None, plot_hist=True, plot_qq=True, plot_ecdf=True, 
                         compare_dists=True, show_plots=True):
    """
    Comprehensive distribution analysis for numerical columns.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The DataFrame to analyze
    columns : list of str, optional
        Specific columns to analyze (default: all numerical columns)
    distributions : list of str, default=['norm', 'uniform', 'expon', 'lognorm']
        Distributions to test and compare
    output_dir : str, optional
        Directory to save output files and plots
    plot_hist : bool, default=True
        Whether to create histogram plots
    plot_qq : bool, default=True
        Whether to create Q-Q plots
    plot_ecdf : bool, default=True
        Whether to create ECDF plots
    compare_dists : bool, default=True
        Whether to compare multiple distributions
    show_plots : bool, default=True
        Whether to display plots
        
    Returns:
    --------
    dict
        Dictionary containing analysis results
    """
    # Set up logging
    logger = logging.getLogger('distribution_analysis')
    logger.setLevel(logging.INFO)
    
    # Create console handler if not already present
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Add file handler if output_dir is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(output_dir, 'distribution_analysis.log'))
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    logger.info("\n=== Distribution Analysis ===")
    
    # Select numerical columns if not specified
    if columns is None:
        num_df = df.select_dtypes(include=np.number)
        columns = num_df.columns.tolist()
    else:
        # Filter to only include numerical columns that exist
        columns = [col for col in columns if col in df.columns and 
                  np.issubdtype(df[col].dtype, np.number)]
    
    if not columns:
        logger.info("No numerical columns for distribution analysis.")
        return {}
    
    logger.info(f"Analyzing distributions for {len(columns)} columns")
    
    # Initialize results dictionary
    results = {
        'distribution_tests': {},
        'moments': {},
        'best_fit_distributions': {},
        'plots': {}
    }
    
    # Create plots directory if needed
    plots_dir = os.path.join(output_dir, 'distribution_plots') if output_dir else None
    
    # Analyze each column
    for col in columns:
        logger.info(f"\nAnalyzing distribution for: {col}")
        
        # Get column data
        data = df[col].dropna()
        
        if len(data) < 3:
            logger.info(f"  Skipping {col}: Insufficient data (n={len(data)})")
            continue
        
        # Calculate distribution moments
        moments = calculate_distribution_moments(data)
        results['moments'][col] = moments
        
        logger.info(f"  Moments: Mean={moments['mean']:.4f}, Variance={moments['variance']:.4f}")
        logger.info(f"           Skewness={moments['skewness']:.4f}, Kurtosis={moments['kurtosis']:.4f}")
        
        # Test distributions
        dist_tests = {}
        for dist_name in distributions:
            test_result = test_distribution_fit(data, dist_name)
            dist_tests[dist_name] = test_result
            
            # Log test results
            if 'p_value' in test_result:
                logger.info(f"  {test_result['test_name']} test for {dist_name.capitalize()}: "
                           f"p-value={test_result.get('p_value', 'N/A'):.6f} - "
                           f"{'Fits' if test_result['fits_distribution'] else 'Does not fit'}")
            else:
                logger.info(f"  {test_result['test_name']} test for {dist_name.capitalize()}: "
                           f"statistic={test_result.get('statistic', 'N/A'):.6f}, "
                           f"critical={test_result.get('critical_value', 'N/A'):.6f} - "
                           f"{'Fits' if test_result['fits_distribution'] else 'Does not fit'}")
        
        results['distribution_tests'][col] = dist_tests
        
        # Create plots
        col_plots = {}
        
        # Histogram with distribution fit
        if plot_hist:
            for dist_name in distributions:
                if dist_tests[dist_name]['fits_distribution']:
                    # Use the first distribution that fits
                    hist_fig = create_distribution_histogram(
                        data, col, dist_name=dist_name, 
                        output_dir=plots_dir, show_plot=show_plots
                    )
                    col_plots['histogram'] = hist_fig
                    break
            else:
                # If no distribution fits, use normal as default
                hist_fig = create_distribution_histogram(
                    data, col, dist_name='norm', 
                    output_dir=plots_dir, show_plot=show_plots
                )
                col_plots['histogram'] = hist_fig
        
        # Q-Q plot
        if plot_qq:
            # Use the best fitting distribution for Q-Q plot
            best_dist = 'norm'  # Default
            for dist_name in distributions:
                if dist_tests[dist_name]['fits_distribution']:
                    best_dist = dist_name
                    break
            
            qq_fig = create_qq_plot(
                data, col, dist_name=best_dist,
                output_dir=plots_dir, show_plot=show_plots
            )
            col_plots['qq_plot'] = qq_fig
        
        # ECDF plot
        if plot_ecdf:
            ecdf_fig = create_probability_plot(
                data, col, output_dir=plots_dir, show_plot=show_plots
            )
            col_plots['ecdf'] = ecdf_fig
        
        # Distribution comparison
        if compare_dists and len(distributions) > 1:
            comp_fig, best_fit = compare_distributions(
                data, col, dist_list=distributions,
                output_dir=plots_dir, show_plot=show_plots
            )
            col_plots['comparison'] = comp_fig
            results['best_fit_distributions'][col] = best_fit
            
            if best_fit:
                logger.info(f"  Best fit distribution: {best_fit.capitalize()}")
        
        results['plots'][col] = col_plots
    
    # Create summary of best fit distributions
    if results['best_fit_distributions']:
        dist_counts = {}
        for dist in results['best_fit_distributions'].values():
            if dist:
                dist_counts[dist] = dist_counts.get(dist, 0) + 1
        
        logger.info("\nSummary of best fit distributions:")
        for dist, count in sorted(dist_counts.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {dist.capitalize()}: {count} columns ({count/len(columns)*100:.1f}%)")
    
    # Save distribution test results to CSV if output directory is provided
    if output_dir and results['distribution_tests']:
        # Create a DataFrame with test results
        test_rows = []
        for col, tests in results['distribution_tests'].items():
            for dist_name, test_result in tests.items():
                row = {
                    'column': col,
                    'distribution': dist_name,
                    'test_name': test_result.get('test_name'),
                    'statistic': test_result.get('statistic'),
                    'p_value': test_result.get('p_value'),
                    'fits_distribution': test_result.get('fits_distribution')
                }
                test_rows.append(row)
        
        test_df = pd.DataFrame(test_rows)
        test_path = os.path.join(output_dir, 'distribution_tests.csv')
        test_df.to_csv(test_path, index=False)
        logger.info(f"\nDistribution test results saved to: {test_path}")
        
        # Create a DataFrame with moments
        moments_df = pd.DataFrame(results['moments']).T
        moments_path = os.path.join(output_dir, 'distribution_moments.csv')
        moments_df.to_csv(moments_path)
        logger.info(f"Distribution moments saved to: {moments_path}")
    
    logger.info("\n=== Distribution Analysis Complete ===")
    
    return results

