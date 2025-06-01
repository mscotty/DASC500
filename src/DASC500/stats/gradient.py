import os
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, signal
from matplotlib.gridspec import GridSpec
from sklearn.linear_model import LinearRegression


def calculate_gradient(data, x=None, method='central', window_size=None):
    """
    Calculates the gradient (derivative) of data with respect to x.
    
    Parameters:
    -----------
    data : array-like
        Data to analyze
    x : array-like, optional
        X-values corresponding to data (if None, uses equally spaced points)
    method : str, default='central'
        Method for gradient calculation ('forward', 'backward', 'central')
    window_size : int, optional
        Window size for smoothed gradient (if None, no smoothing)
        
    Returns:
    --------
    tuple
        (x_values, gradient_values)
    """
    # Convert to numpy arrays
    y = np.array(data)
    
    # Handle x values
    if x is not None:
        x = np.array(x)
    else:
        x = np.arange(len(y))
    
    # Remove NaN values
    valid_mask = ~np.isnan(y) & ~np.isnan(x)
    y = y[valid_mask]
    x = x[valid_mask]
    
    # Sort by x values
    sort_idx = np.argsort(x)
    y = y[sort_idx]
    x = x[sort_idx]
    
    if len(y) < 2:
        return np.array([]), np.array([])
    
    # Calculate gradient based on method
    if method == 'forward':
        # Forward difference: (y[i+1] - y[i]) / (x[i+1] - x[i])
        dx = np.diff(x)
        dy = np.diff(y)
        gradient = dy / dx
        
        # Gradient values correspond to x[:-1]
        x_grad = x[:-1]
    
    elif method == 'backward':
        # Backward difference: (y[i] - y[i-1]) / (x[i] - x[i-1])
        dx = np.diff(x)
        dy = np.diff(y)
        gradient = dy / dx
        
        # Gradient values correspond to x[1:]
        x_grad = x[1:]
    
    elif method == 'central':
        # Central difference: (y[i+1] - y[i-1]) / (x[i+1] - x[i-1])
        if len(y) < 3:
            # Fall back to forward difference if not enough points
            dx = np.diff(x)
            dy = np.diff(y)
            gradient = dy / dx
            x_grad = x[:-1]
        else:
            gradient = np.zeros(len(y) - 2)
            for i in range(len(gradient)):
                gradient[i] = (y[i+2] - y[i]) / (x[i+2] - x[i])
            
            # Gradient values correspond to x[1:-1]
            x_grad = x[1:-1]
    
    else:
        raise ValueError("Method must be 'forward', 'backward', or 'central'")
    
    # Apply smoothing if window_size is provided
    if window_size is not None and window_size > 1 and len(gradient) >= window_size:
        # Use Savitzky-Golay filter for smoothing
        try:
            polyorder = min(3, window_size - 1)
            gradient = signal.savgol_filter(gradient, window_size, polyorder)
        except:
            # Fall back to simple moving average if Savgol fails
            gradient = pd.Series(gradient).rolling(window=min(window_size, len(gradient)), center=True).mean().values
    
    return x_grad, gradient


def calculate_second_derivative(data, x=None, method='central', window_size=None):
    """
    Calculates the second derivative of data with respect to x.
    
    Parameters:
    -----------
    data : array-like
        Data to analyze
    x : array-like, optional
        X-values corresponding to data (if None, uses equally spaced points)
    method : str, default='central'
        Method for derivative calculation ('forward', 'backward', 'central')
    window_size : int, optional
        Window size for smoothed derivative (if None, no smoothing)
        
    Returns:
    --------
    tuple
        (x_values, second_derivative_values)
    """
    # First calculate the first derivative
    x_grad, first_deriv = calculate_gradient(data, x, method, window_size)
    
    # Then calculate the second derivative (derivative of the first derivative)
    x_grad2, second_deriv = calculate_gradient(first_deriv, x_grad, method, window_size)
    
    return x_grad2, second_deriv


def identify_gradient_outliers(data, x=None, method='central', window_size=None, 
                             threshold=3.0, use_abs=False):
    """
    Identifies outliers in gradient data using statistical methods.
    
    Parameters:
    -----------
    data : array-like
        Data to analyze
    x : array-like, optional
        X-values corresponding to data (if None, uses equally spaced points)
    method : str, default='central'
        Method for gradient calculation ('forward', 'backward', 'central')
    window_size : int, optional
        Window size for smoothed gradient (if None, no smoothing)
    threshold : float, default=3.0
        Z-score threshold for outlier detection
    use_abs : bool, default=False
        Whether to use absolute gradient values for outlier detection
        
    Returns:
    --------
    dict
        Dictionary with gradient outlier information
    """
    # Calculate gradient
    x_grad, gradient = calculate_gradient(data, x, method, window_size)
    
    if len(gradient) < 2:
        return {
            'x_values': x_grad,
            'gradient_values': gradient,
            'outlier_indices': [],
            'outlier_x_values': [],
            'outlier_gradient_values': [],
            'z_scores': [],
            'threshold': threshold
        }
    
    # Use absolute values if requested
    if use_abs:
        gradient_for_stats = np.abs(gradient)
    else:
        gradient_for_stats = gradient
    
    # Calculate z-scores
    z_scores = stats.zscore(gradient_for_stats)
    
    # Identify outliers
    outlier_mask = np.abs(z_scores) > threshold
    outlier_indices = np.where(outlier_mask)[0]
    
    # Get outlier values
    outlier_x = x_grad[outlier_mask]
    outlier_gradient = gradient[outlier_mask]
    outlier_zscores = z_scores[outlier_mask]
    
    return {
        'x_values': x_grad,
        'gradient_values': gradient,
        'outlier_indices': outlier_indices,
        'outlier_x_values': outlier_x,
        'outlier_gradient_values': outlier_gradient,
        'z_scores': outlier_zscores,
        'threshold': threshold
    }


def identify_curvature_points(data, x=None, method='central', window_size=None, 
                            inflection_threshold=0.01, extrema_threshold=0.01):
    """
    Identifies inflection points and extrema (maxima/minima) using second derivatives.
    
    Parameters:
    -----------
    data : array-like
        Data to analyze
    x : array-like, optional
        X-values corresponding to data (if None, uses equally spaced points)
    method : str, default='central'
        Method for derivative calculation ('forward', 'backward', 'central')
    window_size : int, optional
        Window size for smoothed derivatives (if None, no smoothing)
    inflection_threshold : float, default=0.01
        Threshold for identifying inflection points (second derivative near zero)
    extrema_threshold : float, default=0.01
        Threshold for identifying extrema (first derivative near zero)
        
    Returns:
    --------
    dict
        Dictionary with curvature point information
    """
    # Convert to numpy arrays
    y = np.array(data)
    
    # Handle x values
    if x is not None:
        x = np.array(x)
    else:
        x = np.arange(len(y))
    
    # Remove NaN values
    valid_mask = ~np.isnan(y) & ~np.isnan(x)
    y = y[valid_mask]
    x = x[valid_mask]
    
    # Sort by x values
    sort_idx = np.argsort(x)
    y = y[sort_idx]
    x = x[sort_idx]
    
    if len(y) < 3:
        return {
            'inflection_points': [],
            'maxima': [],
            'minima': []
        }
    
    # Calculate first derivative
    x_grad, first_deriv = calculate_gradient(y, x, method, window_size)
    
    # Calculate second derivative
    x_grad2, second_deriv = calculate_gradient(first_deriv, x_grad, method, window_size)
    
    if len(x_grad2) < 1:
        return {
            'inflection_points': [],
            'maxima': [],
            'minima': []
        }
    
    # Identify inflection points (where second derivative crosses zero)
    inflection_points = []
    
    for i in range(len(second_deriv) - 1):
        if (second_deriv[i] * second_deriv[i+1] <= 0 and 
            abs(second_deriv[i]) < inflection_threshold):
            # Interpolate to find more precise x value
            if abs(second_deriv[i+1] - second_deriv[i]) > 1e-10:
                t = -second_deriv[i] / (second_deriv[i+1] - second_deriv[i])
                x_infl = x_grad2[i] + t * (x_grad2[i+1] - x_grad2[i])
            else:
                x_infl = x_grad2[i]
            
            # Find corresponding y value (interpolate from original data)
            idx = np.searchsorted(x, x_infl)
            if idx > 0 and idx < len(x):
                t = (x_infl - x[idx-1]) / (x[idx] - x[idx-1])
                y_infl = y[idx-1] + t * (y[idx] - y[idx-1])
            else:
                # Edge case: use nearest point
                idx = min(max(0, idx), len(x) - 1)
                y_infl = y[idx]
            
            inflection_points.append({
                'x': x_infl,
                'y': y_infl,
                'type': 'inflection'
            })
    
    # Identify extrema (where first derivative crosses zero)
    maxima = []
    minima = []
    
    for i in range(len(first_deriv) - 1):
        if (first_deriv[i] * first_deriv[i+1] <= 0 and 
            abs(first_deriv[i]) < extrema_threshold):
            # Interpolate to find more precise x value
            if abs(first_deriv[i+1] - first_deriv[i]) > 1e-10:
                t = -first_deriv[i] / (first_deriv[i+1] - first_deriv[i])
                x_ext = x_grad[i] + t * (x_grad[i+1] - x_grad[i])
            else:
                x_ext = x_grad[i]
            
            # Find corresponding y value (interpolate from original data)
            idx = np.searchsorted(x, x_ext)
            if idx > 0 and idx < len(x):
                t = (x_ext - x[idx-1]) / (x[idx] - x[idx-1])
                y_ext = y[idx-1] + t * (y[idx] - y[idx-1])
            else:
                # Edge case: use nearest point
                idx = min(max(0, idx), len(x) - 1)
                y_ext = y[idx]
            
            # Determine if maximum or minimum
            # Check second derivative at nearest point in x_grad2
            idx2 = np.searchsorted(x_grad2, x_ext)
            idx2 = min(max(0, idx2), len(x_grad2) - 1)
            
            if idx2 < len(second_deriv):
                if second_deriv[idx2] < 0:
                    # Negative second derivative indicates maximum
                    maxima.append({
                        'x': x_ext,
                        'y': y_ext,
                        'type': 'maximum'
                    })
                else:
                    # Positive second derivative indicates minimum
                    minima.append({
                        'x': x_ext,
                        'y': y_ext,
                        'type': 'minimum'
                    })
    
    return {
        'inflection_points': inflection_points,
        'maxima': maxima,
        'minima': minima
    }


def calculate_piecewise_gradient(data, x=None, segment_count=5, method='linear'):
    """
    Calculates piecewise gradients by dividing data into segments.
    
    Parameters:
    -----------
    data : array-like
        Data to analyze
    x : array-like, optional
        X-values corresponding to data (if None, uses equally spaced points)
    segment_count : int, default=5
        Number of segments to divide the data into
    method : str, default='linear'
        Method for gradient calculation ('linear', 'robust')
        
    Returns:
    --------
    dict
        Dictionary with piecewise gradient information
    """
    # Convert to numpy arrays
    y = np.array(data)
    
    # Handle x values
    if x is not None:
        x = np.array(x)
    else:
        x = np.arange(len(y))
    
    # Remove NaN values
    valid_mask = ~np.isnan(y) & ~np.isnan(x)
    y = y[valid_mask]
    x = x[valid_mask]
    
    # Sort by x values
    sort_idx = np.argsort(x)
    y = y[sort_idx]
    x = x[sort_idx]
    
    if len(y) < segment_count * 2:  # Need at least 2 points per segment
        return {
            'segments': [],
            'gradients': [],
            'segment_bounds': []
        }
    
    # Divide data into segments
    segments = []
    gradients = []
    segment_bounds = []
    
    # Calculate segment size
    n_points = len(x)
    points_per_segment = n_points // segment_count
    
    for i in range(segment_count):
        start_idx = i * points_per_segment
        end_idx = (i + 1) * points_per_segment if i < segment_count - 1 else n_points
        
        if end_idx <= start_idx:
            continue
        
        # Extract segment data
        seg_x = x[start_idx:end_idx]
        seg_y = y[start_idx:end_idx]
        
        if len(seg_x) < 2:
            continue
        
        # Calculate gradient based on method
        if method == 'linear':
                        # Use linear regression
            X = seg_x.reshape(-1, 1)
            model = LinearRegression().fit(X, seg_y)
            gradient = model.coef_[0]
            intercept = model.intercept_
        elif method == 'robust':
            # Use Theil-Sen estimator (robust to outliers)
            from scipy.stats import theilslopes
            slope, intercept, _, _ = theilslopes(seg_y, seg_x)
            gradient = slope
        else:
            # Simple first-last point gradient
            gradient = (seg_y[-1] - seg_y[0]) / (seg_x[-1] - seg_x[0])
            intercept = seg_y[0] - gradient * seg_x[0]
        
        segments.append({
            'x_values': seg_x,
            'y_values': seg_y,
            'start_x': seg_x[0],
            'end_x': seg_x[-1],
            'gradient': gradient,
            'intercept': intercept
        })
        
        gradients.append(gradient)
        segment_bounds.append((seg_x[0], seg_x[-1]))
    
    return {
        'segments': segments,
        'gradients': gradients,
        'segment_bounds': segment_bounds
    }


def identify_gradient_discontinuities(data, x=None, window_size=5, threshold=2.0):
    """
    Identifies potential discontinuities in gradient (sudden changes in slope).
    
    Parameters:
    -----------
    data : array-like
        Data to analyze
    x : array-like, optional
        X-values corresponding to data (if None, uses equally spaced points)
    window_size : int, default=5
        Window size for local gradient calculation
    threshold : float, default=2.0
        Threshold for identifying discontinuities (ratio of adjacent gradients)
        
    Returns:
    --------
    dict
        Dictionary with discontinuity information
    """
    # Convert to numpy arrays
    y = np.array(data)
    
    # Handle x values
    if x is not None:
        x = np.array(x)
    else:
        x = np.arange(len(y))
    
    # Remove NaN values
    valid_mask = ~np.isnan(y) & ~np.isnan(x)
    y = y[valid_mask]
    x = x[valid_mask]
    
    # Sort by x values
    sort_idx = np.argsort(x)
    y = y[sort_idx]
    x = x[sort_idx]
    
    if len(y) < window_size * 2:
        return {
            'discontinuities': []
        }
    
    # Calculate local gradients using sliding windows
    discontinuities = []
    
    for i in range(len(y) - window_size * 2 + 1):
        # Calculate gradient in first window
        x1 = x[i:i+window_size]
        y1 = y[i:i+window_size]
        
        # Calculate gradient in second window
        x2 = x[i+window_size:i+window_size*2]
        y2 = y[i+window_size:i+window_size*2]
        
        # Use linear regression for robust gradient calculation
        X1 = x1.reshape(-1, 1)
        model1 = LinearRegression().fit(X1, y1)
        grad1 = model1.coef_[0]
        
        X2 = x2.reshape(-1, 1)
        model2 = LinearRegression().fit(X2, y2)
        grad2 = model2.coef_[0]
        
        # Calculate ratio of gradients (handle division by zero)
        if abs(grad1) < 1e-10:
            if abs(grad2) < 1e-10:
                ratio = 1.0  # Both gradients near zero
            else:
                ratio = threshold + 1.0  # Consider it a discontinuity
        else:
            ratio = abs(grad2 / grad1)
            
            # If gradients have opposite signs, consider it a discontinuity
            if grad1 * grad2 < 0:
                ratio = max(ratio, threshold + 1.0)
        
        # Check if ratio exceeds threshold
        if ratio > threshold or ratio < 1.0/threshold:
            # Point of discontinuity is between the windows
            x_discont = x[i+window_size-1:i+window_size+1].mean()
            y_discont = y[i+window_size-1:i+window_size+1].mean()
            
            discontinuities.append({
                'x': x_discont,
                'y': y_discont,
                'gradient1': grad1,
                'gradient2': grad2,
                'ratio': ratio,
                'window_size': window_size
            })
    
    return {
        'discontinuities': discontinuities
    }


def create_gradient_plot(data, x=None, column_name=None, method='central', window_size=None,
                       highlight_outliers=True, threshold=3.0, use_abs=False,
                       output_dir=None, filename=None, show_plot=True):
    """
    Creates a plot visualizing data and its gradient with optional outlier highlighting.
    
    Parameters:
    -----------
    data : array-like
        Data to analyze
    x : array-like, optional
        X-values corresponding to data (if None, uses equally spaced points)
    column_name : str, optional
        Name of the column/variable
    method : str, default='central'
        Method for gradient calculation ('forward', 'backward', 'central')
    window_size : int, optional
        Window size for smoothed gradient (if None, no smoothing)
    highlight_outliers : bool, default=True
        Whether to highlight gradient outliers
    threshold : float, default=3.0
        Z-score threshold for outlier detection
    use_abs : bool, default=False
        Whether to use absolute gradient values for outlier detection
    output_dir : str, optional
        Directory to save the plot
    filename : str, optional
        Filename for the saved plot
    show_plot : bool, default=True
        Whether to display the plot
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    # Convert to pandas Series
    y = pd.Series(data)
    
    # Use column name if provided, otherwise use Series name
    col_name = column_name if column_name is not None else (y.name if y.name else "Value")
    
    # Handle x values
    if x is not None:
        x = pd.Series(x)
    else:
        x = pd.Series(np.arange(len(y)))
    
    # Remove NaN values
    valid_mask = ~y.isna() & ~x.isna()
    y = y[valid_mask]
    x = x[valid_mask]
    
    # Sort by x values
    sorted_data = pd.DataFrame({'x': x, 'y': y}).sort_values('x')
    x = sorted_data['x']
    y = sorted_data['y']
    
    if len(y) < 2:
        return None
    
    # Generate filename if not provided
    if filename is None:
        safe_col_name = col_name.replace('/', '_').replace('\\', '_')
        filename = f'gradient_{safe_col_name}.png'
    
    # Calculate gradient and identify outliers
    gradient_info = identify_gradient_outliers(
        y, x, method=method, window_size=window_size, 
        threshold=threshold, use_abs=use_abs
    )
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [1, 1]})
    
    # Plot original data in top subplot
    ax1.scatter(x, y, alpha=0.7, label='Data Points')
    ax1.plot(x, y, 'b-', alpha=0.5)
    
    # Set labels for top subplot
    ax1.set_xlabel('X' if x.name is None else x.name)
    ax1.set_ylabel(col_name)
    ax1.set_title(f'Original Data: {col_name}')
    ax1.grid(True, alpha=0.3)
    
    # Plot gradient in bottom subplot
    x_grad = gradient_info['x_values']
    gradient = gradient_info['gradient_values']
    
    ax2.scatter(x_grad, gradient, alpha=0.7, label='Gradient')
    ax2.plot(x_grad, gradient, 'g-', alpha=0.5)
    
    # Highlight outliers if requested
    if highlight_outliers and len(gradient_info['outlier_indices']) > 0:
        outlier_x = gradient_info['outlier_x_values']
        outlier_grad = gradient_info['outlier_gradient_values']
        
        # Highlight in gradient plot
        ax2.scatter(outlier_x, outlier_grad, color='red', s=80, alpha=0.7, 
                   label=f'Outliers (z > {threshold})')
        
        # Find corresponding points in original data
        for x_val in outlier_x:
            # Find nearest x value in original data
            idx = np.abs(x - x_val).argmin()
            
            # Highlight in original data plot
            ax1.scatter(x.iloc[idx], y.iloc[idx], color='red', s=80, alpha=0.7)
    
    # Set labels for bottom subplot
    ax2.set_xlabel('X' if x.name is None else x.name)
    ax2.set_ylabel(f'Gradient of {col_name}')
    ax2.set_title(f'Gradient Analysis: {col_name} (method={method})')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add horizontal line at y=0 in gradient plot
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Add overall title
    plt.suptitle(f'Gradient Analysis for {col_name}', fontsize=16, y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save plot if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, filename)
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    # Show or close plot
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def create_curvature_plot(data, x=None, column_name=None, method='central', window_size=None,
                        highlight_points=True, output_dir=None, filename=None, show_plot=True):
    """
    Creates a plot visualizing data with its curvature points (inflections, extrema).
    
    Parameters:
    -----------
    data : array-like
        Data to analyze
    x : array-like, optional
        X-values corresponding to data (if None, uses equally spaced points)
    column_name : str, optional
        Name of the column/variable
    method : str, default='central'
        Method for derivative calculation ('forward', 'backward', 'central')
    window_size : int, optional
        Window size for smoothed derivatives (if None, no smoothing)
    highlight_points : bool, default=True
        Whether to highlight curvature points
    output_dir : str, optional
        Directory to save the plot
    filename : str, optional
        Filename for the saved plot
    show_plot : bool, default=True
        Whether to display the plot
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    # Convert to pandas Series
    y = pd.Series(data)
    
    # Use column name if provided, otherwise use Series name
    col_name = column_name if column_name is not None else (y.name if y.name else "Value")
    
    # Handle x values
    if x is not None:
        x = pd.Series(x)
    else:
        x = pd.Series(np.arange(len(y)))
    
    # Remove NaN values
    valid_mask = ~y.isna() & ~x.isna()
    y = y[valid_mask]
    x = x[valid_mask]
    
    # Sort by x values
    sorted_data = pd.DataFrame({'x': x, 'y': y}).sort_values('x')
    x = sorted_data['x']
    y = sorted_data['y']
    
    if len(y) < 3:
        return None
    
    # Generate filename if not provided
    if filename is None:
        safe_col_name = col_name.replace('/', '_').replace('\\', '_')
        filename = f'curvature_{safe_col_name}.png'
    
    # Identify curvature points
    curvature_info = identify_curvature_points(
        y, x, method=method, window_size=window_size
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot original data
    ax.scatter(x, y, alpha=0.5, label='Data Points')
    ax.plot(x, y, 'b-', alpha=0.5)
    
    # Highlight curvature points if requested
    if highlight_points:
        # Highlight inflection points
        if curvature_info['inflection_points']:
            infl_x = [p['x'] for p in curvature_info['inflection_points']]
            infl_y = [p['y'] for p in curvature_info['inflection_points']]
            ax.scatter(infl_x, infl_y, color='green', s=100, marker='o', alpha=0.7,
                      label='Inflection Points')
        
        # Highlight maxima
        if curvature_info['maxima']:
            max_x = [p['x'] for p in curvature_info['maxima']]
            max_y = [p['y'] for p in curvature_info['maxima']]
            ax.scatter(max_x, max_y, color='red', s=100, marker='^', alpha=0.7,
                      label='Maxima')
        
        # Highlight minima
        if curvature_info['minima']:
            min_x = [p['x'] for p in curvature_info['minima']]
            min_y = [p['y'] for p in curvature_info['minima']]
            ax.scatter(min_x, min_y, color='purple', s=100, marker='v', alpha=0.7,
                      label='Minima')
    
    # Set labels and title
    ax.set_xlabel('X' if x.name is None else x.name)
    ax.set_ylabel(col_name)
    
    # Create title with curvature information
    title = f'Curvature Analysis: {col_name}\n'
    title += f"Inflection Points: {len(curvature_info['inflection_points'])}, "
    title += f"Maxima: {len(curvature_info['maxima'])}, "
    title += f"Minima: {len(curvature_info['minima'])}"
    
    ax.set_title(title)
    
    # Add legend
    ax.legend(loc='best')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Save plot if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, filename)
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    # Show or close plot
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def create_piecewise_gradient_plot(data, x=None, column_name=None, segment_count=5, 
                                 method='linear', output_dir=None, filename=None, 
                                 show_plot=True):
    """
    Creates a plot visualizing data with piecewise gradients.
    
    Parameters:
    -----------
    data : array-like
        Data to analyze
    x : array-like, optional
        X-values corresponding to data (if None, uses equally spaced points)
    column_name : str, optional
        Name of the column/variable
    segment_count : int, default=5
        Number of segments to divide the data into
    method : str, default='linear'
        Method for gradient calculation ('linear', 'robust')
    output_dir : str, optional
        Directory to save the plot
    filename : str, optional
        Filename for the saved plot
    show_plot : bool, default=True
        Whether to display the plot
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    # Convert to pandas Series
    y = pd.Series(data)
    
    # Use column name if provided, otherwise use Series name
    col_name = column_name if column_name is not None else (y.name if y.name else "Value")
    
    # Handle x values
    if x is not None:
        x = pd.Series(x)
    else:
        x = pd.Series(np.arange(len(y)))
    
    # Remove NaN values
    valid_mask = ~y.isna() & ~x.isna()
    y = y[valid_mask]
    x = x[valid_mask]
    
    # Sort by x values
    sorted_data = pd.DataFrame({'x': x, 'y': y}).sort_values('x')
    x = sorted_data['x']
    y = sorted_data['y']
    
    if len(y) < segment_count * 2:  # Need at least 2 points per segment
        return None
    
    # Generate filename if not provided
    if filename is None:
        safe_col_name = col_name.replace('/', '_').replace('\\', '_')
        filename = f'piecewise_gradient_{safe_col_name}.png'
    
    # Calculate piecewise gradients
    gradient_info = calculate_piecewise_gradient(
        y, x, segment_count=segment_count, method=method
    )
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot original data in top subplot
    ax1.scatter(x, y, alpha=0.5, label='Data Points')
    
    # Plot piecewise linear fits
    for i, segment in enumerate(gradient_info['segments']):
        seg_x = segment['x_values']
        
        # Calculate line values
        line_y = segment['gradient'] * seg_x + segment['intercept']
        
        # Plot segment line
        ax1.plot(seg_x, line_y, '-', linewidth=2, alpha=0.7,
                label=f'Segment {i+1}' if i == 0 else "")
        
        # Add gradient text
        mid_x = (seg_x[0] + seg_x[-1]) / 2
        mid_y = (line_y[0] + line_y[-1]) / 2
        ax1.text(mid_x, mid_y, f"m = {segment['gradient']:.4f}", 
                ha='center', va='bottom', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Set labels and title for top subplot
    ax1.set_xlabel('X' if x.name is None else x.name)
    ax1.set_ylabel(col_name)
    ax1.set_title(f'Piecewise Gradient Analysis: {col_name}')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot gradient values in bottom subplot
    gradients = gradient_info['gradients']
    bounds = gradient_info['segment_bounds']
    
    # Calculate midpoints of segments for bar positions
    positions = [(bound[0] + bound[1]) / 2 for bound in bounds]
    
    # Create bar chart of gradients
    ax2.bar(positions, gradients, width=(x.max() - x.min()) / (segment_count * 2), 
           alpha=0.7, color='green')
    
    # Add horizontal line at y=0
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Set labels for bottom subplot
    ax2.set_xlabel('X' if x.name is None else x.name)
    ax2.set_ylabel('Gradient')
    ax2.set_title('Segment Gradients')
    ax2.grid(True, alpha=0.3)
    
    # Set x-limits to match top subplot
    ax2.set_xlim(ax1.get_xlim())
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, filename)
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    # Show or close plot
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def analyze_gradients(df, columns=None, x_column=None, method='central', window_size=None,
                    segment_count=5, threshold=3.0, output_dir=None, 
                    create_plots=True, show_plots=True):
    """
    Comprehensive gradient analysis for numerical columns.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The DataFrame to analyze
    columns : list of str, optional
        Specific columns to analyze (default: all numerical columns)
    x_column : str, optional
        Column to use as x-axis (default: uses indices)
    method : str, default='central'
        Method for gradient calculation ('forward', 'backward', 'central')
    window_size : int, optional
        Window size for smoothed gradient (if None, no smoothing)
    segment_count : int, default=5
        Number of segments for piecewise gradient analysis
    threshold : float, default=3.0
        Z-score threshold for outlier detection
    output_dir : str, optional
        Directory to save output files and plots
    create_plots : bool, default=True
        Whether to create visualization plots
    show_plots : bool, default=True
        Whether to display plots
        
    Returns:
    --------
    dict
        Dictionary containing gradient analysis results
    """
    # Set up logging
    logger = logging.getLogger('gradient_analysis')
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
            file_handler = logging.FileHandler(os.path.join(output_dir, 'gradient_analysis.log'))
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    logger.info("\n=== Gradient Analysis ===")
    
    # Select numerical columns if not specified
    if columns is None:
        columns = df.select_dtypes(include=np.number).columns.tolist()
        
        # Remove x_column from analysis columns if it's in the list
        if x_column in columns:
            columns.remove(x_column)
    else:
        # Filter to only include columns that exist
        columns = [col for col in columns if col in df.columns]
    
    if not columns:
        logger.info("No valid columns for gradient analysis.")
        return {}
    
    logger.info(f"Analyzing gradients for {len(columns)} columns")
    
    # Get x values if x_column is specified
    x_values = None
    if x_column and x_column in df.columns:
        logger.info(f"Using '{x_column}' as x-axis")
        x_values = df[x_column]
    
    # Initialize results dictionary
    results = {
        'gradient_outliers': {},
        'curvature_points': {},
        'piecewise_gradients': {},
        'discontinuities': {},
        'plots': {}
    }
    
    # Analyze each column
    for col in columns:
        logger.info(f"\nAnalyzing column: {col}")
        
        # Identify gradient outliers
        gradient_info = identify_gradient_outliers(
            df[col], x_values, method=method, window_size=window_size, 
            threshold=threshold
        )
        results['gradient_outliers'][col] = gradient_info
        
        # Identify curvature points
        curvature_info = identify_curvature_points(
            df[col], x_values, method=method, window_size=window_size
        )
        results['curvature_points'][col] = curvature_info
        
        # Calculate piecewise gradients
        piecewise_info = calculate_piecewise_gradient(
            df[col], x_values, segment_count=segment_count
        )
        results['piecewise_gradients'][col] = piecewise_info
        
        # Identify gradient discontinuities
        discontinuity_info = identify_gradient_discontinuities(
            df[col], x_values, window_size=window_size, threshold=2.0
        )
        results['discontinuities'][col] = discontinuity_info
        
        # Log results
        outlier_count = len(gradient_info['outlier_indices'])
        logger.info(f"  Gradient Outliers: {outlier_count}")
        
        inflection_count = len(curvature_info['inflection_points'])
        maxima_count = len(curvature_info['maxima'])
        minima_count = len(curvature_info['minima'])
        logger.info(f"  Curvature Points: {inflection_count} inflections, "
                   f"{maxima_count} maxima, {minima_count} minima")
        
        segment_count_actual = len(piecewise_info['segments'])
        logger.info(f"  Piecewise Gradients: {segment_count_actual} segments")
        
        discontinuity_count = len(discontinuity_info['discontinuities'])
        logger.info(f"  Gradient Discontinuities: {discontinuity_count}")
        
        # Create plots if requested
        if create_plots:
            plots_dir = os.path.join(output_dir, 'gradient_plots') if output_dir else None
            
            # Create gradient plot
            gradient_plot = create_gradient_plot(
                df[col],
                x=x_values,
                column_name=col,
                method=method,
                window_size=window_size,
                threshold=threshold,
                output_dir=plots_dir,
                show_plot=show_plots
            )
            
            # Create curvature plot
            curvature_plot = create_curvature_plot(
                df[col],
                x=x_values,
                column_name=col,
                method=method,
                window_size=window_size,
                output_dir=plots_dir,
                show_plot=show_plots
            )
            
            # Create piecewise gradient plot
            piecewise_plot = create_piecewise_gradient_plot(
                df[col],
                x=x_values,
                column_name=col,
                segment_count=segment_count,
                output_dir=plots_dir,
                show_plot=show_plots
            )
            
            results['plots'][col] = {
                'gradient_plot': gradient_plot,
                'curvature_plot': curvature_plot,
                'piecewise_plot': piecewise_plot
            }
    
    # Create summary table
    summary_data = []
    
    for col in columns:
        gradient_info = results['gradient_outliers'][col]
        curvature_info = results['curvature_points'][col]
        discontinuity_info = results['discontinuities'][col]
        
        summary_data.append({
            'Column': col,
            'Gradient Outliers': len(gradient_info['outlier_indices']),
            'Inflection Points': len(curvature_info['inflection_points']),
            'Maxima': len(curvature_info['maxima']),
            'Minima': len(curvature_info['minima']),
            'Discontinuities': len(discontinuity_info['discontinuities'])
        })
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    results['summary'] = summary_df
    
    # Log summary
    logger.info("\nGradient Analysis Summary:")
    logger.info(f"\n{summary_df}")
    
    # Save summary to CSV if output directory is provided
    if output_dir:
        summary_path = os.path.join(output_dir, "gradient_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"\nSummary saved to: {summary_path}")
    
    logger.info("\n=== Gradient Analysis Complete ===")
    
    return results

