import os
import plotly.graph_objects as go
import plotly.subplots as sp
import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, Dict, List, Union

def plot_synthetic_data(
    data: np.ndarray,
    distribution: str,
    params: Dict = None,
    with_noise: bool = False,
    noise_level: float = 0.1,
    output_dir: Optional[str] = None,
    output_name: Optional[str] = None,
    title_name: Optional[str] = None,
    title_font_size: int = 28,
    title_font_name: str = "Times New Roman",
    axis_font_size: int = 24,
    axis_font_name: str = "Times New Roman",
    hist_color: str = "skyblue",
    line_color: str = "red",
    line_width: int = 2,
    show_params: bool = True,
    param_font_size: int = 16,
    add_qqplot: bool = True,
    qqplot_marker_color: str = "blue",
    qqplot_marker_size: int = 8,
    qqplot_line_color: str = "red",
    qqplot_marker_opacity: float = 0.7,
    add_ecdf: bool = True,
    ecdf_line_color: str = "green",
    ecdf_line_width: int = 2,
    ecdf_theoretical_color: str = "red",
    legend_font_size: int = 16,
    legend_font_name: str = "Times New Roman"
):
    """
    @brief Plot synthetic data with histogram, theoretical PDF, and optional QQ plot and ECDF.
    @param[in] data (np.ndarray) The synthetic data to plot.
    @param[in] distribution (str) Name of the distribution used to generate the data.
    @param[in] params (Dict) Parameters used to generate the data.
    @param[in] with_noise (bool) Whether noise was added to the data.
    @param[in] noise_level (float) Level of noise added to the data.
    @param[in] output_dir (str) Directory to save the plot as a PNG file. If None, the plot is displayed interactively.
    @param[in] output_name (str) Custom file name for the saved plot.
    @param[in] title_name (str) Custom title for the chart. If None, a title is generated based on the distribution.
    @param[in] title_font_size (int) Font size for the chart title.
    @param[in] title_font_name (str) Font name for the chart title.
    @param[in] axis_font_size (int) Font size for axis labels.
    @param[in] axis_font_name (str) Font name for axis labels.
    @param[in] hist_color (str) Color for the histogram.
    @param[in] line_color (str) Color for theoretical PDF line.
    @param[in] line_width (int) Width of theoretical PDF line.
    @param[in] show_params (bool) Whether to display distribution parameters on the plot.
    @param[in] param_font_size (int) Font size for parameter annotations.
    @param[in] add_qqplot (bool) Whether to include a QQ plot.
    @param[in] qqplot_marker_color (str) Color for QQ plot markers.
    @param[in] qqplot_marker_size (int) Size of QQ plot markers.
    @param[in] qqplot_line_color (str) Color for QQ plot reference line.
    @param[in] qqplot_marker_opacity (float) Opacity of QQ plot markers.
    @param[in] add_ecdf (bool) Whether to include an ECDF plot.
    @
    """
    # Determine the number of subplots needed
    n_plots = 1 + int(add_qqplot) + int(add_ecdf)
    
    # Create subplot layout
    fig = sp.make_subplots(
        rows=1, 
        cols=n_plots,
        subplot_titles=["Histogram with PDF"] + 
                      (["QQ Plot"] if add_qqplot else []) + 
                      (["ECDF"] if add_ecdf else [])
    )
    
    # Calculate bin count based on data size
    bin_count = min(50, max(10, int(np.sqrt(len(data)))))
    
    # Add histogram
    fig.add_trace(
        go.Histogram(
            x=data,
            nbinsx=bin_count,
            histnorm='probability density',
            marker_color=hist_color,
            opacity=0.7,
            name='Data'
        ),
        row=1, col=1
    )
    
    # Add theoretical PDF if possible
    if distribution and params:
        try:
            # Get distribution function
            dist = getattr(stats, distribution)
            
            # Create x values for PDF
            x_range = np.linspace(min(data), max(data), 1000)
            
            # Extract parameters for PDF
            if params:
                if isinstance(params, dict):
                    # Handle common parameter formats
                    if 'loc' in params and 'scale' in params:
                        loc = params.get('loc', 0)
                        scale = params.get('scale', 1)
                        
                        # Handle shape parameters
                        shape_params = []
                        for param in ['a', 'b', 'c', 'd', 'k', 's', 'df', 'alpha', 'beta']:
                            if param in params:
                                shape_params.append(params[param])
                        
                        # Calculate PDF
                        if shape_params:
                            y_pdf = dist.pdf(x_range, *shape_params, loc=loc, scale=scale)
                        else:
                            y_pdf = dist.pdf(x_range, loc=loc, scale=scale)
                    else:
                        # Try to use params as is
                        param_values = list(params.values())
                        y_pdf = dist.pdf(x_range, *param_values)
                else:
                    # If params is not a dict, try using it directly
                    y_pdf = dist.pdf(x_range, *params)
            else:
                # If no params, try fitting the distribution
                fitted_params = dist.fit(data)
                y_pdf = dist.pdf(x_range, *fitted_params)
                
            # Add PDF line
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=y_pdf,
                    mode='lines',
                    line=dict(color=line_color, width=line_width),
                    name=f'{distribution.capitalize()} PDF'
                ),
                row=1, col=1
            )
            
        except Exception as e:
            print(f"Could not plot theoretical PDF: {e}")
    
    # Add QQ plot if requested
    if add_qqplot:
        try:
            # Calculate theoretical quantiles
            quantiles = np.linspace(0.01, 0.99, min(100, len(data)))
            
            # Get theoretical distribution
            if distribution:
                dist = getattr(stats, distribution)
                if params:
                    # Extract parameters
                    if isinstance(params, dict):
                        loc = params.get('loc', 0)
                        scale = params.get('scale', 1)
                        shape_params = []
                        for param in ['a', 'b', 'c', 'd', 'k', 's', 'df', 'alpha', 'beta']:
                            if param in params:
                                shape_params.append(params[param])
                        
                        # Calculate theoretical quantiles
                        if shape_params:
                            theoretical_quantiles = dist.ppf(quantiles, *shape_params, loc=loc, scale=scale)
                        else:
                            theoretical_quantiles = dist.ppf(quantiles, loc=loc, scale=scale)
                    else:
                        theoretical_quantiles = dist.ppf(quantiles, *params)
                else:
                    # Use normal distribution if no parameters
                    theoretical_quantiles = stats.norm.ppf(quantiles)
            else:
                # Use normal distribution if no distribution specified
                theoretical_quantiles = stats.norm.ppf(quantiles)
                
            # Get empirical quantiles
            empirical_quantiles = np.quantile(data, quantiles)
            
            # Add QQ plot
            fig.add_trace(
                go.Scatter(
                    x=theoretical_quantiles,
                    y=empirical_quantiles,
                    mode='markers',
                    marker=dict(
                        color=qqplot_marker_color,
                        size=qqplot_marker_size,
                        opacity=qqplot_marker_opacity
                    ),
                    name='QQ Plot'
                ),
                row=1, col=2
            )
            
            # Add reference line
            min_val = min(np.min(theoretical_quantiles), np.min(empirical_quantiles))
            max_val = max(np.max(theoretical_quantiles), np.max(empirical_quantiles))
            
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    line=dict(color=qqplot_line_color, width=line_width, dash='dash'),
                    name='Reference Line'
                ),
                row=1, col=2
            )
            
            # Update QQ plot axes
            fig.update_xaxes(
                title_text="Theoretical Quantiles",
                title_font=dict(size=axis_font_size, family=axis_font_name),
                row=1, col=2
            )
            
            fig.update_yaxes(
                title_text="Sample Quantiles",
                title_font=dict(size=axis_font_size, family=axis_font_name),
                row=1, col=2
            )
            
        except Exception as e:
            print(f"Could not create QQ plot: {e}")
    
    # Add ECDF if requested
    if add_ecdf:
        try:
            # Calculate ECDF
            x_sorted = np.sort(data)
            y_ecdf = np.arange(1, len(data) + 1) / len(data)
            
            # Add ECDF line
            fig.add_trace(
                go.Scatter(
                    x=x_sorted,
                    y=y_ecdf,
                    mode='lines',
                    line=dict(color=ecdf_line_color, width=ecdf_line_width),
                    name='Empirical CDF'
                ),
                row=1, col=3 if add_qqplot else 2
            )
            
            # Add theoretical CDF if distribution is provided
            if distribution and params:
                try:
                    # Get distribution function
                    dist = getattr(stats, distribution)
                    
                    # Create x values for CDF
                    x_range = np.linspace(min(data), max(data), 1000)
                    
                    # Calculate theoretical CDF
                    if isinstance(params, dict):
                        loc = params.get('loc', 0)
                        scale = params.get('scale', 1)
                        shape_params = []
                        for param in ['a', 'b', 'c', 'd', 'k', 's', 'df', 'alpha', 'beta']:
                            if param in params:
                                shape_params.append(params[param])
                        
                        if shape_params:
                            y_cdf = dist.cdf(x_range, *shape_params, loc=loc, scale=scale)
                        else:
                            y_cdf = dist.cdf(x_range, loc=loc, scale=scale)
                    else:
                        y_cdf = dist.cdf(x_range, *params)
                        
                    # Add theoretical CDF line
                    fig.add_trace(
                        go.Scatter(
                            x=x_range,
                            y=y_cdf,
                            mode='lines',
                            line=dict(color=ecdf_theoretical_color, width=line_width, dash='dot'),
                            name=f'Theoretical CDF'
                        ),
                        row=1, col=3 if add_qqplot else 2
                    )
                    
                except Exception as e:
                    print(f"Could not plot theoretical CDF: {e}")
            
            # Update ECDF axes
            fig.update_xaxes(
                title_text="Value",
                title_font=dict(size=axis_font_size, family=axis_font_name),
                row=1, col=3 if add_qqplot else 2
            )
            
            fig.update_yaxes(
                title_text="Cumulative Probability",
                title_font=dict(size=axis_font_size, family=axis_font_name),
                row=1, col=3 if add_qqplot else 2
            )
            
        except Exception as e:
            print(f"Could not create ECDF plot: {e}")
    
    # Update histogram axes
    fig.update_xaxes(
        title_text="Value",
        title_font=dict(size=axis_font_size, family=axis_font_name),
        row=1, col=1
    )
    
    fig.update_yaxes(
        title_text="Density",
        title_font=dict(size=axis_font_size, family=axis_font_name),
        row=1, col=1
    )
    
    # Add parameter annotation if requested
    if show_params and params:
        param_text = f"Distribution: {distribution}<br>"
        if with_noise:
            param_text += f"Noise Level: {noise_level}<br>"
            
        if isinstance(params, dict):
            for key, value in params.items():
                param_text += f"{key}: {value:.4f}<br>" if isinstance(value, float) else f"{key}: {value}<br>"
        else:
            param_text += f"Parameters: {params}"
            
        fig.add_annotation(
            x=0.02,
            y=0.98,
            xref="paper",
            yref="paper",
            text=param_text,
            showarrow=False,
            font=dict(size=param_font_size, color="black"),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            borderpad=4,
            align="left"
        )
    
    # Set title
    if title_name is None:
        title_name = f"Synthetic {distribution.capitalize()} Distribution"
        if with_noise:
            title_name += f" with Noise (level={noise_level})"
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=title_name,
            font=dict(family=title_font_name, size=title_font_size),
            x=0.5
        ),
        legend=dict(
            font=dict(family=legend_font_name, size=legend_font_size)
        ),
        height=600,
        width=500 * n_plots,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    # Update subplot titles font
    for i in range(len(fig.layout.annotations)):
        fig.layout.annotations[i].font.size = axis_font_size
    
    # Save or display the plot
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        file_path = os.path.join(output_dir, output_name if output_name else f"synthetic_{distribution}_data.png")
        fig.write_image(file_path, format="png")
    else:
        fig.show()

