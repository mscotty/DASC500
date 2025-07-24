import os
import plotly.graph_objects as go
import plotly.subplots as sp
import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, Dict

def plot_parameter_optimization(
    data: np.ndarray,
    optimization_results: Dict,
    distribution: str,
    output_dir: Optional[str] = None,
    output_name: Optional[str] = None,
    title_name: Optional[str] = None,
    title_font_size: int = 28,
    title_font_name: str = "Times New Roman",
    axis_font_size: int = 24,
    axis_font_name: str = "Times New Roman",
    hist_color: str = "skyblue",
    fit_color: str = "red",
    line_width: int = 2,
    show_params: bool = True,
    param_font_size: int = 16,
    add_qqplot: bool = True,
    qqplot_marker_color: str = "blue",
    qqplot_marker_size: int = 8,
    qqplot_line_color: str = "red",
    qqplot_marker_opacity: float = 0.7,
    legend_font_size: int = 16,
    legend_font_name: str = "Times New Roman"
):
    # Extract optimization results
    method = optimization_results.get('method', 'unknown')
    parameters = optimization_results.get('parameters', [])
    log_likelihood = optimization_results.get('log_likelihood', None)
    
    # Determine number of subplots
    n_plots = 2 if add_qqplot else 1
    
    # Create subplots
    fig = sp.make_subplots(
        rows=1, 
        cols=n_plots,
        subplot_titles=["Data and Fitted Distribution"] + (["QQ Plot"] if add_qqplot else [])
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
    
    # Add fitted distribution PDF
    try:
        # Get distribution function
        dist = getattr(stats, distribution)
        
        # Create x values for PDF
        x_range = np.linspace(min(data), max(data), 1000)
        
        # Calculate PDF with optimized parameters
        y_pdf = dist.pdf(x_range, *parameters)
        
        # Add PDF line
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=y_pdf,
                mode='lines',
                line=dict(color=fit_color, width=line_width),
                name=f'Fitted {distribution.capitalize()}'
            ),
            row=1, col=1
        )
        
    except Exception as e:
        print(f"Could not plot fitted PDF: {e}")
    
    # Add QQ plot if requested
    if add_qqplot:
        try:
            # Calculate theoretical quantiles
            quantiles = np.linspace(0.01, 0.99, min(100, len(data)))
            theoretical_quantiles = dist.ppf(quantiles, *parameters)
            
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
    if show_params:
        param_text = f"Distribution: {distribution}<br>"
        param_text += f"Method: {method}<br>"
        
        # Add parameters
        if isinstance(parameters, (list, tuple, np.ndarray)):
            param_names = []
            # Try to get parameter names from the distribution
            try:
                param_names = dist._param_names
            except:
                # If not available, use generic names
                param_names = [f"param{i}" for i in range(len(parameters))]
            
            for i, (name, value) in enumerate(zip(param_names, parameters)):
                param_text += f"{name}: {value:.4f}<br>"
        
        # Add log-likelihood if available
        if log_likelihood is not None:
            param_text += f"Log-Likelihood: {log_likelihood:.4f}<br>"
        
        fig.add_annotation(
            x=0.02,
            y=0.98,
            xref="x",
            yref="y",
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
        title_name = f"Parameter Optimization for {distribution.capitalize()} Distribution"
    
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
        if i < n_plots:  # Only update subplot titles, not parameter annotations
            fig.layout.annotations[i].font.size = axis_font_size
    
    # Save or display the plot
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        file_path = os.path.join(output_dir, output_name if output_name else f"parameter_optimization_{distribution}.png")
        fig.write_image(file_path, format="png")
    else:
        fig.show()
