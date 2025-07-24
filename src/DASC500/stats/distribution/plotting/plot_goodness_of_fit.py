import os
import plotly.graph_objects as go
import plotly.subplots as sp
import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, Dict


def plot_goodness_of_fit(
    data: np.ndarray,
    distribution: str,
    test_results: Dict,
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
    show_test_results: bool = True,
    result_font_size: int = 16,
    add_pp_plot: bool = True,
    pp_marker_color: str = "blue",
    pp_marker_size: int = 8,
    pp_line_color: str = "red",
    pp_marker_opacity: float = 0.7,
    legend_font_size: int = 16,
    legend_font_name: str = "Times New Roman"
):
    """
    Plot goodness-of-fit test results with histogram, fitted PDF, and P-P plot.
    
    Parameters:
    -----------
    data : np.ndarray
        The data to analyze
    distribution : str
        Name of the distribution to fit
    test_results : Dict
        Dictionary containing test results including parameters, test name, statistic, and p-value
    output_dir : Optional[str]
        Directory to save the output plot
    output_name : Optional[str]
        Filename for the saved plot
    title_name : Optional[str]
        Custom title for the plot
    title_font_size : int
        Font size for the title
    title_font_name : str
        Font family for the title
    axis_font_size : int
        Font size for axis labels
    axis_font_name : str
        Font family for axis labels
    hist_color : str
        Color for the histogram
    fit_color : str
        Color for the fitted distribution line
    line_width : int
        Width of the distribution line
    show_test_results : bool
        Whether to display test statistics on the plot
    result_font_size : int
        Font size for test results
    add_pp_plot : bool
        Whether to add a P-P plot alongside the histogram
    pp_marker_color : str
        Color for P-P plot markers
    pp_marker_size : int
        Size of P-P plot markers
    pp_line_color : str
        Color for the reference line in P-P plot
    pp_marker_opacity : float
        Opacity of P-P plot markers
    legend_font_size : int
        Font size for legend
    legend_font_name : str
        Font family for legend
        
    Returns:
    --------
    go.Figure
        The plotly figure object
    """
    # Extract test results
    params = test_results.get('parameters', None)
    test_name = test_results.get('test', 'Unknown')
    statistic = test_results.get('statistic', None)
    p_value = test_results.get('p_value', None)
    
    # Determine number of subplots
    n_plots = 2 if add_pp_plot else 1
    
    # Create subplots
    fig = sp.make_subplots(
        rows=1, 
        cols=n_plots,
        subplot_titles=["Data and Fitted Distribution"] + (["P-P Plot"] if add_pp_plot else [])
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
    
    # Add fitted distribution PDF if parameters are available
    if params is not None:
        try:
            # Get distribution function
            dist = getattr(stats, distribution)
            
            # Create x values for PDF
            x_range = np.linspace(min(data), max(data), 1000)
            
            # Calculate PDF with fitted parameters
            y_pdf = dist.pdf(x_range, *params)
            
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
    
    # Add P-P plot if requested
    if add_pp_plot and params is not None:
        try:
            # Get distribution function
            dist = getattr(stats, distribution)
            
            # Calculate empirical CDF
            x_sorted = np.sort(data)
            y_ecdf = np.arange(1, len(data) + 1) / len(data)
            
            # Calculate theoretical CDF
            y_cdf = dist.cdf(x_sorted, *params)
            
            # Add P-P plot
            fig.add_trace(
                go.Scatter(
                    x=y_cdf,
                    y=y_ecdf,
                    mode='markers',
                    marker=dict(
                        color=pp_marker_color,
                        size=pp_marker_size,
                        opacity=pp_marker_opacity
                    ),
                    name='P-P Plot'
                ),
                row=1, col=2
            )
            
            # Add reference line
            fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode='lines',
                    line=dict(color=pp_line_color, width=line_width, dash='dash'),
                    name='Reference Line'
                ),
                row=1, col=2
            )
            
            # Update P-P plot axes
            fig.update_xaxes(
                title_text="Theoretical Probability",
                title_font=dict(size=axis_font_size, family=axis_font_name),
                range=[0, 1],
                row=1, col=2
            )
            
            fig.update_yaxes(
                title_text="Empirical Probability",
                title_font=dict(size=axis_font_size, family=axis_font_name),
                range=[0, 1],
                row=1, col=2
            )
            
        except Exception as e:
            print(f"Could not create P-P plot: {e}")
    
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
    
    # Add test results annotation if requested
    if show_test_results:
        result_text = f"Test: {test_name}<br>"
        
        if statistic is not None:
            result_text += f"Statistic: {statistic:.4f}<br>"
            
        if p_value is not None:
            result_text += f"p-value: {p_value:.4f}<br>"
            result_text += f"Conclusion: {'Reject H₀' if p_value < 0.05 else 'Fail to reject H₀'} (α=0.05)"
        
        fig.add_annotation(
            x=0.02,
            y=0.98,
            xref="x domain",
            yref="y domain",
            text=result_text,
            showarrow=False,
            font=dict(
                size=result_font_size,
                family=title_font_name
            ),
            align="left",
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="black",
            borderwidth=1,
            borderpad=4,
            row=1, col=1
        )
    
    # Update layout
    plot_title = title_name if title_name else f"Goodness-of-Fit: {distribution.capitalize()} Distribution"
    
    fig.update_layout(
        title=dict(
            text=plot_title,
            font=dict(
                size=title_font_size,
                family=title_font_name
            )
        ),
        legend=dict(
            font=dict(
                size=legend_font_size,
                family=legend_font_name
            )
        ),
        template="plotly_white",
        height=600,
        width=1200 if add_pp_plot else 800,
        showlegend=True
    )
    
    # Save figure if output directory and name are provided
    if output_dir and output_name:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        output_path = os.path.join(output_dir, f"{output_name}.html")
        fig.write_html(output_path)
        
        # Also save as PNG
        try:
            output_path_png = os.path.join(output_dir, f"{output_name}.png")
            fig.write_image(output_path_png, scale=2)
            print(f"Plot saved to {output_path} and {output_path_png}")
        except Exception as e:
            print(f"Could not save PNG: {e}. HTML version saved to {output_path}")
    
    return fig
