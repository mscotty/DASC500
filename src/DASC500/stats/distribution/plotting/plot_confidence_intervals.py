import os
import plotly.graph_objects as go
import plotly.subplots as sp
import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, Dict, List

def plot_confidence_intervals(
    data: np.ndarray,
    distribution: str,
    confidence_results: Dict,
    output_dir: Optional[str] = None,
    output_name: Optional[str] = None,
    title_name: Optional[str] = None,
    title_font_size: int = 28,
    title_font_name: str = "Times New Roman",
    axis_font_size: int = 24,
    axis_font_name: str = "Times New Roman",
    param_colors: List[str] = None,
    point_estimate_color: str = "red",
    point_estimate_size: int = 10,
    interval_color: str = "blue",
    interval_opacity: float = 0.3,
    show_summary: bool = True,
    summary_font_size: int = 16,
    legend_font_size: int = 16,
    legend_font_name: str = "Times New Roman"
):
    """
    Plot confidence intervals for distribution parameters.
    """
    # Extract confidence interval results
    params = confidence_results.get('parameters', {})
    confidence_level = confidence_results.get('confidence_level', 0.95)
    method = confidence_results.get('method', 'Unknown')
    
    # If no parameters with confidence intervals, return
    if not params:
        print("No parameters with confidence intervals found")
        return
    
    # Set default parameter colors if not provided
    if param_colors is None:
        param_colors = ['blue', 'green', 'purple', 'orange', 'cyan', 'magenta']
    
    # Create figure
    fig = go.Figure()
    
    # Add confidence intervals for each parameter
    for i, (param_name, param_info) in enumerate(params.items()):
        point_estimate = param_info.get('estimate', None)
        lower_bound = param_info.get('lower', None)
        upper_bound = param_info.get('upper', None)
        
        if point_estimate is not None and lower_bound is not None and upper_bound is not None:
            # Add interval as a horizontal bar
            fig.add_trace(
                go.Scatter(
                    x=[lower_bound, upper_bound],
                    y=[param_name, param_name],
                    mode='lines',
                    line=dict(
                        color=param_colors[i % len(param_colors)],
                        width=8
                    ),
                    opacity=interval_opacity,
                    name=f'{param_name} CI'
                )
            )
            
            # Add point estimate
            fig.add_trace(
                go.Scatter(
                    x=[point_estimate],
                    y=[param_name],
                    mode='markers',
                    marker=dict(
                        color=point_estimate_color,
                        size=point_estimate_size,
                        symbol='diamond'
                    ),
                    name=f'{param_name} Estimate'
                )
            )
    
    # Update layout
    if title_name is None:
        title_name = f"{(confidence_level*100):.0f}% Confidence Intervals for {distribution.capitalize()} Parameters"
    
    fig.update_layout(
        title=dict(
            text=title_name,
            font=dict(family=title_font_name, size=title_font_size),
            x=0.5
        ),
        xaxis=dict(
            title=dict(text="Parameter Value", font=dict(size=axis_font_size, family=axis_font_name)),
            tickfont=dict(family=axis_font_name, size=axis_font_size-4)
        ),
        yaxis=dict(
            title=dict(text="Parameter", font=dict(size=axis_font_size, family=axis_font_name)),
            tickfont=dict(family=axis_font_name, size=axis_font_size-4),
            categoryorder='array',
            categoryarray=list(params.keys())
        ),
        legend=dict(
            font=dict(family=legend_font_name, size=legend_font_size)
        ),
        height=100 + 100 * len(params),
        width=800,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=100, r=50, t=80, b=50)
    )
    
    # Add summary annotation if requested
    if show_summary:
        summary_text = f"Distribution: {distribution}<br>"
        summary_text += f"Method: {method}<br>"
        summary_text += f"Confidence Level: {confidence_level*100:.0f}%<br>"
        
        # Add sample size if available
        if 'sample_size' in confidence_results:
            summary_text += f"Sample Size: {confidence_results['sample_size']}<br>"
        
        fig.add_annotation(
            x=0.98,
            y=0.98,
            xref="paper",
            yref="paper",
            text=summary_text,
            showarrow=False,
            font=dict(size=summary_font_size, color="black"),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            borderpad=4,
            align="right"
        )
    
    # Save or display the plot
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        file_path = os.path.join(output_dir, output_name if output_name else f"confidence_intervals_{distribution}.png")
        fig.write_image(file_path, format="png")
    else:
        fig.show()