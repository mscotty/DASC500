import os
import plotly.graph_objects as go
import plotly.subplots as sp
import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, Dict

def plot_distribution_distance(
    data1: np.ndarray,
    data2: np.ndarray,
    distance_results: Dict,
    output_dir: Optional[str] = None,
    output_name: Optional[str] = None,
    title_name: Optional[str] = None,
    title_font_size: int = 28,
    title_font_name: str = "Times New Roman",
    axis_font_size: int = 24,
    axis_font_name: str = "Times New Roman",
    hist1_color: str = "blue",
    hist2_color: str = "red",
    hist_opacity: float = 0.7,
    show_metrics: bool = True,
    metric_font_size: int = 16,
    add_ecdf: bool = True,
    ecdf1_color: str = "blue",
    ecdf2_color: str = "red",
    line_width: int = 2,
    legend_font_size: int = 16,
    legend_font_name: str = "Times New Roman",
    data1_name: str = "Distribution 1",
    data2_name: str = "Distribution 2"
):
    # Determine number of subplots
    n_plots = 2 if add_ecdf else 1
    
    # Create subplots
    fig = sp.make_subplots(
        rows=1, 
        cols=n_plots,
        subplot_titles=["Histogram Comparison"] + (["ECDF Comparison"] if add_ecdf else [])
    )
    
    # Calculate bin count based on data size
    bin_count = min(50, max(10, int(np.sqrt(min(len(data1), len(data2))))))
    
    # Add histograms
    fig.add_trace(
        go.Histogram(
            x=data1,
            nbinsx=bin_count,
            histnorm='probability density',
            marker_color=hist1_color,
            opacity=hist_opacity,
            name=data1_name
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Histogram(
            x=data2,
            nbinsx=bin_count,
            histnorm='probability density',
            marker_color=hist2_color,
            opacity=hist_opacity,
            name=data2_name
        ),
        row=1, col=1
    )
    
    # Add ECDF if requested
    if add_ecdf:
        # Calculate ECDFs
        x1_sorted = np.sort(data1)
        y1_ecdf = np.arange(1, len(data1) + 1) / len(data1)
        
        x2_sorted = np.sort(data2)
        y2_ecdf = np.arange(1, len(data2) + 1) / len(data2)
        
        # Add ECDF lines
        fig.add_trace(
            go.Scatter(
                x=x1_sorted,
                y=y1_ecdf,
                mode='lines',
                line=dict(color=ecdf1_color, width=line_width),
                name=f'{data1_name} ECDF'
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=x2_sorted,
                y=y2_ecdf,
                mode='lines',
                line=dict(color=ecdf2_color, width=line_width),
                name=f'{data2_name} ECDF'
            ),
            row=1, col=2
        )
        
        # Update ECDF axes
        fig.update_xaxes(
            title_text="Value",
            title_font=dict(size=axis_font_size, family=axis_font_name),
            row=1, col=2
        )
        
        fig.update_yaxes(
            title_text="Cumulative Probability",
            title_font=dict(size=axis_font_size, family=axis_font_name),
            row=1, col=2
        )
    
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
    
    # Add distance metrics annotation if requested
    if show_metrics and distance_results:
        metric_text = "Distance Metrics:<br>"
        
        # Add KS test results if available
        if 'ks_statistic' in distance_results:
            metric_text += f"KS Statistic: {distance_results['ks_statistic']:.4f}<br>"
        if 'ks_p_value' in distance_results:
            metric_text += f"KS p-value: {distance_results['ks_p_value']:.4f}<br>"
            
        # Add Wasserstein distance if available
        if 'wasserstein_distance' in distance_results:
            metric_text += f"Wasserstein: {distance_results['wasserstein_distance']:.4f}<br>"
            
        # Add Energy distance if available
        if 'energy_distance' in distance_results:
            metric_text += f"Energy: {distance_results['energy_distance']:.4f}<br>"
            
        # Add KL divergence if available
        if 'kl_divergence' in distance_results:
            metric_text += f"KL Divergence: {distance_results['kl_divergence']:.4f}<br>"
            
        # Add JS divergence if available
        if 'js_divergence' in distance_results:
            metric_text += f"JS Divergence: {distance_results['js_divergence']:.4f}<br>"
        
        fig.add_annotation(
            x=0.98,
            y=0.98,
            xref="x",
            yref="y",
            text=metric_text,
            showarrow=False,
            font=dict(size=metric_font_size, color="black"),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            borderpad=4,
            align="right"
        )
    
    # Set title
    if title_name is None:
        title_name = f"Distribution Distance Comparison"
    
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
        margin=dict(l=50, r=50, t=80, b=50),
        barmode='overlay'
    )
    
    # Update subplot titles font
    for i in range(len(fig.layout.annotations)):
        fig.layout.annotations[i].font.size = axis_font_size
    
    # Save or display the plot
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        file_path = os.path.join(output_dir, output_name if output_name else "distribution_distance.png")
        fig.write_image(file_path, format="png")
    else:
        fig.show()
        