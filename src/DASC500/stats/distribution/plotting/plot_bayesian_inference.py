import os
import plotly.graph_objects as go
import plotly.subplots as sp
import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, Dict

def plot_bayesian_inference(
    data: np.ndarray,
    bayesian_results: Dict,
    output_dir: Optional[str] = None,
    output_name: Optional[str] = None,
    title_name: Optional[str] = None,
    title_font_size: int = 28,
    title_font_name: str = "Times New Roman",
    axis_font_size: int = 24,
    axis_font_name: str = "Times New Roman",
    hist_color: str = "skyblue",
    posterior_color: str = "blue",
    credible_interval_color: str = "rgba(0, 0, 255, 0.2)",
    show_hdi: bool = True,
    hdi_prob: float = 0.95,
    show_summary: bool = True,
    summary_font_size: int = 16,
    legend_font_size: int = 16,
    legend_font_name: str = "Times New Roman"
):
    # Extract Bayesian inference results
    trace = bayesian_results.get('trace', None)
    summary = bayesian_results.get('summary', None)
    distribution = bayesian_results.get('distribution', 'unknown')
    
    if trace is None:
        print("No trace found in Bayesian results")
        return
    
    # Create figure with subplots - one for each parameter
    param_names = list(trace.varnames)
    # Filter out likelihood and other non-parameter variables
    param_names = [p for p in param_names if not (p.endswith('_') or p == 'likelihood')]
    
    n_params = len(param_names)
    fig = sp.make_subplots(
        rows=n_params, 
        cols=1,
        subplot_titles=[f"Posterior for {p}" for p in param_names]
    )
    
    # Plot posterior for each parameter
    for i, param in enumerate(param_names):
        # Extract parameter samples
        param_samples = trace[param]
        
        # Add histogram of posterior
        fig.add_trace(
            go.Histogram(
                x=param_samples,
                histnorm='probability density',
                marker_color=posterior_color,
                opacity=0.7,
                name=f'{param} Posterior'
            ),
            row=i+1, col=1
        )
        
        # Add HDI (Highest Density Interval) if requested
        if show_hdi:
            # Calculate HDI
            param_samples_sorted = np.sort(param_samples)
            n_samples = len(param_samples_sorted)
            n_samples_in_hdi = int(np.floor(hdi_prob * n_samples))
            hdi_width = [param_samples_sorted[i+n_samples_in_hdi] - param_samples_sorted[i] 
                        for i in range(n_samples - n_samples_in_hdi)]
            hdi_start_idx = np.argmin(hdi_width)
            hdi_start = param_samples_sorted[hdi_start_idx]
            hdi_end = param_samples_sorted[hdi_start_idx + n_samples_in_hdi]
            
            # Add HDI shading
            fig.add_shape(
                type="rect",
                xref=f"x{i+1}" if i > 0 else "x",
                yref=f"y{i+1}" if i > 0 else "y",
                x0=hdi_start,
                x1=hdi_end,
                y0=0,
                y1=1,
                fillcolor=credible_interval_color,
                opacity=0.5,
                layer="below",
                line_width=0
            )
            
            # Add HDI annotation
            fig.add_annotation(
                x=(hdi_start + hdi_end) / 2,
                y=0.9,
                xref=f"x{i+1}" if i > 0 else "x",
                yref=f"y{i+1}" if i > 0 else "y",
                text=f"{hdi_prob*100:.0f}% HDI<br>[{hdi_start:.4f}, {hdi_end:.4f}]",
                showarrow=False,
                font=dict(size=summary_font_size, color="black"),
                bgcolor="white",
                bordercolor="black",
                borderwidth=1,
                borderpad=4,
                align="center"
            )
        
        # Add summary statistics if available and requested
        if show_summary and summary is not None and param in summary.index:
            param_summary = summary.loc[param]
            
            summary_text = f"Mean: {param_summary['mean']:.4f}<br>"
            summary_text += f"SD: {param_summary['sd']:.4f}<br>"
            if 'hpd_3%' in param_summary and 'hpd_97%' in param_summary:
                summary_text += f"94% HPD: [{param_summary['hpd_3%']:.4f}, {param_summary['hpd_97%']:.4f}]"
            
            fig.add_annotation(
                x=0.02,
                y=0.98,
                xref=f"x{i+1}" if i > 0 else "x",
                yref=f"y{i+1}" if i > 0 else "y",
                text=summary_text,
                showarrow=False,
                font=dict(size=summary_font_size, color="black"),
                bgcolor="white",
                bordercolor="black",
                borderwidth=1,
                borderpad=4,
                align="left"
            )
    
    # Set title
    if title_name is None:
        title_name = f"Bayesian Inference for {distribution.capitalize()} Distribution"
    
        # Update layout
    fig.update_layout(
        title=dict(
            text=title_name,
            font=dict(family=title_font_name, size=title_font_size),
            x=0.5
        ),
        height=400 * n_params,
        width=800,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=50, r=50, t=80, b=50),
        showlegend=False
    )
    
    # Update subplot titles font
    for i in range(len(fig.layout.annotations)):
        fig.layout.annotations[i].font.size = axis_font_size
    
    # Update x and y axis labels for each subplot
    for i in range(n_params):
        fig.update_xaxes(
            title_text=param_names[i],
            title_font=dict(size=axis_font_size, family=axis_font_name),
            row=i+1, col=1
        )
        
        fig.update_yaxes(
            title_text="Density",
            title_font=dict(size=axis_font_size, family=axis_font_name),
            row=i+1, col=1
        )
    
    # Save or display the plot
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        file_path = os.path.join(output_dir, output_name if output_name else f"bayesian_inference_{distribution}.png")
        fig.write_image(file_path, format="png")
    else:
        fig.show()

