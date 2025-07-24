import os
import plotly.graph_objects as go
import plotly.subplots as sp
import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, Dict, List

def plot_mixture_model(
    data: np.ndarray,
    mixture_results: Dict,
    distributions: List[str] = None,
    output_dir: Optional[str] = None,
    output_name: Optional[str] = None,
    title_name: Optional[str] = None,
    title_font_size: int = 28,
    title_font_name: str = "Times New Roman",
    axis_font_size: int = 24,
    axis_font_name: str = "Times New Roman",
    hist_color: str = "skyblue",
    component_colors: List[str] = None,
    mixture_color: str = "red",
    line_width: int = 2,
    show_components: bool = True,
    show_metrics: bool = True,
    metric_font_size: int = 16,
    legend_font_size: int = 16,
    legend_font_name: str = "Times New Roman"
):
    # Extract mixture model results
    weights = mixture_results.get('weights', [])
    component_params = mixture_results.get('component_params', [])
    n_components = mixture_results.get('n_components', len(weights))
    bic = mixture_results.get('bic', None)
    aic = mixture_results.get('aic', None)
    
    # Set default component colors if not provided
    if component_colors is None:
        component_colors = ['blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'brown']
        # Ensure we have enough colors
        if n_components > len(component_colors):
            import colorsys
            # Generate additional colors if needed
            for i in range(n_components - len(component_colors)):
                h = i / (n_components - len(component_colors))
                r, g, b = colorsys.hsv_to_rgb(h, 0.8, 0.8)
                component_colors.append(f'rgb({int(r*255)}, {int(g*255)}, {int(b*255)})')
    
    # Create figure
    fig = go.Figure()
    
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
        )
    )
    
    # Create x values for PDF
    x_range = np.linspace(min(data), max(data), 1000)
    
    # Add component PDFs if requested
    if show_components and len(component_params) > 0:
        # Determine distribution type if not provided
        if distributions is None:
            # Default to normal distribution for each component
            distributions = ['norm'] * n_components
        elif isinstance(distributions, str):
            # If a single distribution is provided, use it for all components
            distributions = [distributions] * n_components
        
        # Ensure we have enough distributions
        if len(distributions) < n_components:
            distributions.extend(['norm'] * (n_components - len(distributions)))
        
        # Plot each component
        for i, (weight, params, dist_name) in enumerate(zip(weights, component_params, distributions)):
            try:
                # Get distribution function
                dist = getattr(stats, dist_name)
                
                # Calculate component PDF
                if isinstance(params, dict):
                    loc = params.get('loc', 0)
                    scale = params.get('scale', 1)
                    
                    # Extract shape parameters
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
                    # If params is not a dict, try using it directly
                    y_pdf = dist.pdf(x_range, *params)
                
                # Scale by weight
                y_pdf = y_pdf * weight
                
                # Add component PDF
                fig.add_trace(
                    go.Scatter(
                        x=x_range,
                        y=y_pdf,
                        mode='lines',
                        line=dict(color=component_colors[i % len(component_colors)], width=line_width, dash='dash'),
                        name=f'Component {i+1} (w={weight:.2f})'
                    )
                )
                
            except Exception as e:
                print(f"Could not plot component {i+1}: {e}")
    
    # Calculate and add mixture PDF
    try:
        mixture_pdf = np.zeros_like(x_range)
        
        for i, (weight, params) in enumerate(zip(weights, component_params)):
            # Get distribution
            dist_name = distributions[i] if distributions else 'norm'
            dist = getattr(stats, dist_name)
            
            # Calculate component PDF
            if isinstance(params, dict):
                loc = params.get('loc', 0)
                scale = params.get('scale', 1)
                
                # Extract shape parameters
                shape_params = []
                for param in ['a', 'b', 'c', 'd', 'k', 's', 'df', 'alpha', 'beta']:
                    if param in params:
                        shape_params.append(params[param])
                
                # Calculate PDF
                if shape_params:
                    component_pdf = dist.pdf(x_range, *shape_params, loc=loc, scale=scale)
                else:
                    component_pdf = dist.pdf(x_range, loc=loc, scale=scale)
            else:
                # If params is not a dict, try using it directly
                component_pdf = dist.pdf(x_range, *params)
            
            # Add weighted component to mixture
            mixture_pdf += weight * component_pdf
        
        # Add mixture PDF
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=mixture_pdf,
                mode='lines',
                line=dict(color=mixture_color, width=line_width+1),
                name='Mixture PDF'
            )
        )
        
    except Exception as e:
        print(f"Could not plot mixture PDF: {e}")
    
    # Add metrics annotation if requested
    if show_metrics and (bic is not None or aic is not None):
        metric_text = f"Number of Components: {n_components}<br>"
        if bic is not None:
            metric_text += f"BIC: {bic:.2f}<br>"
        if aic is not None:
            metric_text += f"AIC: {aic:.2f}<br>"
            
        fig.add_annotation(
            x=0.98,
            y=0.98,
            xref="paper",
            yref="paper",
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
        title_name = f"Mixture Model with {n_components} Components"
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=title_name,
            font=dict(family=title_font_name, size=title_font_size),
            x=0.5
        ),
        xaxis=dict(
            title=dict(text="Value", font=dict(size=axis_font_size, family=axis_font_name)),
            tickfont=dict(family=axis_font_name, size=axis_font_size-4)
        ),
        yaxis=dict(
            title=dict(text="Density", font=dict(size=axis_font_size, family=axis_font_name)),
            tickfont=dict(family=axis_font_name, size=axis_font_size-4)
        ),
        legend=dict(
            font=dict(family=legend_font_name, size=legend_font_size)
        ),
        height=600,
        width=900,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    # Save or display the plot
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        file_path = os.path.join(output_dir, output_name if output_name else f"mixture_model_{n_components}_components.png")
        fig.write_image(file_path, format="png")
    else:
        fig.show()

