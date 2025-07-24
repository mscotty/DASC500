import os
import plotly.graph_objects as go
import plotly.subplots as sp
import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, Dict, List

def plot_distribution_comparison(
    data: np.ndarray,
    distributions: List[str],
    fitted_params: Dict[str, List] = None,
    output_dir: Optional[str] = None,
    output_name: Optional[str] = None,
    title_name: Optional[str] = None,
    title_font_size: int = 28,
    title_font_name: str = "Times New Roman",
    axis_font_size: int = 24,
    axis_font_name: str = "Times New Roman",
    hist_color: str = "skyblue",
    line_colors: List[str] = None,
    line_width: int = 2,
    show_params: bool = True,
    param_font_size: int = 16,
    add_qqplots: bool = True,
    legend_font_size: int = 16,
    legend_font_name: str = "Times New Roman"
):
    """
    Plot data histogram with multiple fitted distributions for comparison.
    """
    # Set default line colors if not provided
    if line_colors is None:
        line_colors = ['red', 'green', 'blue', 'purple', 'orange', 'cyan', 'magenta']
    
    # Ensure we have enough colors
    if len(distributions) > len(line_colors):
        import colorsys
        # Generate additional colors if needed
        for i in range(len(distributions) - len(line_colors)):
            h = i / (len(distributions) - len(line_colors))
            r, g, b = colorsys.hsv_to_rgb(h, 0.8, 0.8)
            line_colors.append(f'rgb({int(r*255)}, {int(g*255)}, {int(b*255)})')
    
    # Create main figure for histogram and PDFs
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
    
    # Add fitted distribution PDFs
    x_range = np.linspace(min(data), max(data), 1000)
    
    for i, dist_name in enumerate(distributions):
        try:
            # Get distribution function
            dist = getattr(stats, dist_name)
            
            # Get parameters if provided
            if fitted_params and dist_name in fitted_params:
                params = fitted_params[dist_name]
            else:
                # Fit distribution to data
                params = dist.fit(data)
            
            # Calculate PDF
            y_pdf = dist.pdf(x_range, *params)
            
            # Add PDF line
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=y_pdf,
                    mode='lines',
                    line=dict(color=line_colors[i % len(line_colors)], width=line_width),
                    name=f'{dist_name.capitalize()}'
                )
            )
            
        except Exception as e:
            print(f"Could not plot {dist_name} PDF: {e}")
    
    # Update layout
    if title_name is None:
        title_name = "Distribution Comparison"
        
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
    
    # Create QQ plots if requested
    if add_qqplots:
        # Create a figure with subplots for QQ plots
        n_cols = min(3, len(distributions))
        n_rows = (len(distributions) + n_cols - 1) // n_cols  # Ceiling division
        
        qq_fig = sp.make_subplots(
            rows=n_rows, 
            cols=n_cols,
            subplot_titles=[f"{dist.capitalize()} QQ Plot" for dist in distributions]
        )
        
        # Add QQ plots for each distribution
        for i, dist_name in enumerate(distributions):
            try:
                # Get distribution function
                dist = getattr(stats, dist_name)
                
                # Get parameters if provided
                if fitted_params and dist_name in fitted_params:
                    params = fitted_params[dist_name]
                else:
                    # Fit distribution to data
                    params = dist.fit(data)
                
                # Calculate theoretical quantiles
                quantiles = np.linspace(0.01, 0.99, min(100, len(data)))
                theoretical_quantiles = dist.ppf(quantiles, *params)
                
                # Get empirical quantiles
                empirical_quantiles = np.quantile(data, quantiles)
                
                # Calculate row and column
                row = i // n_cols + 1
                col = i % n_cols + 1
                
                # Add QQ plot
                qq_fig.add_trace(
                    go.Scatter(
                        x=theoretical_quantiles,
                        y=empirical_quantiles,
                        mode='markers',
                        marker=dict(
                            color=line_colors[i % len(line_colors)],
                            size=8,
                            opacity=0.7
                        ),
                        name=f'{dist_name.capitalize()}'
                    ),
                    row=row, col=col
                )
                
                # Add reference line
                min_val = min(np.min(theoretical_quantiles), np.min(empirical_quantiles))
                max_val = max(np.max(theoretical_quantiles), np.max(empirical_quantiles))
                
                qq_fig.add_trace(
                    go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        line=dict(color='red', width=2, dash='dash'),
                        name='Reference Line'
                    ),
                    row=row, col=col
                )
                
                # Update axes
                qq_fig.update_xaxes(
                    title_text="Theoretical Quantiles",
                    title_font=dict(size=axis_font_size-8, family=axis_font_name),
                    row=row, col=col
                )
                
                qq_fig.update_yaxes(
                    title_text="Sample Quantiles",
                    title_font=dict(size=axis_font_size-8, family=axis_font_name),
                    row=row, col=col
                )
                
            except Exception as e:
                print(f"Could not create QQ plot for {dist_name}: {e}")
        
        # Update QQ plots layout
        qq_fig.update_layout(
            title=dict(
                text="QQ Plots for Fitted Distributions",
                font=dict(family=title_font_name, size=title_font_size),
                x=0.5
            ),
            height=400 * n_rows,
            width=300 * n_cols,
            plot_bgcolor="white",
            paper_bgcolor="white",
            showlegend=False
        )
        
        # Update subplot titles font
        for i in range(len(qq_fig.layout.annotations)):
            qq_fig.layout.annotations[i].font.size = axis_font_size - 6
    
    # Save or display the plots
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save main plot
        main_file_path = os.path.join(output_dir, output_name if output_name else "distribution_comparison.png")
        fig.write_image(main_file_path, format="png")
        
        # Save QQ plots if created
        if add_qqplots:
            qq_file_path = os.path.join(output_dir, f"qq_plots_{'_'.join(distributions)}.png")
            qq_fig.write_image(qq_file_path, format="png")
    else:
        fig.show()
        if add_qqplots:
            qq_fig.show()

