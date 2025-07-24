import os
import plotly.graph_objects as go
import plotly.subplots as sp
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Union, Optional

def plot_transformations(
    data: np.ndarray,
    transformation_results: Dict,
    output_dir: Optional[str] = None,
    output_name: Optional[str] = "transformation_comparison.png",
    title_name: Optional[str] = "Distribution Transformation Comparison",
    title_font_size: int = 28,
    title_font_name: str = "Times New Roman",
    subplot_title_font_size: int = 18,
    axis_font_size: int = 16,
    axis_font_name: str = "Times New Roman",
    hist_color: str = "skyblue",
    best_hist_color: str = "green",
    original_hist_color: str = "blue",
    line_color: str = "red",
    line_width: int = 2,
    show_p_values: bool = True,
    p_value_font_size: int = 14,
    p_value_precision: int = 4,
    add_qqplots: bool = True,
    qqplot_line_color: str = "red",
    qqplot_marker_color: str = "blue",
    qqplot_marker_size: int = 8,
    qqplot_marker_opacity: float = 0.7,
    show_improvement: bool = True
):
    """
    @brief Plot histograms and QQ plots for original and transformed data.
    @param[in] data (np.ndarray) Original data array.
    @param[in] transformation_results (Dict) Results from optimal_transformation method.
    @param[in] output_dir (str) Directory to save the plot as a PNG file. If None, the plot is displayed interactively.
    @param[in] output_name (str) Custom file name for the saved plot.
    @param[in] title_name (str) Custom title for the chart.
    @param[in] title_font_size (int) Font size for the chart title.
    @param[in] title_font_name (str) Font name for the chart title.
    @param[in] subplot_title_font_size (int) Font size for subplot titles.
    @param[in] axis_font_size (int) Font size for axis labels.
    @param[in] axis_font_name (str) Font name for axis labels.
    @param[in] hist_color (str) Color for histograms of transformed data.
    @param[in] best_hist_color (str) Color for the best transformation histogram.
    @param[in] original_hist_color (str) Color for the original data histogram.
    @param[in] line_color (str) Color for normal distribution line.
    @param[in] line_width (int) Width of normal distribution line.
    @param[in] show_p_values (bool) Whether to display p-values on the plots.
    @param[in] p_value_font_size (int) Font size for p-value annotations.
    @param[in] p_value_precision (int) Number of decimal places for p-values.
    @param[in] add_qqplots (bool) Whether to include QQ plots for each transformation.
    @param[in] qqplot_line_color (str) Color for QQ plot reference line.
    @param[in] qqplot_marker_color (str) Color for QQ plot markers.
    @param[in] qqplot_marker_size (int) Size of QQ plot markers.
    @param[in] qqplot_marker_opacity (float) Opacity of QQ plot markers.
    @param[in] show_improvement (bool) Whether to display improvement factor.
    """
    
    # Extract results
    original_p = transformation_results['original_p_value']
    best_transform = transformation_results['best_transformation']
    best_p = transformation_results['best_p_value']
    transformed_data = transformation_results['transformed_data']
    all_results = transformation_results['all_results']
    improvement_factor = transformation_results.get('improvement_factor', 0)
    
    # Determine number of transformations to plot
    transformations = ['none'] + list(all_results.keys())
    
    # Calculate number of rows and columns for subplots
    n_plots = len(transformations)
    n_cols = 2 if add_qqplots else 1
    n_rows = n_plots
    
    # Create subplots
    fig = sp.make_subplots(
        rows=n_rows, 
        cols=n_cols,
        subplot_titles=[f"{t.capitalize()} Transformation" for t in transformations for _ in range(n_cols)],
        specs=[[{"type": "histogram"}, {"type": "scatter"}] if add_qqplots else [{"type": "histogram"}] 
              for _ in range(n_rows)]
    )
    
    # Plot for each transformation
    for i, transform in enumerate(transformations):
        row = i + 1
        
        # Get the appropriate data and p-value
        if transform == 'none':
            plot_data = data
            p_value = original_p
            hist_color_use = original_hist_color
        else:
            # Skip if there was an error with this transformation
            if 'error' in all_results[transform]:
                continue
                
            if transform == best_transform:
                plot_data = transformed_data
                hist_color_use = best_hist_color
            else:
                # Try to get transformed data from results
                try:
                    transform_result = all_results[transform]
                    plot_data = transform_result.get('transformed_data', data)
                    hist_color_use = hist_color
                except:
                    continue
                    
            p_value = all_results[transform].get('p_value', 0)
        
        # Add histogram
        # Calculate bin count based on data size
        bin_count = min(50, max(10, int(np.sqrt(len(plot_data)))))
        
        # Fit normal distribution to the data
        mu, std = np.mean(plot_data), np.std(plot_data)
        x_norm = np.linspace(np.min(plot_data), np.max(plot_data), 100)
        y_norm = stats.norm.pdf(x_norm, mu, std)
        
        # Scale the normal PDF to match histogram height
        hist, bin_edges = np.histogram(plot_data, bins=bin_count, density=True)
        max_height = np.max(hist) if len(hist) > 0 else 1
        y_norm = y_norm * (max_height / np.max(y_norm)) if np.max(y_norm) > 0 else y_norm
        
        # Add histogram
        fig.add_trace(
            go.Histogram(
                x=plot_data,
                nbinsx=bin_count,
                histnorm='probability density',
                marker_color=hist_color_use,
                opacity=0.7,
                name=f"{transform.capitalize()}"
            ),
            row=row, col=1
        )
        
        # Add normal distribution line
        fig.add_trace(
            go.Scatter(
                x=x_norm,
                y=y_norm,
                mode='lines',
                line=dict(color=line_color, width=line_width),
                name='Normal Fit'
            ),
            row=row, col=1
        )
        
        # Add p-value annotation
        if show_p_values:
            annotation_text = f"p-value: {p_value:.{p_value_precision}f}"
            if transform == best_transform and show_improvement:
                annotation_text += f"<br>Improvement: {improvement_factor:.2f}x"
                
            fig.add_annotation(
                x=0.95,
                y=0.95,
                xref=f"x{row}" if row > 1 else "x",
                yref=f"y{row}" if row > 1 else "y",
                text=annotation_text,
                showarrow=False,
                font=dict(size=p_value_font_size, color="black"),
                bgcolor="white",
                bordercolor="black",
                borderwidth=1,
                borderpad=4,
                align="right"
            )
        
        # Add QQ Plot if requested
        if add_qqplots:
            # Calculate theoretical quantiles
            quantiles = np.linspace(0.01, 0.99, min(100, len(plot_data)))
            theoretical_quantiles = stats.norm.ppf(quantiles)
            
            # Get empirical quantiles
            empirical_quantiles = np.quantile(plot_data, quantiles)
            
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
                row=row, col=2
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
                row=row, col=2
            )
            
            # Update QQ plot axes
            fig.update_xaxes(
                title_text="Theoretical Quantiles",
                title_font=dict(size=axis_font_size, family=axis_font_name),
                row=row, col=2
            )
            
            fig.update_yaxes(
                title_text="Sample Quantiles",
                title_font=dict(size=axis_font_size, family=axis_font_name),
                row=row, col=2
            )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=title_name,
            font=dict(family=title_font_name, size=title_font_size),
            x=0.5
        ),
        showlegend=False,
        height=300 * n_rows,
        width=1000 if add_qqplots else 600,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    # Update subplot titles font
    for i in range(len(fig.layout.annotations)):
        fig.layout.annotations[i].font.size = subplot_title_font_size
    
    # Update x and y axis titles for histograms
    for i in range(n_rows):
        fig.update_xaxes(
            title_text="Value",
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
        file_path = os.path.join(output_dir, output_name)
        fig.write_image(file_path, format="png")
    else:
        fig.show()
