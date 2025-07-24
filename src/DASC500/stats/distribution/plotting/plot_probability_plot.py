import os
import plotly.graph_objects as go
import plotly.subplots as sp
import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, Dict, List

def plot_probability_plot(
    data: np.ndarray,
    distribution: str = 'norm',
    fitted_params: List = None,
    output_dir: Optional[str] = None,
    output_name: Optional[str] = None,
    title_name: Optional[str] = None,
    title_font_size: int = 28,
    title_font_name: str = "Times New Roman",
    axis_font_size: int = 24,
    axis_font_name: str = "Times New Roman",
    marker_color: str = "blue",
    marker_size: int = 8,
    marker_opacity: float = 0.7,
    line_color: str = "red",
    line_width: int = 2,
    show_stats: bool = True,
    stats_font_size: int = 16,
    legend_font_size: int = 16,
    legend_font_name: str = "Times New Roman",
    plot_type: str = 'qq'  # 'qq' or 'pp'
):
    """
    Create probability plots (QQ or PP) for data against a theoretical distribution.
    """
    # Create figure
    fig = go.Figure()
    
    try:
        # Get distribution function
        dist = getattr(stats, distribution)
        
        if plot_type.lower() == 'qq':
            # QQ plot (quantile-quantile)
            if fitted_params is None:
                # Fit distribution if parameters not provided
                fitted_params = dist.fit(data)
                
            # Calculate theoretical quantiles
            quantiles = np.linspace(0.01, 0.99, min(100, len(data)))
            theoretical_quantiles = dist.ppf(quantiles, *fitted_params)
            
            # Get empirical quantiles
            empirical_quantiles = np.quantile(data, quantiles)
            
            # Add QQ plot
            fig.add_trace(
                go.Scatter(
                    x=theoretical_quantiles,
                    y=empirical_quantiles,
                    mode='markers',
                    marker=dict(
                        color=marker_color,
                        size=marker_size,
                        opacity=marker_opacity
                    ),
                    name='QQ Plot'
                )
            )
            
            # Add reference line
            min_val = min(np.min(theoretical_quantiles), np.min(empirical_quantiles))
            max_val = max(np.max(theoretical_quantiles), np.max(empirical_quantiles))
            
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    line=dict(color=line_color, width=line_width, dash='dash'),
                    name='Reference Line'
                )
            )
            
            # Calculate R² for QQ plot
            slope, intercept, r_value, p_value, std_err = stats.linregress(theoretical_quantiles, empirical_quantiles)
            r_squared = r_value**2
            
            # Set axis titles
            x_title = "Theoretical Quantiles"
            y_title = "Sample Quantiles"
            
        elif plot_type.lower() == 'pp':
            # PP plot (probability-probability)
            if fitted_params is None:
                # Fit distribution if parameters not provided
                fitted_params = dist.fit(data)
                
            # Calculate empirical CDF
            x_sorted = np.sort(data)
            y_ecdf = np.arange(1, len(data) + 1) / len(data)
            
            # Calculate theoretical CDF
            y_cdf = dist.cdf(x_sorted, *fitted_params)
            
            # Add PP plot
            fig.add_trace(
                go.Scatter(
                    x=y_cdf,
                    y=y_ecdf,
                    mode='markers',
                    marker=dict(
                        color=marker_color,
                        size=marker_size,
                        opacity=marker_opacity
                    ),
                    name='PP Plot'
                )
            )
            
            # Add reference line
            fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode='lines',
                    line=dict(color=line_color, width=line_width, dash='dash'),
                    name='Reference Line'
                )
            )
            
            # Calculate R² for PP plot
            slope, intercept, r_value, p_value, std_err = stats.linregress(y_cdf, y_ecdf)
            r_squared = r_value**2
            
            # Set axis titles
            x_title = "Theoretical Probability"
            y_title = "Empirical Probability"
            
        else:
            raise ValueError(f"Invalid plot_type: {plot_type}. Must be 'qq' or 'pp'.")
        
        # Add stats annotation if requested
        if show_stats:
            stats_text = f"Distribution: {distribution}<br>"
            stats_text += f"R²: {r_squared:.4f}<br>"
            stats_text += f"Slope: {slope:.4f}<br>"
            stats_text += f"Intercept: {intercept:.4f}<br>"
            
            fig.add_annotation(
                x=0.02,
                y=0.98,
                xref="paper",
                yref="paper",
                text=stats_text,
                showarrow=False,
                font=dict(
                    size=stats_font_size,
                    family=title_font_name
                ),
                align="left",
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="black",
                borderwidth=1,
                borderpad=4
            )
        
        # Update layout
        plot_title = title_name if title_name else f"{plot_type.upper()} Plot for {distribution.capitalize()} Distribution"
        
        fig.update_layout(
            title=dict(
                text=plot_title,
                font=dict(
                    size=title_font_size,
                    family=title_font_name
                )
            ),
            xaxis=dict(
                title=dict(
                    text=x_title,
                    font=dict(
                        size=axis_font_size,
                        family=axis_font_name
                    )
                )
            ),
            yaxis=dict(
                title=dict(
                    text=y_title,
                    font=dict(
                        size=axis_font_size,
                        family=axis_font_name
                    )
                )
            ),
            legend=dict(
                font=dict(
                    size=legend_font_size,
                    family=legend_font_name
                )
            ),
            template="plotly_white"
        )
        
        # Save figure if output directory and name are provided
        if output_dir and output_name:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            output_path = os.path.join(output_dir, f"{output_name}.html")
            fig.write_html(output_path)
            
            # Also save as PNG
            output_path_png = os.path.join(output_dir, f"{output_name}.png")
            fig.write_image(output_path_png, scale=2)
            
            print(f"Plot saved to {output_path} and {output_path_png}")
        
        return fig
    
    except Exception as e:
        print(f"Error creating probability plot: {e}")
        return None
