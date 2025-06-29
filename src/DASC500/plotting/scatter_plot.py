import os
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
from scipy import stats

def plot_scatter(data,
                 x_column,
                 y_column,
                 color_column=None,
                 size_column=None,
                 output_dir=None,
                 output_name=None,
                 marker_color='blue',
                 marker_size=10,
                 marker_opacity=0.7,
                 add_trendline=False,
                 trendline_color='red',
                 trendline_width=2,
                 show_r2=True,
                 title_name=None,
                 title_font_size=28,
                 title_font_name='Times New Roman',
                 x_axis_name=None,
                 x_axis_font_size=24,
                 x_axis_font_name='Times New Roman',
                 y_axis_name=None,
                 y_axis_font_size=24,
                 y_axis_font_name='Times New Roman',
                 legend_title=None,
                 legend_font_size=16,
                 legend_font_name='Times New Roman'):
    """
    @brief Plot a scatter plot of the specified data columns.
    @param[in] data DataFrame containing the data to be plotted.
    @param[in] x_column (str) Column name for the x-axis.
    @param[in] y_column (str) Column name for the y-axis.
    @param[in] color_column (str) Optional column name for coloring points.
    @param[in] size_column (str) Optional column name for sizing points.
    @param[in] output_dir (str) Directory to save the plot as a PNG file. If None, the plot is displayed interactively.
    @param[in] output_name (str) Custom file name for the saved plot.
    @param[in] marker_color (str) Color for the markers when not using color_column.
    @param[in] marker_size (int) Size of the markers when not using size_column.
    @param[in] marker_opacity (float) Opacity of the markers (0-1).
    @param[in] add_trendline (bool) Whether to add a linear trendline.
    @param[in] trendline_color (str) Color for the trendline.
    @param[in] trendline_width (int) Width of the trendline.
    @param[in] show_r2 (bool) Whether to display R² value with the trendline.
    @param[in] title_name (str) Custom title for the chart. If None, a generic title is used.
    @param[in] title_font_size (int) Font size for the chart title.
    @param[in] title_font_name (str) Font name for the chart title.
    @param[in] x_axis_name (str) Custom label for the x-axis. If None, the x_column name is used.
    @param[in] x_axis_font_size (int) Font size for the x-axis labels.
    @param[in] x_axis_font_name (str) Font name for the x-axis labels.
    @param[in] y_axis_name (str) Custom label for the y-axis. If None, the y_column name is used.
    @param[in] y_axis_font_size (int) Font size for the y-axis labels.
    @param[in] y_axis_font_name (str) Font name for the y-axis labels.
    @param[in] legend_title (str) Custom title for the legend. If None, a generic title is used.
    @param[in] legend_font_size (int) Font size for the legend.
    @param[in] legend_font_name (str) Font name for the legend.
    """
    fig = go.Figure()

    # Prepare marker settings
    marker_dict = dict(
        opacity=marker_opacity,
        line=dict(width=1, color='black')
    )
    
    # Set marker color
    if color_column is not None:
        marker_dict['color'] = data[color_column]
        marker_dict['colorscale'] = 'Viridis'
        marker_dict['colorbar'] = dict(
            title=color_column,
            titlefont=dict(family=legend_font_name, size=legend_font_size)
        )
    else:
        marker_dict['color'] = marker_color
    
    # Set marker size
    if size_column is not None:
        # Normalize size values between 5 and 25
        sizes = data[size_column]
        if sizes.min() != sizes.max():  # Avoid division by zero
            normalized_sizes = 5 + 20 * (sizes - sizes.min()) / (sizes.max() - sizes.min())
        else:
            normalized_sizes = [marker_size] * len(sizes)
        marker_dict['size'] = normalized_sizes
        marker_dict['sizemode'] = 'diameter'
        marker_dict['sizeref'] = 1.0
    else:
        marker_dict['size'] = marker_size

    # Add scatter trace
    fig.add_trace(
        go.Scatter(
            x=data[x_column],
            y=data[y_column],
            mode='markers',
            marker=marker_dict,
            name='Data Points'
        )
    )

    # Add trendline if requested
    if add_trendline and len(data) > 1:
        # Remove NaN values for regression
        valid_data = data[[x_column, y_column]].dropna()
        x = valid_data[x_column]
        y = valid_data[y_column]
        
        if len(x) > 1:  # Need at least 2 points for regression
            # Calculate linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            r_squared = r_value**2
            
            # Create x values for the line
            x_line = np.array([min(x), max(x)])
            y_line = slope * x_line + intercept
            
            # Add trendline annotation
            trendline_name = f'Trendline: y = {slope:.3f}x + {intercept:.3f}'
            if show_r2:
                trendline_name += f', R² = {r_squared:.3f}'
            
            # Add trendline trace
            fig.add_trace(
                go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode='lines',
                    line=dict(color=trendline_color, width=trendline_width),
                    name=trendline_name
                )
            )

    # Update layout
    fig.update_layout(
        title=dict(
            text=title_name if title_name else f"Scatter Plot of {y_column} vs {x_column}",
            font=dict(family=title_font_name, size=title_font_size),
            x=0.5
        ),
        xaxis=dict(
            title=dict(text=x_axis_name if x_axis_name else x_column, 
                      font=dict(family=x_axis_font_name, size=x_axis_font_size)),
            tickfont=dict(family=x_axis_font_name, size=x_axis_font_size-4),
            gridcolor="lightgray"
        ),
        yaxis=dict(
            title=dict(text=y_axis_name if y_axis_name else y_column, 
                      font=dict(family=y_axis_font_name, size=y_axis_font_size)),
            tickfont=dict(family=y_axis_font_name, size=y_axis_font_size-4),
            gridcolor="lightgray"
        ),
        legend=dict(
            title=dict(
                text=legend_title if legend_title else "Legend",
                font=dict(family=legend_font_name, size=legend_font_size)
            ),
            font=dict(family=legend_font_name, size=legend_font_size-2)
        ),
        font=dict(family="Times New Roman", size=14),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=50, r=50, t=50, b=50)
    )

    # Add grid lines
    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor="lightgray")
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor="lightgray")

    # Save or display the plot
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        file_path = os.path.join(output_dir, output_name if output_name else f"scatter_plot_{x_column}_vs_{y_column}.png")
        fig.write_image(file_path, format="png", width=800, height=600)
    else:
        fig.show()
