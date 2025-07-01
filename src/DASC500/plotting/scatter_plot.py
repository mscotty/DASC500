import os
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
import pandas as pd
from scipy import stats

def plot_scatter(data,
                 x_column,
                 y_column,
                 color_column=None,
                 size_column=None,
                 label_column=None,
                 label_points=None,  # List/array of indices, boolean mask, or filtered DataFrame
                 output_dir=None,
                 output_name=None,
                 marker_color='blue',
                 marker_size=10,
                 marker_opacity=0.7,
                 labeled_marker_color='red',
                 labeled_marker_size=12,
                 add_trendline=False,
                 trendline_color='red',
                 trendline_width=2,
                 show_r2=True,
                 label_font_size=12,
                 label_font_color='black',
                 label_offset_x=5,
                 label_offset_y=5,
                 labeled_points_name='Labeled Points',
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
    @brief Plot a scatter plot of the specified data columns with optional point labeling.
    @param[in] data DataFrame containing the data to be plotted.
    @param[in] x_column (str) Column name for the x-axis.
    @param[in] y_column (str) Column name for the y-axis.
    @param[in] color_column (str) Optional column name for coloring points.
    @param[in] size_column (str) Optional column name for sizing points.
    @param[in] label_column (str) Column name containing labels for points (e.g., 'Name').
    @param[in] label_points (list/array/Series/DataFrame) Indices, boolean mask, list of indices, or filtered DataFrame for points to label.
    @param[in] output_dir (str) Directory to save the plot as a PNG file. If None, the plot is displayed interactively.
    @param[in] output_name (str) Custom file name for the saved plot.
    @param[in] marker_color (str) Color for the normal markers when not using color_column.
    @param[in] marker_size (int) Size of the normal markers when not using size_column.
    @param[in] marker_opacity (float) Opacity of the markers (0-1).
    @param[in] labeled_marker_color (str) Color for labeled points.
    @param[in] labeled_marker_size (int) Size for labeled points.
    @param[in] add_trendline (bool) Whether to add a linear trendline.
    @param[in] trendline_color (str) Color for the trendline.
    @param[in] trendline_width (int) Width of the trendline.
    @param[in] show_r2 (bool) Whether to display R² value with the trendline.
    @param[in] label_font_size (int) Font size for point labels.
    @param[in] label_font_color (str) Color for point labels.
    @param[in] label_offset_x (int) Horizontal offset for labels.
    @param[in] label_offset_y (int) Vertical offset for labels.
    @param[in] labeled_points_name (str) Name for labeled points in legend.
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
    
    # Convert label_points to boolean mask if needed
    label_mask = np.zeros(len(data), dtype=bool)
    if label_points is not None:
        # Handle DataFrame input (filtered dataframe)
        if isinstance(label_points, pd.DataFrame):
            # Find matching indices between the filtered dataframe and original data
            try:
                # Method 1: Use index alignment (most reliable for filtered DataFrames)
                common_indices = data.index.intersection(label_points.index)
                if len(common_indices) > 0:
                    # Convert to boolean mask based on original data's index
                    label_mask = data.index.isin(common_indices)
                else:
                    print("Warning: No matching indices found between original data and filtered DataFrame")
            except Exception as e:
                print(f"Warning: Could not match DataFrame indices: {e}")
                
        # Handle list/array inputs
        elif isinstance(label_points, (list, np.ndarray)) and len(label_points) > 0:
            if isinstance(label_points[0], bool):
                # Already a boolean mask
                label_mask = np.array(label_points)
            elif isinstance(label_points[0], (int, np.integer)):
                # List of indices
                label_mask[label_points] = True
                
        # Handle pandas Series
        elif hasattr(label_points, 'dtype'):
            if label_points.dtype == bool:
                # Pandas Series or numpy array of booleans
                label_mask = label_points.values if hasattr(label_points, 'values') else label_points
            else:
                # Try to interpret as indices
                try:
                    label_mask[list(label_points)] = True
                except Exception as e:
                    print(f"Warning: Could not interpret label_points as indices: {e}")
                    
        # Handle other iterable types
        else:
            try:
                label_mask[list(label_points)] = True
            except Exception as e:
                print(f"Warning: Could not interpret label_points parameter: {e}")
    
    # Separate normal points and labeled points
    normal_data = data[~label_mask]
    labeled_data = data[label_mask]
    
    # Prepare marker settings for normal points
    marker_dict = dict(
        opacity=marker_opacity,
        line=dict(width=1, color='black')
    )
    
    # Set marker color for normal points
    if color_column is not None:
        marker_dict['color'] = normal_data[color_column]
        marker_dict['colorscale'] = 'Viridis'
        marker_dict['colorbar'] = dict(
            title=color_column,
            titlefont=dict(family=legend_font_name, size=legend_font_size)
        )
    else:
        marker_dict['color'] = marker_color
    
    # Set marker size for normal points
    if size_column is not None:
        sizes = normal_data[size_column]
        if len(sizes) > 0 and sizes.min() != sizes.max():
            normalized_sizes = 5 + 20 * (sizes - sizes.min()) / (sizes.max() - sizes.min())
        else:
            normalized_sizes = [marker_size] * len(sizes)
        marker_dict['size'] = normalized_sizes
        marker_dict['sizemode'] = 'diameter'
        marker_dict['sizeref'] = 1.0
    else:
        marker_dict['size'] = marker_size

    # Add normal scatter trace
    if len(normal_data) > 0:
        fig.add_trace(
            go.Scatter(
                x=normal_data[x_column],
                y=normal_data[y_column],
                mode='markers',
                marker=marker_dict,
                name='Data Points',
                hovertemplate='<b>%{text}</b><br>' +
                             f'{x_column}: %{{x}}<br>' +
                             f'{y_column}: %{{y}}<extra></extra>' if label_column else None,
                text=normal_data[label_column] if label_column else None
            )
        )

    # Add labeled scatter trace
    if len(labeled_data) > 0:
        labeled_marker_dict = dict(
            color=labeled_marker_color,
            size=labeled_marker_size,
            opacity=marker_opacity,
            line=dict(width=2, color='black')
        )
        
        fig.add_trace(
            go.Scatter(
                x=labeled_data[x_column],
                y=labeled_data[y_column],
                mode='markers',
                marker=labeled_marker_dict,
                name=labeled_points_name,
                hovertemplate='<b>%{text}</b><br>' +
                             f'{x_column}: %{{x}}<br>' +
                             f'{y_column}: %{{y}}<br>' +
                             f'<b>{labeled_points_name.upper()}</b><extra></extra>' if label_column else None,
                text=labeled_data[label_column] if label_column else None
            )
        )
        
        # Add text labels for labeled points
        if label_column is not None:
            for idx, row in labeled_data.iterrows():
                fig.add_annotation(
                    x=row[x_column],
                    y=row[y_column],
                    text=str(row[label_column]),
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=1,
                    arrowcolor=label_font_color,
                    ax=label_offset_x,
                    ay=-label_offset_y,
                    font=dict(
                        size=label_font_size,
                        color=label_font_color,
                        family=legend_font_name
                    ),
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor=label_font_color,
                    borderwidth=1
                )

    # Add trendline if requested
    if add_trendline and len(data) > 1:
        valid_data = data[[x_column, y_column]].dropna()
        x = valid_data[x_column]
        y = valid_data[y_column]
        
        if len(x) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            r_squared = r_value**2
            
            x_line = np.array([min(x), max(x)])
            y_line = slope * x_line + intercept
            
            trendline_name = f'Trendline: y = {slope:.3f}x + {intercept:.3f}'
            if show_r2:
                trendline_name += f', R² = {r_squared:.3f}'
            
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