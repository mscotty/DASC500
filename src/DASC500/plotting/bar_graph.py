import os
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd

def plot_bar_chart(data,
                   x_column,
                   y_column,
                   output_dir=None,
                   output_name=None,
                   orientation='v',
                   color='skyblue',
                   bar_width=None,
                   title_name=None,
                   title_font_size=28,
                   title_font_name='Times New Roman',
                   x_axis_name=None,
                   x_axis_font_size=24,
                   x_axis_font_name='Times New Roman',
                   y_axis_name=None,
                   y_axis_font_size=24,
                   y_axis_font_name='Times New Roman',
                   show_values=False,
                   value_font_size=12,
                   value_precision=1,
                   sort_values=True,
                   sort_ascending=False):
    """
    @brief Plot a bar chart using the specified data columns.
    @param[in] data DataFrame containing the data to be plotted.
    @param[in] x_column (str) Column name for the x-axis (categories).
    @param[in] y_column (str) Column name for the y-axis (values).
    @param[in] output_dir (str) Directory to save the plot as a PNG file. If None, the plot is displayed interactively.
    @param[in] output_name (str) Custom file name for the saved plot.
    @param[in] orientation (str) Bar orientation: 'v' for vertical, 'h' for horizontal.
    @param[in] color (str) Color for the bars.
    @param[in] bar_width (float) Width of the bars (between 0 and 1).
    @param[in] title_name (str) Custom title for the chart. If None, a generic title is used.
    @param[in] title_font_size (int) Font size for the chart title.
    @param[in] title_font_name (str) Font name for the chart title.
    @param[in] x_axis_name (str) Custom label for the x-axis. If None, the x_column name is used.
    @param[in] x_axis_font_size (int) Font size for the x-axis labels.
    @param[in] x_axis_font_name (str) Font name for the x-axis labels.
    @param[in] y_axis_name (str) Custom label for the y-axis. If None, the y_column name is used.
    @param[in] y_axis_font_size (int) Font size for the y-axis labels.
    @param[in] y_axis_font_name (str) Font name for the y-axis labels.
    @param[in] show_values (bool) Whether to display values on top of bars.
    @param[in] value_font_size (int) Font size for the values displayed on bars.
    @param[in] value_precision (int) Number of decimal places for displayed values.
    @param[in] sort_values (bool) Whether to sort the data by y-values. Default is True.
    @param[in] sort_ascending (bool) Whether to sort in ascending order. Default is False (descending).
    """
    # Create a copy of the data to avoid modifying the original
    plot_data = data.copy()
    
    # Sort the data by y-values if requested
    if sort_values:
        sort_column = y_column if orientation == 'v' else x_column
        plot_data = plot_data.sort_values(by=sort_column, ascending=sort_ascending).reset_index(drop=True)
    
    fig = go.Figure()

    # Set up bar trace based on orientation
    if orientation == 'v':
        x_data = plot_data[x_column]
        y_data = plot_data[y_column]
        text_position = 'outside'
    else:  # horizontal
        x_data = plot_data[y_column]
        y_data = plot_data[x_column]
        text_position = 'auto'

    # Format text for values if needed
    text = None
    if show_values:
        if orientation == 'v':
            text = [f'{val:.{value_precision}f}' for val in plot_data[y_column]]
        else:
            text = [f'{val:.{value_precision}f}' for val in plot_data[x_column]]

    # Add bar trace
    fig.add_trace(
        go.Bar(
            x=x_data,
            y=y_data,
            orientation=orientation,
            marker=dict(
                color=color,
                line=dict(color='black', width=1)
            ),
            width=bar_width,
            text=text,
            textposition=text_position,
            textfont=dict(
                size=value_font_size,
                family='Times New Roman'
            )
        )
    )

    # Determine axis titles based on orientation
    if orientation == 'v':
        x_title = x_axis_name if x_axis_name else x_column
        y_title = y_axis_name if y_axis_name else y_column
    else:
        x_title = y_axis_name if y_axis_name else y_column
        y_title = x_axis_name if x_axis_name else x_column

    # Update layout
    fig.update_layout(
        title=dict(
            text=title_name if title_name else f"Bar Chart of {y_column} by {x_column}",
            font=dict(family=title_font_name, size=title_font_size),
            x=0.5
        ),
        xaxis=dict(
            title=dict(text=x_title, font=dict(family=x_axis_font_name, size=x_axis_font_size)),
            tickfont=dict(family=x_axis_font_name, size=x_axis_font_size-4),
            gridcolor="lightgray"
        ),
        yaxis=dict(
            title=dict(text=y_title, font=dict(family=y_axis_font_name, size=y_axis_font_size)),
            tickfont=dict(family=y_axis_font_name, size=y_axis_font_size-4),
            gridcolor="lightgray"
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
        file_path = os.path.join(output_dir, output_name if output_name else f"bar_chart.png")
        fig.write_image(file_path, format="png", width=800, height=600)
    else:
        fig.show()
