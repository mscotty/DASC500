import os
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO


def plot_box(
    data,
    value_column,
    group_column=None,
    output_dir=None,
    output_name=None,
    box_color="skyblue",
    box_colors=None,
    show_points=True,
    point_size=4,
    notched=False,
    title_name=None,
    title_font_size=28,
    title_font_name="Times New Roman",
    x_axis_name=None,
    x_axis_font_size=24,
    x_axis_font_name="Times New Roman",
    y_axis_name=None,
    y_axis_font_size=24,
    y_axis_font_name="Times New Roman",
    style="plotly",
    show_mean=True,
    show_outliers=True,
    box_width=0.5,
    orientation="vertical",
    fig_width=10,
    fig_height=6,
    dpi=100,
):
    """
    @brief Plot a box plot of the specified data, optionally grouped by a categorical variable.
    @param[in] data DataFrame containing the data to be plotted.
    @param[in] value_column (str) Column name for the values to be represented in the box plot.
    @param[in] group_column (str) Optional column name for grouping the data. If None, a single box plot is created.
    @param[in] output_dir (str) Directory to save the plot as a PNG file. If None, the plot is displayed interactively.
    @param[in] output_name (str) Custom file name for the saved plot.
    @param[in] box_color (str) Color for the box plot when not grouped.
    @param[in] box_colors (list) List of colors for the box plots when grouped.
    @param[in] show_points (bool) Whether to display individual data points.
    @param[in] point_size (int) Size of the individual data points.
    @param[in] notched (bool) Whether to display notched box plots.
    @param[in] title_name (str) Custom title for the chart. If None, a generic title is used.
    @param[in] title_font_size (int) Font size for the chart title.
    @param[in] title_font_name (str) Font name for the chart title.
    @param[in] x_axis_name (str) Custom label for the x-axis. If None, the group_column name is used.
    @param[in] x_axis_font_size (int) Font size for the x-axis labels.
    @param[in] x_axis_font_name (str) Font name for the x-axis labels.
    @param[in] y_axis_name (str) Custom label for the y-axis. If None, the value_column name is used.
    @param[in] y_axis_font_size (int) Font size for the y-axis labels.
    @param[in] y_axis_font_name (str) Font name for the y-axis labels.
    @param[in] style (str) Box plot style: 'plotly' for interactive, 'seaborn' or 'matplotlib' for standard.
    @param[in] show_mean (bool) Whether to display the mean marker.
    @param[in] show_outliers (bool) Whether to display outliers.
    @param[in] box_width (float) Width of the boxes (between 0 and 1).
    @param[in] orientation (str) Orientation of the box plot: 'vertical' or 'horizontal'.
    @param[in] fig_width (int) Width of the figure in inches (for matplotlib/seaborn).
    @param[in] fig_height (int) Height of the figure in inches (for matplotlib/seaborn).
    @param[in] dpi (int) DPI for the output figure (for matplotlib/seaborn).
    """
    # Determine the appropriate style to use
    if style.lower() in ["plotly", "interactive"]:
        return _plot_box_plotly(
            data,
            value_column,
            group_column=group_column,
            output_dir=output_dir,
            output_name=output_name,
            box_color=box_color,
            box_colors=box_colors,
            show_points=show_points,
            point_size=point_size,
            notched=notched,
            title_name=title_name,
            title_font_size=title_font_size,
            title_font_name=title_font_name,
            x_axis_name=x_axis_name,
            x_axis_font_size=x_axis_font_size,
            x_axis_font_name=x_axis_font_name,
            y_axis_name=y_axis_name,
            y_axis_font_size=y_axis_font_size,
            y_axis_font_name=y_axis_font_name,
            show_mean=show_mean,
            show_outliers=show_outliers,
            orientation=orientation,
        )
    elif style.lower() in ["seaborn", "sns"]:
        return _plot_box_seaborn(
            data,
            value_column,
            group_column=group_column,
            output_dir=output_dir,
            output_name=output_name,
            box_color=box_color,
            box_colors=box_colors,
            show_points=show_points,
            notched=notched,
            title_name=title_name,
            title_font_size=title_font_size,
            title_font_name=title_font_name,
            x_axis_name=x_axis_name,
            x_axis_font_size=x_axis_font_size,
            x_axis_font_name=x_axis_font_name,
            y_axis_name=y_axis_name,
            y_axis_font_size=y_axis_font_size,
            y_axis_font_name=y_axis_font_name,
            show_mean=show_mean,
            show_outliers=show_outliers,
            box_width=box_width,
            orientation=orientation,
            fig_width=fig_width,
            fig_height=fig_height,
            dpi=dpi,
        )
    elif style.lower() in ["matplotlib", "mpl"]:
        return _plot_box_matplotlib(
            data,
            value_column,
            group_column=group_column,
            output_dir=output_dir,
            output_name=output_name,
            box_color=box_color,
            box_colors=box_colors,
            show_points=show_points,
            notched=notched,
            title_name=title_name,
            title_font_size=title_font_size,
            title_font_name=title_font_name,
            x_axis_name=x_axis_name,
            x_axis_font_size=x_axis_font_size,
            x_axis_font_name=x_axis_font_name,
            y_axis_name=y_axis_name,
            y_axis_font_size=y_axis_font_size,
            y_axis_font_name=y_axis_font_name,
            show_mean=show_mean,
            show_outliers=show_outliers,
            box_width=box_width,
            orientation=orientation,
            fig_width=fig_width,
            fig_height=fig_height,
            dpi=dpi,
        )
    else:
        raise ValueError(
            f"Unknown style: {style}. Choose from 'plotly', 'seaborn', or 'matplotlib'."
        )


def _plot_box_plotly(
    data,
    value_column,
    group_column=None,
    output_dir=None,
    output_name=None,
    box_color="skyblue",
    box_colors=None,
    show_points=True,
    point_size=4,
    notched=False,
    title_name=None,
    title_font_size=28,
    title_font_name="Times New Roman",
    x_axis_name=None,
    x_axis_font_size=24,
    x_axis_font_name="Times New Roman",
    y_axis_name=None,
    y_axis_font_size=24,
    y_axis_font_name="Times New Roman",
    show_mean=True,
    show_outliers=True,
    orientation="horizontal",
):
    """Helper function to create a Plotly box plot."""
    fig = go.Figure()

    # Point display mode - modified to only show outliers
    point_mode = "outliers" if show_outliers else False

    # Determine orientation
    is_vertical = orientation.lower() in ["vertical", "v"]

    # Single box plot (no grouping)
    if group_column is None:
        if is_vertical:
            fig.add_trace(
                go.Box(
                    y=data[value_column],
                    name=value_column,
                    boxmean=show_mean,
                    notched=notched,
                    marker=dict(color=box_color, size=point_size),
                    line=dict(color="black", width=1),
                    fillcolor=box_color,
                    boxpoints=point_mode,
                )
            )
        else:  # horizontal
            fig.add_trace(
                go.Box(
                    x=data[value_column],
                    name=value_column,
                    boxmean=show_mean,
                    notched=notched,
                    marker=dict(color=box_color, size=point_size),
                    line=dict(color="black", width=1),
                    fillcolor=box_color,
                    boxpoints=point_mode,
                    orientation='h',  # Explicitly set horizontal orientation
                )
            )
    # Grouped box plot
    else:
        # Get unique groups
        groups = data[group_column].unique()

        # Default colors if not provided
        if box_colors is None:
            import plotly.express as px

            box_colors = px.colors.qualitative.Plotly[: len(groups)]
            # Repeat colors if we have more groups than colors
            if len(groups) > len(box_colors):
                box_colors = box_colors * (len(groups) // len(box_colors) + 1)

        # Add a box plot for each group
        for i, group in enumerate(groups):
            group_data = data[data[group_column] == group]
            color = box_colors[i % len(box_colors)]

            if is_vertical:
                fig.add_trace(
                    go.Box(
                        y=group_data[value_column],
                        name=str(group),
                        boxmean=show_mean,
                        notched=notched,
                        marker=dict(color=color, size=point_size),
                        line=dict(color="black", width=1),
                        fillcolor=color,
                        boxpoints=point_mode,
                    )
                )
            else:  # horizontal
                fig.add_trace(
                    go.Box(
                        x=group_data[value_column],
                        name=str(group),
                        boxmean=show_mean,
                        notched=notched,
                        marker=dict(color=color, size=point_size),
                        line=dict(color="black", width=1),
                        fillcolor=color,
                        boxpoints=point_mode,
                        orientation='h',  # Explicitly set horizontal orientation
                    )
                )

    # Update layout
    if is_vertical:
        x_title_text = (
            x_axis_name if x_axis_name else (group_column if group_column else "")
        )
        y_title_text = y_axis_name if y_axis_name else value_column
    else:
        x_title_text = y_axis_name if y_axis_name else value_column
        y_title_text = (
            x_axis_name if x_axis_name else (group_column if group_column else "")
        )

    # Adjust the figure dimensions based on orientation
    if is_vertical:
        width = 800
        height = 600
    else:
        width = 1000  # Wider for horizontal orientation
        height = 400 if group_column is None else 100 + 80 * len(data[group_column].unique())  # Dynamic height based on groups

    fig.update_layout(
        title=dict(
            text=(
                title_name
                if title_name
                else f"Box Plot of {value_column}"
                + (f" by {group_column}" if group_column else "")
            ),
            font=dict(family=title_font_name, size=title_font_size),
            x=0.5,
        ),
        xaxis=dict(
            title=dict(
                text=x_title_text,
                font=dict(family=x_axis_font_name, size=x_axis_font_size),
            ),
            tickfont=dict(family=x_axis_font_name, size=x_axis_font_size - 4),
            gridcolor="lightgray",
        ),
        yaxis=dict(
            title=dict(
                text=y_title_text,
                font=dict(family=y_axis_font_name, size=y_axis_font_size),
            ),
            tickfont=dict(family=y_axis_font_name, size=y_axis_font_size - 4),
            gridcolor="lightgray",
        ),
        font=dict(family="Times New Roman", size=14),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=100 if group_column and not is_vertical else 50, r=50, t=80, b=50),
        width=width,
        height=height,
    )

    # Add grid lines
    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor="lightgray")
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor="lightgray")

    # Save or display the plot
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        file_path = os.path.join(
            output_dir,
            (
                output_name
                if output_name
                else f"box_plot_{value_column}"
                + (f"_by_{group_column}" if group_column else "")
                + ".png"
            ),
        )
        fig.write_image(file_path, format="png", width=width, height=height)
    else:
        fig.show()

    return fig


def _plot_box_seaborn(
    data,
    value_column,
    group_column=None,
    output_dir=None,
    output_name=None,
    box_color="Paired",
    box_colors=None,
    show_points=True,
    point_size=4,
    notched=False,
    title_name=None,
    title_font_size=28,
    title_font_name="Times New Roman",
    x_axis_name=None,
    x_axis_font_size=24,
    x_axis_font_name="Times New Roman",
    y_axis_name=None,
    y_axis_font_size=24,
    y_axis_font_name="Times New Roman",
    show_mean=True,
    show_outliers=True,
    box_width=0.5,
    orientation="vertical",
    fig_width=10,
    fig_height=6,
    dpi=100,
):
    """Helper function to create a Seaborn box plot."""
    # Set up the matplotlib figure
    plt.figure(figsize=(fig_width, fig_height), dpi=dpi)

    # Set the style
    sns.set_style("whitegrid")

    # Determine orientation and axis variables
    is_vertical = orientation.lower() in ["vertical", "v"]

    if is_vertical:
        x = group_column if group_column else None
        y = value_column
    else:
        x = value_column
        y = group_column if group_column else None

    # Create the box plot
    if show_points:
        # Use stripplot for showing all data points
        ax = sns.boxplot(
            x=x,
            y=y,
            data=data,
            notch=notched,
            showmeans=show_mean,
            showfliers=show_outliers,
            width=box_width,
            palette=box_colors if group_column else box_color,
            orient="v" if is_vertical else "h",
        )
        sns.stripplot(
            x=x,
            y=y,
            data=data,
            size=point_size / 2,
            color="black",
            alpha=0.5,
            orient="v" if is_vertical else "h",
        )
    else:
        ax = sns.boxplot(
            x=x,
            y=y,
            data=data,
            notch=notched,
            showmeans=show_mean,
            showfliers=show_outliers,
            width=box_width,
            palette=box_colors if group_column else box_color,
            orient="v" if is_vertical else "h",
        )

    # Set title and labels
    plt.title(
        (
            title_name
            if title_name
            else f"Box Plot of {value_column}"
            + (f" by {group_column}" if group_column else "")
        ),
        fontsize=title_font_size,
        fontname=title_font_name,
    )

    if is_vertical:
        plt.xlabel(
            x_axis_name if x_axis_name else (group_column if group_column else ""),
            fontsize=x_axis_font_size,
            fontname=x_axis_font_name,
        )
        plt.ylabel(
            y_axis_name if y_axis_name else value_column,
            fontsize=y_axis_font_size,
            fontname=y_axis_font_name,
        )
    else:
        plt.xlabel(
            x_axis_name if x_axis_name else value_column,
            fontsize=x_axis_font_size,
            fontname=x_axis_font_name,
        )
        plt.ylabel(
            y_axis_name if y_axis_name else (group_column if group_column else ""),
            fontsize=y_axis_font_size,
            fontname=y_axis_font_name,
        )

    # Set tick label font
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname(x_axis_font_name)
        label.set_fontsize(x_axis_font_size - 4)

    # Adjust layout
    plt.tight_layout()

    # Save or display the plot
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        file_path = os.path.join(
            output_dir,
            (
                output_name
                if output_name
                else f"box_plot_{value_column}"
                + (f"_by_{group_column}" if group_column else "")
                + ".png"
            ),
        )
        plt.savefig(file_path, dpi=dpi, bbox_inches="tight")
    else:
        plt.show()

    return plt.gcf()


def _plot_box_matplotlib(
    data,
    value_column,
    group_column=None,
    output_dir=None,
    output_name=None,
    box_color="Paired",
    box_colors=None,
    show_points=True,
    point_size=4,
    notched=False,
    title_name=None,
    title_font_size=28,
    title_font_name="Times New Roman",
    x_axis_name=None,
    x_axis_font_size=24,
    x_axis_font_name="Times New Roman",
    y_axis_name=None,
    y_axis_font_size=24,
    y_axis_font_name="Times New Roman",
    show_mean=True,
    show_outliers=True,
    box_width=0.5,
    orientation="vertical",
    fig_width=10,
    fig_height=6,
    dpi=100,
):
    """Helper function to create a Matplotlib box plot."""
    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)

    # Determine orientation
    is_vertical = orientation.lower() in ["vertical", "v"]

    # Prepare data for plotting
    if group_column is None:
        # Single boxplot
        plot_data = [data[value_column].dropna()]
        labels = [value_column]
    else:
        # Grouped boxplot
        groups = data[group_column].unique()
        plot_data = [
            data[data[group_column] == group][value_column].dropna() for group in groups
        ]
        labels = [str(group) for group in groups]

    # Set colors
    if group_column is not None and box_colors is not None:
        colors = box_colors[: len(groups)]
    else:
        colors = [box_color] * len(plot_data)

    # Create the box plot
    boxplot = ax.boxplot(
        plot_data,
        vert=is_vertical,
        notch=notched,
        patch_artist=True,
        showmeans=show_mean,
        showfliers=show_outliers,
        widths=box_width,
        labels=labels,
    )

    # Customize box colors
    for box, color in zip(boxplot["boxes"], colors):
        box.set(facecolor=color, edgecolor="black", linewidth=1)

    # Customize other box plot elements
    for element in ["whiskers", "caps", "medians"]:
        for item in boxplot[element]:
            item.set(color="black", linewidth=1)

    if show_mean:
        for mean in boxplot["means"]:
            mean.set(
                marker="o",
                markeredgecolor="black",
                markerfacecolor="white",
                markersize=8,
            )

    # Add scatter points if requested
    if show_points:
        for i, (data_group, color) in enumerate(zip(plot_data, colors)):
            # Position for the points
            x_pos = i + 1  # Box positions are 1-based

            # Add jitter to avoid overlap
            jitter = np.random.normal(0, 0.05, size=len(data_group))

            if is_vertical:
                ax.scatter(
                    [x_pos + j for j in jitter],
                    data_group,
                    alpha=0.5,
                    color="black",
                    s=point_size * 5,
                    zorder=1,
                )
            else:
                ax.scatter(
                    data_group,
                    [x_pos + j for j in jitter],
                    alpha=0.5,
                    color="black",
                    s=point_size * 5,
                    zorder=1,
                )

    # Set title and labels
    ax.set_title(
        (
            title_name
            if title_name
            else f"Box Plot of {value_column}"
            + (f" by {group_column}" if group_column else "")
        ),
        fontsize=title_font_size,
        fontname=title_font_name,
    )

    if is_vertical:
        ax.set_xlabel(
            x_axis_name if x_axis_name else (group_column if group_column else ""),
            fontsize=x_axis_font_size,
            fontname=x_axis_font_name,
        )
        ax.set_ylabel(
            y_axis_name if y_axis_name else value_column,
            fontsize=y_axis_font_size,
            fontname=y_axis_font_name,
        )
    else:
        ax.set_xlabel(
            x_axis_name if x_axis_name else value_column,
            fontsize=x_axis_font_size,
            fontname=x_axis_font_name,
        )
        ax.set_ylabel(
            y_axis_name if y_axis_name else (group_column if group_column else ""),
            fontsize=y_axis_font_size,
            fontname=y_axis_font_name,
        )

    # Set tick label font
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname(x_axis_font_name)
        label.set_fontsize(x_axis_font_size - 4)

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.7)

    # Adjust layout
    plt.tight_layout()

    # Save or display the plot
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        file_path = os.path.join(
            output_dir,
            (
                output_name
                if output_name
                else f"box_plot_{value_column}"
                + (f"_by_{group_column}" if group_column else "")
                + ".png"
            ),
        )
        plt.savefig(file_path, dpi=dpi, bbox_inches="tight")
    else:
        plt.show()

    return fig
