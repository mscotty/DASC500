import os
import logging
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform


def compute_correlation_matrix(df, columns=None, method="pearson"):
    """
    Computes the correlation matrix for numerical columns.

    Parameters:
    -----------
    df : pandas DataFrame
        The DataFrame to analyze
    columns : list of str, optional
        Specific columns to analyze (default: all numerical columns)
    method : str, default='pearson'
        Correlation method ('pearson', 'spearman', or 'kendall')

    Returns:
    --------
    pandas DataFrame
        Correlation matrix
    """
    # Select numerical columns if not specified
    if columns is None:
        num_df = df.select_dtypes(include=np.number)
    else:
        # Filter to only include numerical columns that exist
        valid_cols = [
            col
            for col in columns
            if col in df.columns and np.issubdtype(df[col].dtype, np.number)
        ]
        num_df = df[valid_cols]

    if num_df.empty:
        return pd.DataFrame()

    # Compute correlation matrix
    corr_matrix = num_df.corr(method=method)

    return corr_matrix


def create_correlation_heatmap(
    corr_matrix,
    title="Correlation Matrix",
    cmap="coolwarm",
    annot=True,
    fmt=".2f",
    output_dir=None,
    filename="correlation_heatmap.png",
    show_plot=True,
    mask_upper=False,
    figsize=None,
):
    """
    Creates a heatmap visualization of a correlation matrix.

    Parameters:
    -----------
    corr_matrix : pandas DataFrame
        Correlation matrix to visualize
    title : str, default='Correlation Matrix'
        Title for the plot
    cmap : str or matplotlib colormap, default='coolwarm'
        Colormap for the heatmap
    annot : bool, default=True
        Whether to annotate cells with correlation values
    fmt : str, default='.2f'
        Format string for annotations
    output_dir : str, optional
        Directory to save the plot
    filename : str, default='correlation_heatmap.png'
        Filename for the saved plot
    show_plot : bool, default=True
        Whether to display the plot
    mask_upper : bool, default=False
        Whether to mask the upper triangle of the matrix
    figsize : tuple, optional
        Figure size (width, height) in inches

    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    if corr_matrix.empty:
        return None

    # Determine figure size based on matrix dimensions
    if figsize is None:
        size = max(8, corr_matrix.shape[0] * 0.5)
        figsize = (size, size)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create mask for upper triangle if requested
    mask = None
    if mask_upper:
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Create heatmap
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap=cmap,
        annot=annot,
        fmt=fmt,
        square=True,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )

    # Set title and adjust layout
    ax.set_title(title)
    fig.tight_layout()

    # Save plot if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, filename)
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")

    # Show or close plot
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return fig


def create_clustered_correlation_heatmap(
    corr_matrix,
    title="Clustered Correlation Matrix",
    method="complete",
    metric="euclidean",
    output_dir=None,
    filename="clustered_correlation_heatmap.png",
    show_plot=True,
    figsize=None,
):
    """
    Creates a clustered heatmap visualization of a correlation matrix.

    Parameters:
    -----------
    corr_matrix : pandas DataFrame
        Correlation matrix to visualize
    title : str, default='Clustered Correlation Matrix'
        Title for the plot
    method : str, default='complete'
        Linkage method for hierarchical clustering
    metric : str, default='euclidean'
        Distance metric for clustering
    output_dir : str, optional
        Directory to save the plot
    filename : str, default='clustered_correlation_heatmap.png'
        Filename for the saved plot
    show_plot : bool, default=True
        Whether to display the plot
    figsize : tuple, optional
        Figure size (width, height) in inches

    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    if corr_matrix.empty:
        return None

    # Determine figure size based on matrix dimensions
    if figsize is None:
        size = max(10, corr_matrix.shape[0] * 0.6)
        figsize = (size, size)

    # Compute distance matrix (1 - |correlation|)
    distance_matrix = 1 - np.abs(corr_matrix)

    # Perform hierarchical clustering
    Z = hierarchy.linkage(squareform(distance_matrix), method=method)

    # Create figure with 2 subplots (dendrogram and heatmap)
    fig = plt.figure(figsize=figsize)

    # Add gridspec for better control
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[0.2, 0.8],
        height_ratios=[0.2, 0.8],
        wspace=0.01,
        hspace=0.01,
    )

    # Top dendrogram
    ax1 = fig.add_subplot(gs[0, 1])
    hierarchy.dendrogram(Z, ax=ax1, orientation="top", no_labels=True)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["bottom"].set_visible(False)
    ax1.spines["left"].set_visible(False)

    # Left dendrogram
    ax2 = fig.add_subplot(gs[1, 0])
    hierarchy.dendrogram(Z, ax=ax2, orientation="left", no_labels=True)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["bottom"].set_visible(False)
    ax2.spines["left"].set_visible(False)

    # Heatmap
    ax3 = fig.add_subplot(gs[1, 1])

    # Reorder correlation matrix based on clustering
    idx = hierarchy.leaves_list(Z)
    reordered_corr = corr_matrix.iloc[idx, idx]

    # Create heatmap
    sns.heatmap(
        reordered_corr,
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        ax=ax3,
        cbar_kws={"shrink": 0.8},
    )

    # Set title
    fig.suptitle(title, fontsize=16, y=0.98)

    # Save plot if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, filename)
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")

    # Show or close plot
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return fig


def identify_highly_correlated_pairs(corr_matrix, threshold=0.7):
    """
    Identifies pairs of variables with correlation above a threshold.

    Parameters:
    -----------
    corr_matrix : pandas DataFrame
        Correlation matrix
    threshold : float, default=0.7
        Correlation threshold

    Returns:
    --------
    pandas DataFrame
        DataFrame with highly correlated pairs
    """
    if corr_matrix.empty:
        return pd.DataFrame()

    # Get variable pairs with correlation above threshold
    pairs = []

    # Iterate through lower triangle of correlation matrix
    for i, j in itertools.combinations(range(len(corr_matrix)), 2):
        if abs(corr_matrix.iloc[i, j]) >= threshold:
            var1 = corr_matrix.index[i]
            var2 = corr_matrix.columns[j]
            corr_val = corr_matrix.iloc[i, j]
            pairs.append(
                {
                    "Variable1": var1,
                    "Variable2": var2,
                    "Correlation": corr_val,
                    "AbsCorrelation": abs(corr_val),
                }
            )

    # Create DataFrame and sort by absolute correlation
    if pairs:
        pairs_df = pd.DataFrame(pairs)
        pairs_df = pairs_df.sort_values("AbsCorrelation", ascending=False)
        return pairs_df
    else:
        return pd.DataFrame(
            columns=["Variable1", "Variable2", "Correlation", "AbsCorrelation"]
        )


def compute_vif_factors(df, columns=None):
    """
    Computes Variance Inflation Factor (VIF) for multicollinearity assessment.

    Parameters:
    -----------
    df : pandas DataFrame
        The DataFrame to analyze
    columns : list of str, optional
        Specific columns to analyze (default: all numerical columns)

    Returns:
    --------
    pandas DataFrame
        DataFrame with VIF values for each variable
    """
    # Select numerical columns if not specified
    if columns is None:
        num_df = df.select_dtypes(include=np.number)
    else:
        # Filter to only include numerical columns that exist
        valid_cols = [
            col
            for col in columns
            if col in df.columns and np.issubdtype(df[col].dtype, np.number)
        ]
        num_df = df[valid_cols]

    if num_df.empty or num_df.shape[1] < 2:
        return pd.DataFrame()

    # Drop rows with NaN values
    X = num_df.dropna()

    if X.shape[0] < 2:
        return pd.DataFrame()

    # Compute VIF for each variable
    vif_data = []

    for i, col in enumerate(X.columns):
        try:
            # Create a DataFrame with all variables except the current one
            X_others = X.drop(col, axis=1)

            # Add a constant term
            X_others = pd.DataFrame(
                StandardScaler().fit_transform(X_others), columns=X_others.columns
            )

            # Add the current variable
            y = X[col]

            # Calculate R-squared from regression
            r_squared = np.corrcoef(X_others, y, rowvar=False)[-1, :-1]
            r_squared = np.dot(
                r_squared, np.linalg.inv(np.corrcoef(X_others, rowvar=False))
            ).dot(r_squared)

            # Calculate VIF
            vif = 1 / (1 - r_squared)

            vif_data.append({"Variable": col, "VIF": vif})
        except:
            # If calculation fails, add NaN
            vif_data.append({"Variable": col, "VIF": np.nan})

    # Create DataFrame and sort by VIF
    vif_df = pd.DataFrame(vif_data)
    vif_df = vif_df.sort_values("VIF", ascending=False)

    return vif_df


def create_correlation_network(
    corr_matrix,
    threshold=0.5,
    output_dir=None,
    filename="correlation_network.png",
    show_plot=True,
):
    """
    Creates a network visualization of correlations above a threshold.

    Parameters:
    -----------
    corr_matrix : pandas DataFrame
        Correlation matrix
    threshold : float, default=0.5
        Correlation threshold for including edges
    output_dir : str, optional
        Directory to save the plot
    filename : str, default='correlation_network.png'
        Filename for the saved plot
    show_plot : bool, default=True
        Whether to display the plot

    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    try:
        import networkx as nx

        if corr_matrix.empty:
            return None

        # Create graph
        G = nx.Graph()

        # Add nodes
        for var in corr_matrix.columns:
            G.add_node(var)

        # Add edges for correlations above threshold
        for i, var1 in enumerate(corr_matrix.columns):
            for j, var2 in enumerate(corr_matrix.columns):
                if i < j:  # Upper triangle only
                    corr_val = corr_matrix.loc[var1, var2]
                    if abs(corr_val) >= threshold:
                        G.add_edge(
                            var1,
                            var2,
                            weight=abs(corr_val),
                            color="red" if corr_val < 0 else "blue",
                        )

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))

        # Set node positions using spring layout
        pos = nx.spring_layout(G, k=0.3, iterations=50)

        # Get edge colors and weights
        edge_colors = [G[u][v]["color"] for u, v in G.edges()]
        edge_weights = [G[u][v]["weight"] * 3 for u, v in G.edges()]

        # Draw network
        nx.draw_networkx_nodes(
            G, pos, node_size=700, node_color="lightblue", alpha=0.8, ax=ax
        )
        nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
        nx.draw_networkx_edges(
            G, pos, width=edge_weights, alpha=0.7, edge_color=edge_colors, ax=ax
        )

        # Add legend
        pos_patch = plt.Line2D(
            [0], [0], color="blue", linewidth=3, label="Positive Correlation"
        )
        neg_patch = plt.Line2D(
            [0], [0], color="red", linewidth=3, label="Negative Correlation"
        )
        plt.legend(handles=[pos_patch, neg_patch], loc="upper right")

        # Set title and remove axes
        plt.title(f"Correlation Network (threshold = {threshold})")
        plt.axis("off")

        # Save plot if output directory is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plot_path = os.path.join(output_dir, filename)
            fig.savefig(plot_path, dpi=300, bbox_inches="tight")

        # Show or close plot
        if show_plot:
            plt.show()
        else:
            plt.close(fig)

        return fig

    except ImportError:
        print("NetworkX library is required for correlation network visualization.")
        return None


def create_pairplot(
    df,
    columns=None,
    hue=None,
    diag_kind="kde",
    output_dir=None,
    filename="pairplot.png",
    show_plot=True,
):
    """
    Creates a pairplot for visualizing pairwise relationships.

    Parameters:
    -----------
    df : pandas DataFrame
        The DataFrame to analyze
    columns : list of str, optional
        Specific columns to include (default: all numerical columns)
    hue : str, optional
        Column name for color-coding points
    diag_kind : str, default='kde'
        Kind of plot for diagonal elements ('hist', 'kde')
    output_dir : str, optional
        Directory to save the plot
    filename : str, default='pairplot.png'
        Filename for the saved plot
    show_plot : bool, default=True
        Whether to display the plot

    Returns:
    --------
    seaborn.axisgrid.PairGrid
        The created pairplot
    """
    # Select numerical columns if not specified
    if columns is None:
        num_df = df.select_dtypes(include=np.number)
        columns = num_df.columns.tolist()
    else:
        # Filter to only include columns that exist
        columns = [col for col in columns if col in df.columns]

    if not columns:
        return None

    # Create pairplot
    g = sns.pairplot(
        df,
        vars=columns,
        hue=hue,
        diag_kind=diag_kind,
        plot_kws={"alpha": 0.6},
        diag_kws={"alpha": 0.6},
    )

    # Set title
    g.fig.suptitle("Pairwise Relationships", y=1.02, fontsize=16)

    # Save plot if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, filename)
        g.savefig(plot_path, dpi=300, bbox_inches="tight")

    # Show or close plot
    if show_plot:
        plt.show()
    else:
        plt.close(g.fig)

    return g


def create_correlation_clustermap(
    corr_matrix,
    method="complete",
    metric="euclidean",
    cmap="coolwarm",
    output_dir=None,
    filename="correlation_clustermap.png",
    show_plot=True,
):
    """
    Creates a clustermap of the correlation matrix using seaborn.

    Parameters:
    -----------
    corr_matrix : pandas DataFrame
        Correlation matrix to visualize
    method : str, default='complete'
        Linkage method for hierarchical clustering
    metric : str, default='euclidean'
        Distance metric for clustering
    cmap : str or matplotlib colormap, default='coolwarm'
        Colormap for the heatmap
    output_dir : str, optional
        Directory to save the plot
    filename : str, default='correlation_clustermap.png'
        Filename for the saved plot
    show_plot : bool, default=True
        Whether to display the plot

    Returns:
    --------
    seaborn.matrix.ClusterGrid
        The created clustermap
    """
    if corr_matrix.empty:
        return None

    # Create clustermap
    g = sns.clustermap(
        corr_matrix,
        method=method,
        metric=metric,
        cmap=cmap,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        figsize=(12, 12),
    )

    # Set title
    plt.suptitle("Clustered Correlation Matrix", y=0.98, fontsize=16)

    # Save plot if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, filename)
        g.savefig(plot_path, dpi=300, bbox_inches="tight")

    # Show or close plot
    if show_plot:
        plt.show()
    else:
        plt.close(g.fig)

    return g


def analyze_correlation(
    df,
    columns=None,
    method="pearson",
    threshold=0.7,
    output_dir=None,
    create_heatmap=True,
    create_clustermap=True,
    create_network=True,
    create_pairplot_bool=True,
    check_multicollinearity=True,
    show_plots=True,
):
    """
    Comprehensive correlation analysis for numerical columns.

    Parameters:
    -----------
    df : pandas DataFrame
        The DataFrame to analyze
    columns : list of str, optional
        Specific columns to analyze (default: all numerical columns)
    method : str, default='pearson'
        Correlation method ('pearson', 'spearman', or 'kendall')
    threshold : float, default=0.7
        Threshold for identifying high correlations
    output_dir : str, optional
        Directory to save output files and plots
    create_heatmap : bool, default=True
        Whether to create a correlation heatmap
    create_clustermap : bool, default=True
        Whether to create a clustered correlation heatmap
    create_network : bool, default=True
        Whether to create a correlation network visualization
    create_pairplot : bool, default=True
        Whether to create a pairplot
    check_multicollinearity : bool, default=True
        Whether to compute VIF factors for multicollinearity assessment
    show_plots : bool, default=True
        Whether to display plots

    Returns:
    --------
    dict
        Dictionary containing analysis results
    """
    # Set up logging
    logger = logging.getLogger("correlation_analysis")
    logger.setLevel(logging.INFO)

    # Create console handler if not already present
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(message)s")
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Add file handler if output_dir is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            file_handler = logging.FileHandler(
                os.path.join(output_dir, "correlation_analysis.log")
            )
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    logger.info("\n=== Correlation Analysis ===")

    # Select numerical columns if not specified
    if columns is None:
        num_df = df.select_dtypes(include=np.number)
        columns = num_df.columns.tolist()
    else:
        # Filter to only include numerical columns that exist
        columns = [
            col
            for col in columns
            if col in df.columns and np.issubdtype(df[col].dtype, np.number)
        ]

    if len(columns) < 2:
        logger.info("Correlation analysis requires at least 2 numerical columns.")
        return {}

    logger.info(
        f"Analyzing correlations for {len(columns)} columns using {method} method"
    )

    # Initialize results dictionary
    results = {
        "correlation_matrix": None,
        "high_correlations": None,
        "vif_factors": None,
        "plots": {},
    }

    # Compute correlation matrix
    corr_matrix = compute_correlation_matrix(df[columns], method=method)
    results["correlation_matrix"] = corr_matrix

    # Log correlation matrix
    logger.info(f"\nCorrelation Matrix ({method}):")
    logger.info(f"{corr_matrix}")

    # Save correlation matrix to CSV if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        corr_path = os.path.join(output_dir, f"correlation_matrix_{method}.csv")
        corr_matrix.to_csv(corr_path)
        logger.info(f"Correlation matrix saved to: {corr_path}")

    # Identify highly correlated pairs
    high_corr = identify_highly_correlated_pairs(corr_matrix, threshold)
    results["high_correlations"] = high_corr

    # Log high correlations
    if not high_corr.empty:
        logger.info(f"\nHighly correlated pairs (|r| >= {threshold}):")
        for _, row in high_corr.iterrows():
            logger.info(
                f"  {row['Variable1']} - {row['Variable2']}: {row['Correlation']:.4f}"
            )

        # Save high correlations to CSV if output directory is provided
        if output_dir:
            high_corr_path = os.path.join(output_dir, "high_correlations.csv")
            high_corr.to_csv(high_corr_path, index=False)
            logger.info(f"High correlations saved to: {high_corr_path}")
    else:
        logger.info(f"\nNo variable pairs with correlation above {threshold} found.")

    # Check multicollinearity if requested
    if check_multicollinearity:
        vif_df = compute_vif_factors(df[columns])
        results["vif_factors"] = vif_df

        if not vif_df.empty:
            logger.info("\nVariance Inflation Factors (VIF):")
            for _, row in vif_df.iterrows():
                vif_value = row["VIF"]
                var_name = row["Variable"]

                if vif_value > 10:
                    severity = "severe"
                elif vif_value > 5:
                    severity = "moderate"
                else:
                    severity = "low"

                logger.info(
                    f"  {var_name}: {vif_value:.2f} ({severity} multicollinearity)"
                )

            # Save VIF factors to CSV if output directory is provided
            if output_dir:
                vif_path = os.path.join(output_dir, "vif_factors.csv")
                vif_df.to_csv(vif_path, index=False)
                logger.info(f"VIF factors saved to: {vif_path}")

    # Create plots directory if needed
    plots_dir = os.path.join(output_dir, "correlation_plots") if output_dir else None

    # Create correlation heatmap if requested
    if create_heatmap:
        logger.info("\nCreating correlation heatmap...")
        heatmap_fig = create_correlation_heatmap(
            corr_matrix,
            title=f"{method.capitalize()} Correlation Matrix",
            output_dir=plots_dir,
            show_plot=show_plots,
        )
        results["plots"]["heatmap"] = heatmap_fig

    # Create clustered correlation heatmap if requested
    if create_clustermap:
        logger.info("\nCreating clustered correlation heatmap...")
        clustermap_fig = create_correlation_clustermap(
            corr_matrix, output_dir=plots_dir, show_plot=show_plots
        )
        results["plots"]["clustermap"] = clustermap_fig

    # Create correlation network if requested
    if create_network:
        logger.info("\nCreating correlation network visualization...")
        network_fig = create_correlation_network(
            corr_matrix,
            threshold=max(
                0.3, threshold - 0.2
            ),  # Lower threshold for network to show more connections
            output_dir=plots_dir,
            show_plot=show_plots,
        )
        results["plots"]["network"] = network_fig

    # Create pairplot if requested
    if (
        create_pairplot_bool and len(columns) <= 10
    ):  # Limit to 10 columns to avoid overcrowding
        logger.info("\nCreating pairplot...")
        pairplot_fig = create_pairplot(
            df, columns=columns, output_dir=plots_dir, show_plot=show_plots
        )
        results["plots"]["pairplot"] = pairplot_fig
    elif create_pairplot_bool:
        logger.info("\nSkipping pairplot due to large number of columns (>10).")

    logger.info("\n=== Correlation Analysis Complete ===")

    return results
