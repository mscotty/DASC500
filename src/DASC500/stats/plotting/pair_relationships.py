import os
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from matplotlib.gridspec import GridSpec


def create_basic_pairplot(df, columns=None, hue=None, diag_kind='kde', 
                         corner=False, markers=None, height=2.5,
                         output_dir=None, filename='basic_pairplot.png', 
                         show_plot=True):
    """
    Creates a basic pairplot using seaborn's pairplot function.
    
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
    corner : bool, default=False
        Whether to show only the lower triangle
    markers : str or list, optional
        Markers for scatter plots
    height : float, default=2.5
        Height (in inches) of each facet
    output_dir : str, optional
        Directory to save the plot
    filename : str, default='basic_pairplot.png'
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
    
    if not columns or len(columns) < 2:
        return None
    
    # Create pairplot
    g = sns.pairplot(
        df, 
        vars=columns, 
        hue=hue, 
        diag_kind=diag_kind,
        corner=corner,
        markers=markers,
        height=height,
        plot_kws={'alpha': 0.6, 's': 30},
        diag_kws={'alpha': 0.6}
    )
    
    # Set title
    g.fig.suptitle('Pairwise Relationships', y=1.02, fontsize=16)
    g.fig.tight_layout()
    
    # Save plot if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, filename)
        g.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    # Show or close plot
    if show_plot:
        plt.show()
    else:
        plt.close(g.fig)
    
    return g


def create_advanced_pairplot(df, columns=None, hue=None, plot_type='scatter',
                           add_trendline=True, add_corr=True, add_kde=True,
                           output_dir=None, filename='advanced_pairplot.png',
                           show_plot=True):
    """
    Creates an advanced pairplot with customizable features.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The DataFrame to analyze
    columns : list of str, optional
        Specific columns to include (default: all numerical columns)
    hue : str, optional
        Column name for color-coding points
    plot_type : str, default='scatter'
        Type of plot ('scatter', 'hex', 'kde', 'reg')
    add_trendline : bool, default=True
        Whether to add trendlines to scatter plots
    add_corr : bool, default=True
        Whether to add correlation coefficients
    add_kde : bool, default=True
        Whether to add KDE plots on diagonal
    output_dir : str, optional
        Directory to save the plot
    filename : str, default='advanced_pairplot.png'
        Filename for the saved plot
    show_plot : bool, default=True
        Whether to display the plot
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    # Select numerical columns if not specified
    if columns is None:
        num_df = df.select_dtypes(include=np.number)
        columns = num_df.columns.tolist()
    else:
        # Filter to only include columns that exist
        columns = [col for col in columns if col in df.columns]
    
    if not columns or len(columns) < 2:
        return None
    
    # Calculate the number of variables
    n_vars = len(columns)
    
    # Create figure and axes grid
    fig, axes = plt.subplots(n_vars, n_vars, figsize=(n_vars * 2.5, n_vars * 2.5))
    
    # If there's only one row/column, make axes 2D
    if n_vars == 1:
        axes = np.array([[axes]])
    
    # Iterate over variable pairs
    for i, row_var in enumerate(columns):
        for j, col_var in enumerate(columns):
            ax = axes[i, j]
            
            # Diagonal: Distribution plots
            if i == j:
                if add_kde:
                    sns.kdeplot(df[row_var].dropna(), ax=ax, fill=True)
                else:
                    sns.histplot(df[row_var].dropna(), ax=ax, kde=False)
                ax.set_title(row_var)
            
            # Off-diagonal: Relationship plots
            else:
                # Get data without NaNs
                valid_data = df[[row_var, col_var]].dropna()
                
                if hue is not None:
                    valid_data = valid_data.join(df[hue]).dropna()
                
                if not valid_data.empty:
                    if plot_type == 'scatter':
                        if hue is not None:
                            sns.scatterplot(x=col_var, y=row_var, hue=hue, data=valid_data, ax=ax, alpha=0.6)
                        else:
                            sns.scatterplot(x=col_var, y=row_var, data=valid_data, ax=ax, alpha=0.6)
                        
                        # Add trendline if requested
                        if add_trendline:
                            sns.regplot(x=col_var, y=row_var, data=valid_data, ax=ax, 
                                      scatter=False, line_kws={'color': 'red'})
                    
                    elif plot_type == 'hex':
                        if hue is None:  # Hexbin not compatible with hue
                            sns.hexbin(x=col_var, y=row_var, data=valid_data, ax=ax, gridsize=15, cmap='Blues')
                    
                    elif plot_type == 'kde':
                        if hue is None:  # 2D KDE not easily compatible with hue
                            sns.kdeplot(x=col_var, y=row_var, data=valid_data, ax=ax, fill=True, cmap='Blues')
                    
                    elif plot_type == 'reg':
                        if hue is not None:
                            sns.lmplot(x=col_var, y=row_var, hue=hue, data=valid_data, ax=ax)
                        else:
                            sns.regplot(x=col_var, y=row_var, data=valid_data, ax=ax)
                    
                    # Add correlation coefficient if requested
                    if add_corr and hue is None:
                        corr_val = valid_data[row_var].corr(valid_data[col_var])
                        ax.annotate(f'r = {corr_val:.2f}', xy=(0.05, 0.95), xycoords='axes fraction',
                                  fontsize=10, ha='left', va='top',
                                  bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7))
            
            # Set labels only on edge plots
            if i == n_vars - 1:
                ax.set_xlabel(col_var)
            else:
                ax.set_xlabel('')
            
            if j == 0:
                ax.set_ylabel(row_var)
            else:
                ax.set_ylabel('')
    
    plt.tight_layout()
    
    # Save plot if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, filename)
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    # Show or close plot
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def create_feature_relationship_grid(df, target, features=None, n_bins=10,
                                    output_dir=None, filename='feature_relationship_grid.png',
                                    show_plot=True):
    """
    Creates a grid of plots showing relationships between features and a target variable.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The DataFrame to analyze
    target : str
        Name of the target variable
    features : list of str, optional
        List of feature variables (default: all numerical columns except target)
    n_bins : int, default=10
        Number of bins for target variable in trend plots
    output_dir : str, optional
        Directory to save the plot
    filename : str, default='feature_relationship_grid.png'
        Filename for the saved plot
    show_plot : bool, default=True
        Whether to display the plot
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    # Ensure target exists in DataFrame
    if target not in df.columns:
        return None
    
    # Select numerical columns if features not specified
    if features is None:
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        features = [col for col in num_cols if col != target]
    else:
        # Filter to only include columns that exist
        features = [col for col in features if col in df.columns and col != target]
    
    if not features:
        return None
    
    # Calculate grid dimensions
    n_features = len(features)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    # Create figure
    fig = plt.figure(figsize=(n_cols * 5, n_rows * 4))
    gs = GridSpec(n_rows, n_cols, figure=fig)
    
    # Create plots for each feature
    for i, feature in enumerate(features):
        row, col = divmod(i, n_cols)
        ax = fig.add_subplot(gs[row, col])
        
        # Get data without NaNs
        valid_data = df[[feature, target]].dropna()
        
        if valid_data.empty:
            ax.text(0.5, 0.5, f"No valid data for\n{feature} vs {target}", 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Create scatter plot
        sns.scatterplot(x=feature, y=target, data=valid_data, ax=ax, alpha=0.6)
        
        # Add trend line
        sns.regplot(x=feature, y=target, data=valid_data, ax=ax, 
                   scatter=False, line_kws={'color': 'red'})
        
        # Calculate and display correlation
        corr_val = valid_data[feature].corr(valid_data[target])
        ax.annotate(f'r = {corr_val:.2f}', xy=(0.05, 0.95), xycoords='axes fraction',
                  fontsize=10, ha='left', va='top',
                  bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7))
        
        # Set title and labels
        ax.set_title(f'{feature} vs {target}')
        ax.set_xlabel(feature)
        ax.set_ylabel(target)
    
    plt.tight_layout()
    
    # Save plot if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, filename)
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    # Show or close plot
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def create_joint_distribution_plot(df, x, y, kind='scatter', hue=None, 
                                 output_dir=None, filename=None, show_plot=True):
    """
    Creates a joint distribution plot for two variables.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The DataFrame to analyze
    x : str
        Name of the x-axis variable
    y : str
        Name of the y-axis variable
    kind : str, default='scatter'
        Kind of plot ('scatter', 'kde', 'hex', 'reg', 'resid')
    hue : str, optional
        Column name for color-coding points
    output_dir : str, optional
        Directory to save the plot
    filename : str, optional
        Filename for the saved plot (default: 'joint_{x}_{y}.png')
    show_plot : bool, default=True
        Whether to display the plot
        
    Returns:
    --------
    seaborn.axisgrid.JointGrid
        The created joint plot
    """
    # Ensure variables exist in DataFrame
    if x not in df.columns or y not in df.columns:
        return None
    
    # Generate filename if not provided
    if filename is None:
        safe_x = x.replace('/', '_').replace('\\', '_')
        safe_y = y.replace('/', '_').replace('\\', '_')
        filename = f'joint_{safe_x}_{safe_y}.png'
    
    # Create joint plot
    g = sns.jointplot(
        data=df,
        x=x,
        y=y,
        kind=kind,
        hue=hue,
        height=8,
        ratio=3,
        marginal_kws=dict(bins=20, fill=True),
        joint_kws=dict(alpha=0.7)
    )
    
    # Add correlation coefficient
    valid_data = df[[x, y]].dropna()
    if not valid_data.empty:
        corr_val = valid_data[x].corr(valid_data[y])
        g.ax_joint.annotate(f'r = {corr_val:.2f}', xy=(0.05, 0.95), xycoords='axes fraction',
                          fontsize=12, ha='left', va='top',
                          bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7))
    
    # Set title
    g.fig.suptitle(f'Joint Distribution: {x} vs {y}', y=1.02, fontsize=16)
    
    # Save plot if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, filename)
        g.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    # Show or close plot
    if show_plot:
        plt.show()
    else:
        plt.close(g.fig)
    
    return g


def create_conditional_plot(df, x, y, z, kind='scatter', n_levels=5, 
                          output_dir=None, filename=None, show_plot=True):
    """
    Creates a conditional plot showing relationship between x and y conditioned on z.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The DataFrame to analyze
    x : str
        Name of the x-axis variable
    y : str
        Name of the y-axis variable
    z : str
        Name of the conditioning variable
    kind : str, default='scatter'
        Kind of plot ('scatter', 'line', 'reg')
    n_levels : int, default=5
        Number of levels for conditioning variable
    output_dir : str, optional
        Directory to save the plot
    filename : str, optional
        Filename for the saved plot (default: 'conditional_{x}_{y}_{z}.png')
    show_plot : bool, default=True
        Whether to display the plot
        
    Returns:
    --------
    seaborn.axisgrid.FacetGrid
        The created facet grid
    """
    # Ensure variables exist in DataFrame
    if x not in df.columns or y not in df.columns or z not in df.columns:
        return None
    
    # Generate filename if not provided
    if filename is None:
        safe_x = x.replace('/', '_').replace('\\', '_')
        safe_y = y.replace('/', '_').replace('\\', '_')
        safe_z = z.replace('/', '_').replace('\\', '_')
        filename = f'conditional_{safe_x}_{safe_y}_{safe_z}.png'
    
    # Create bins for conditioning variable
    if df[z].dtype.kind in 'bifc':  # If z is numeric
        # Create bins
        z_bins = pd.qcut(df[z].dropna(), q=n_levels, duplicates='drop')
        df_with_bins = df.copy()
        df_with_bins[f'{z}_binned'] = z_bins
        z_col = f'{z}_binned'
    else:
        # Use z as is if categorical
        df_with_bins = df
        z_col = z
    
    # Create conditional plot
    g = sns.FacetGrid(df_with_bins, col=z_col, col_wrap=min(3, n_levels), height=4, aspect=1.2)
    
    if kind == 'scatter':
        g.map(sns.scatterplot, x, y, alpha=0.7)
    elif kind == 'line':
        g.map(sns.lineplot, x, y)
    elif kind == 'reg':
        g.map(sns.regplot, x, y, scatter_kws={'alpha': 0.5})
    
    # Add title and labels
    g.fig.suptitle(f'Relationship between {x} and {y} conditioned on {z}', y=1.02, fontsize=16)
    g.set_axis_labels(x, y)
    g.set_titles(col_template="{col_name}")
    
    # Adjust layout
    g.tight_layout()
    
    # Save plot if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, filename)
        g.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    # Show or close plot
    if show_plot:
        plt.show()
    else:
        plt.close(g.fig)
    
    return g


def create_3d_scatter(df, x, y, z, hue=None, angle=(30, 45),
                    output_dir=None, filename=None, show_plot=True):
    """
    Creates a 3D scatter plot for three variables.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The DataFrame to analyze
    x : str
        Name of the x-axis variable
    y : str
        Name of the y-axis variable
    z : str
        Name of the z-axis variable
    hue : str, optional
        Column name for color-coding points
    angle : tuple, default=(30, 45)
        Viewing angle (elevation, azimuth)
    output_dir : str, optional
        Directory to save the plot
    filename : str, optional
        Filename for the saved plot (default: '3d_scatter_{x}_{y}_{z}.png')
    show_plot : bool, default=True
        Whether to display the plot
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    # Ensure variables exist in DataFrame
    if x not in df.columns or y not in df.columns or z not in df.columns:
        return None
    
    # Generate filename if not provided
    if filename is None:
        safe_x = x.replace('/', '_').replace('\\', '_')
        safe_y = y.replace('/', '_').replace('\\', '_')
        safe_z = z.replace('/', '_').replace('\\', '_')
        filename = f'3d_scatter_{safe_x}_{safe_y}_{safe_z}.png'
    
    # Get data without NaNs
    if hue is not None and hue in df.columns:
        valid_data = df[[x, y, z, hue]].dropna()
    else:
        valid_data = df[[x, y, z]].dropna()
        hue = None
    
    if valid_data.empty:
        return None
    
    # Create 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    if hue is not None:
        # Color by hue variable
        scatter = ax.scatter(
            valid_data[x],
            valid_data[y],
            valid_data[z],
            c=valid_data[hue],
            cmap='viridis',
            alpha=0.7,
            s=30
        )
        
        # Add colorbar
        cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
        cbar.set_label(hue)
    else:
        # Single color
        ax.scatter(
            valid_data[x],
            valid_data[y],
            valid_data[z],
            alpha=0.7,
            s=30
        )
    
    # Set labels and title
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)
    ax.set_title(f'3D Scatter Plot: {x}, {y}, and {z}')
    
    # Set viewing angle
    ax.view_init(elev=angle[0], azim=angle[1])
    
    # Save plot if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, filename)
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    # Show or close plot
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_pair_relationships(df, columns=None, hue=None, plot_type='basic',
                          target=None, n_bins=10, show_3d=False, 
                          output_dir=None, show_plots=True):
    """
    Comprehensive analysis of pairwise relationships between variables.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The DataFrame to analyze
    columns : list of str, optional
        Specific columns to analyze (default: all numerical columns)
    hue : str, optional
        Column name for color-coding points
    plot_type : str, default='basic'
        Type of pairplot ('basic', 'advanced', 'target_focused')
    target : str, optional
        Target variable for target_focused plots
    n_bins : int, default=10
        Number of bins for target variable in trend plots
    show_3d : bool, default=False
        Whether to create 3D scatter plots for selected triplets
    output_dir : str, optional
        Directory to save output files and plots
    show_plots : bool, default=True
        Whether to display plots
        
    Returns:
    --------
    dict
        Dictionary containing analysis results
    """
    # Set up logging
    logger = logging.getLogger('pair_relationships')
    logger.setLevel(logging.INFO)
    
    # Create console handler if not already present
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Add file handler if output_dir is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(output_dir, 'pair_relationships.log'))
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    logger.info("\n=== Pairwise Relationship Analysis ===")
    
    # Select numerical columns if not specified
    if columns is None:
        num_df = df.select_dtypes(include=np.number)
        columns = num_df.columns.tolist()
    else:
        # Filter to only include columns that exist
        columns = [col for col in columns if col in df.columns]
    
    if len(columns) < 2:
        logger.info("Pairwise analysis requires at least 2 columns.")
        return {}
    
    logger.info(f"Analyzing pairwise relationships for {len(columns)} columns")
    
    # Initialize results dictionary
    results = {
        'plots': {}
    }
    
    # Create plots directory if needed
    plots_dir = os.path.join(output_dir, 'pair_relationship_plots') if output_dir else None
    
    # Create appropriate plots based on plot_type
    if plot_type == 'basic':
        logger.info("\nCreating basic pairplot...")
        pairplot = create_basic_pairplot(
            df,
            columns=columns,
            hue=hue,
            output_dir=plots_dir,
            show_plot=show_plots
        )
        results['plots']['basic_pairplot'] = pairplot
    
    elif plot_type == 'advanced':
        logger.info("\nCreating advanced pairplot...")
        adv_pairplot = create_advanced_pairplot(
            df,
            columns=columns,
            hue=hue,
            output_dir=plots_dir,
            show_plot=show_plots
        )
        results['plots']['advanced_pairplot'] = adv_pairplot
    
    elif plot_type == 'target_focused' and target is not None:
        if target not in df.columns:
            logger.info(f"Target variable '{target}' not found in DataFrame.")
        else:
            logger.info(f"\nCreating feature-target relationship grid for target: {target}...")
            features = [col for col in columns if col != target]
            
            if features:
                feature_grid = create_feature_relationship_grid(
                    df,
                    target=target,
                    features=features,
                    n_bins=n_bins,
                    output_dir=plots_dir,
                    show_plot=show_plots
                )
                results['plots']['feature_target_grid'] = feature_grid
    
        # Create joint plots for selected pairs
    if len(columns) >= 2:
        logger.info("\nCreating joint distribution plots for selected pairs...")
        joint_plots = {}
        
        # Select a subset of pairs to avoid too many plots
        max_pairs = min(5, len(columns) * (len(columns) - 1) // 2)
        
        # Prioritize pairs with target if specified
        if target is not None and target in columns:
            pairs = [(target, col) for col in columns if col != target]
            pairs = pairs[:max_pairs]
        else:
            # Otherwise, select some random pairs
            import random
            import itertools
            random.seed(42)  # For reproducibility
            pairs = list(itertools.combinations(columns, 2))
            if len(pairs) > max_pairs:
                pairs = random.sample(pairs, max_pairs)
        
        for x, y in pairs:
            logger.info(f"  Creating joint plot for {x} vs {y}...")
            joint_plot = create_joint_distribution_plot(
                df,
                x=x,
                y=y,
                hue=hue,
                output_dir=plots_dir,
                show_plot=show_plots
            )
            joint_plots[(x, y)] = joint_plot
        
        results['plots']['joint_plots'] = joint_plots
    
    # Create conditional plots if a conditioning variable is available
    if hue is not None and hue in df.columns and len(columns) >= 2:
        logger.info(f"\nCreating conditional plots with conditioning variable: {hue}...")
        conditional_plots = {}
        
        # Select a few pairs for conditional plots
        max_pairs = min(3, len(columns) * (len(columns) - 1) // 2)
        
        # Prioritize pairs with target if specified
        if target is not None and target in columns:
            pairs = [(target, col) for col in columns if col != target]
            pairs = pairs[:max_pairs]
        else:
            # Otherwise, select some random pairs
            import random
            import itertools
            random.seed(42)  # For reproducibility
            pairs = list(itertools.combinations(columns, 2))
            if len(pairs) > max_pairs:
                pairs = random.sample(pairs, max_pairs)
        
        for x, y in pairs:
            logger.info(f"  Creating conditional plot for {x} vs {y} conditioned on {hue}...")
            cond_plot = create_conditional_plot(
                df,
                x=x,
                y=y,
                z=hue,
                output_dir=plots_dir,
                show_plot=show_plots
            )
            conditional_plots[(x, y, hue)] = cond_plot
        
        results['plots']['conditional_plots'] = conditional_plots
    
    # Create 3D scatter plots if requested
    if show_3d and len(columns) >= 3:
        logger.info("\nCreating 3D scatter plots for selected triplets...")
        scatter_3d_plots = {}
        
        # Select a few triplets for 3D plots
        max_triplets = min(3, len(columns) * (len(columns) - 1) * (len(columns) - 2) // 6)
        
        # Prioritize triplets with target if specified
        if target is not None and target in columns:
            import itertools
            triplets = [(x, y, target) for x, y in itertools.combinations([col for col in columns if col != target], 2)]
            triplets = triplets[:max_triplets]
        else:
            # Otherwise, select some random triplets
            import random
            import itertools
            random.seed(42)  # For reproducibility
            triplets = list(itertools.combinations(columns, 3))
            if len(triplets) > max_triplets:
                triplets = random.sample(triplets, max_triplets)
        
        for x, y, z in triplets:
            logger.info(f"  Creating 3D scatter plot for {x}, {y}, {z}...")
            scatter_3d = create_3d_scatter(
                df,
                x=x,
                y=y,
                z=z,
                hue=hue,
                output_dir=plots_dir,
                show_plot=show_plots
            )
            scatter_3d_plots[(x, y, z)] = scatter_3d
        
        results['plots']['3d_scatter_plots'] = scatter_3d_plots
    
    # Calculate and log correlation statistics
    if len(columns) >= 2:
        logger.info("\nCalculating correlation statistics:")
        
        # Compute correlation matrix
        corr_matrix = df[columns].corr()
        
        # Find strongest positive and negative correlations
        corr_pairs = []
        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns):
                if i < j:  # Lower triangle only
                    corr_pairs.append((col1, col2, corr_matrix.loc[col1, col2]))
        
        # Sort by absolute correlation
        corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        # Log top correlations
        logger.info("Top correlations:")
        for col1, col2, corr_val in corr_pairs[:5]:  # Show top 5
            logger.info(f"  {col1} - {col2}: {corr_val:.4f}")
        
        # Log strongest positive correlation
        pos_pairs = [p for p in corr_pairs if p[2] > 0]
        if pos_pairs:
            col1, col2, corr_val = pos_pairs[0]
            logger.info(f"\nStrongest positive correlation: {col1} - {col2}: {corr_val:.4f}")
        
        # Log strongest negative correlation
        neg_pairs = [p for p in corr_pairs if p[2] < 0]
        if neg_pairs:
            col1, col2, corr_val = neg_pairs[0]
            logger.info(f"Strongest negative correlation: {col1} - {col2}: {corr_val:.4f}")
        
        # Add correlation statistics to results
        results['correlation_stats'] = {
            'correlation_matrix': corr_matrix,
            'top_correlations': corr_pairs[:5],
            'strongest_positive': pos_pairs[0] if pos_pairs else None,
            'strongest_negative': neg_pairs[0] if neg_pairs else None
        }
    
    logger.info("\n=== Pairwise Relationship Analysis Complete ===")
    
    return results


