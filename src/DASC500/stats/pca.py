import os
import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

def find_optimal_components_variance(explained_variance_ratio, threshold=0.95):
    """
    Finds the optimal number of components based on explained variance threshold.
    
    Parameters:
    -----------
    explained_variance_ratio : array-like
        The explained variance ratio of each component
    threshold : float, default=0.95
        The cumulative variance threshold to reach
        
    Returns:
    --------
    int
        Optimal number of components
    """
    cumulative_variance = np.cumsum(explained_variance_ratio)
    optimal_k = np.argmax(cumulative_variance >= threshold) + 1
    return optimal_k


def kaiser_criterion(pca):
    """
    Finds the optimal number of components based on the Kaiser criterion.
    
    Parameters:
    -----------
    pca : PCA object
        The fitted PCA model
        
    Returns:
    --------
    int
        Number of components with eigenvalues > 1
    """
    eigenvalues = pca.explained_variance_ 
    optimal_k = np.sum(eigenvalues > 1)
    return optimal_k


def create_scree_plot(explained_variance_ratio, optimal_k=None, variance_threshold=0.95, 
                     output_dir=None, show_plot=True):
    """
    Creates a scree plot for PCA explained variance.
    
    Parameters:
    -----------
    explained_variance_ratio : array-like
        The explained variance ratio of each component
    optimal_k : int, optional
        The optimal number of components to highlight
    variance_threshold : float, default=0.95
        The variance threshold to show on the plot
    output_dir : str, optional
        Directory to save the plot
    show_plot : bool, default=True
        Whether to display the plot
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    
    
    fig = plt.figure(figsize=(10, 6))
    
    # Individual variance
    plt.plot(range(1, len(explained_variance_ratio) + 1), 
            explained_variance_ratio, marker='o', linestyle='--')
    
    # Cumulative variance
    cumulative_variance = np.cumsum(explained_variance_ratio)
    plt.plot(range(1, len(explained_variance_ratio) + 1), 
            cumulative_variance, marker='o', linestyle='-')
    
    # Threshold line
    plt.axhline(y=variance_threshold, color='r', linestyle='-', alpha=0.5)
    
    # Optimal k line if provided
    if optimal_k is not None:
        plt.axvline(x=optimal_k, color='r', linestyle='--', alpha=0.5)
    
    plt.title("Scree Plot with Cumulative Variance")
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio / Cumulative Variance")
    plt.grid(True)
    
    legend_items = ['Individual Variance', 'Cumulative Variance', 
                   f'{variance_threshold*100}% Threshold']
    if optimal_k is not None:
        legend_items.append(f'Optimal Components: {optimal_k}')
    
    plt.legend(legend_items)
    
    # Save or show plot
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, "pca_scree_plot.png")
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Scree plot saved to: {plot_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def create_pca_scatter(pca_df, loadings_df, explained_variance, feature_names=None,
                      output_dir=None, show_plot=True):
    """
    Creates a scatter plot of the first two principal components with feature vectors.
    
    Parameters:
    -----------
    pca_df : pandas DataFrame
        DataFrame containing the PCA results
    loadings_df : pandas DataFrame
        DataFrame containing the loadings
    explained_variance : array-like
        The explained variance ratio of each component
    feature_names : list, optional
        List of original feature names
    output_dir : str, optional
        Directory to save the plot
    show_plot : bool, default=True
        Whether to display the plot
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    
    if pca_df.shape[1] < 2:
        print("Need at least 2 principal components for scatter plot.")
        return None
    
    fig = plt.figure(figsize=(10, 8))
    
    # Plot scatter points
    plt.scatter(pca_df.iloc[:, 0], pca_df.iloc[:, 1], alpha=0.7)
    
    # Axis labels with variance
    plt.xlabel(f"{pca_df.columns[0]} ({explained_variance[0]:.4f})")
    plt.ylabel(f"{pca_df.columns[1]} ({explained_variance[1]:.4f})")
    plt.title('PCA: First Two Principal Components')
    plt.grid(True)
    
    # Add feature vectors if loadings are provided
    if loadings_df is not None:
        for i, feature in enumerate(loadings_df.index):
            feature_name = feature_names[i] if feature_names is not None else feature
            plt.arrow(0, 0, 
                     loadings_df.iloc[i, 0] * 5, 
                     loadings_df.iloc[i, 1] * 5, 
                     head_width=0.1, head_length=0.1, fc='r', ec='r')
            plt.text(loadings_df.iloc[i, 0] * 5.2, 
                    loadings_df.iloc[i, 1] * 5.2, 
                    feature_name, color='r')
        
        # Add circle
        circle = plt.Circle((0, 0), 5, fill=False, linestyle='--')
        plt.gca().add_patch(circle)
        plt.axis('equal')
    
    # Save or show plot
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, "pca_scatter_plot.png")
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"PCA scatter plot saved to: {plot_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def create_loadings_heatmap(loadings_df, output_dir=None, show_plot=True):
    """
    Creates a heatmap of PCA component loadings.
    
    Parameters:
    -----------
    loadings_df : pandas DataFrame
        DataFrame containing the loadings
    output_dir : str, optional
        Directory to save the plot
    show_plot : bool, default=True
        Whether to display the plot
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    fig = plt.figure(figsize=(12, 8))
    sns.heatmap(loadings_df, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('PCA Component Loadings')
    plt.tight_layout()
    
    # Save or show plot
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, "pca_loadings_heatmap.png")
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"PCA loadings heatmap saved to: {plot_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def perform_pca_analysis(df, columns=None, n_components=None, variance_threshold=0.95, 
                       output_dir=None, plot_scree=True, plot_scatter=True, 
                       plot_loadings=True, show_plots=True):
    """
    Perform Principal Component Analysis on numerical columns in a DataFrame.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The DataFrame to analyze
    columns : list, optional
        List of column names to include in PCA (default: all numerical columns)
    n_components : int, optional
        Number of components to keep (if None, determined automatically)
    variance_threshold : float, default=0.95
        Threshold for explained variance when determining optimal components
    output_dir : str, optional
        Directory to save output files and plots
    plot_scree : bool, default=True
        Whether to create scree plot
    plot_scatter : bool, default=True
        Whether to create scatter plot of first two PCs
    plot_loadings : bool, default=True
        Whether to create a heatmap of component loadings
    show_plots : bool, default=True
        Whether to display plots (if in interactive environment)
        
    Returns:
    --------
    tuple
        (pca_df, pca_model, loadings_df) - DataFrame with PCA results, 
        the fitted PCA model, and loadings DataFrame
    """
    # Set up logging
    logger = logging.getLogger('pca_analysis')
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
            file_handler = logging.FileHandler(os.path.join(output_dir, 'pca_analysis.log'))
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    logger.info("\n=== Principal Component Analysis (PCA) ===")
    
    # Select columns for PCA
    if columns is None:
        # Use all numerical columns
        num_df = df.select_dtypes(include=np.number)
        dependent_vars = num_df.columns.tolist()
    else:
        # Use specified columns (ensure they exist and are numerical)
        dependent_vars = [col for col in columns 
                         if col in df.columns and np.issubdtype(df[col].dtype, np.number)]
    
    # Check if we have enough data
    if len(dependent_vars) < 2:
        logger.info("PCA requires at least 2 numerical features. Skipping.")
        return None, None, None
    
    # Drop rows with NaN values in the selected columns
    df_clean = df[dependent_vars].dropna()
    
    if df_clean.shape[0] < 2:
        logger.info("PCA requires at least 2 samples after dropping NaNs. Skipping.")
        return None, None, None
    
    logger.info(f"Performing PCA on {len(dependent_vars)} variables with {df_clean.shape[0]} samples")
    logger.info(f"Variables: {dependent_vars}")
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_clean)
    
    # Determine optimal number of components if not specified
    if n_components is None:
        # Initial PCA to determine optimal components
        initial_pca = PCA()
        initial_pca.fit(scaled_data)
        
        # Calculate optimal components based on variance threshold
        optimal_k = find_optimal_components_variance(initial_pca.explained_variance_ratio_, 
                                                    threshold=variance_threshold)
        
        # Calculate Kaiser criterion recommendation
        kaiser_k = kaiser_criterion(initial_pca)
        
        logger.info(f"Optimal number of components for {variance_threshold*100}% explained variance: {optimal_k}")
        logger.info(f"Kaiser Criterion recommended number of components: {kaiser_k}")
        
        # Use the optimal k for the final PCA
        n_components = optimal_k
        
        # Create scree plot
        if plot_scree:
            create_scree_plot(
                initial_pca.explained_variance_ratio_,
                optimal_k=optimal_k,
                variance_threshold=variance_threshold,
                output_dir=output_dir,
                show_plot=show_plots
            )
    
    # Perform final PCA with determined number of components
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(scaled_data)
    
    # Create DataFrame with PCA results
    column_names = [f"PC{i+1}" for i in range(n_components)]
    pca_df = pd.DataFrame(principal_components, columns=column_names, index=df_clean.index)
    
    # Log explained variance
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.sum(explained_variance)
    
    logger.info(f"\nNumber of components selected: {n_components}")
    logger.info(f"Total explained variance: {cumulative_variance:.4f}")
    logger.info("\nExplained variance by component:")
    for i, var in enumerate(explained_variance):
        logger.info(f"  PC{i+1}: {var:.4f} ({var*100:.2f}%)")
    
    # Create loadings DataFrame
    loadings_df = pd.DataFrame(
        pca.components_.T, 
        columns=column_names, 
        index=dependent_vars
    )
    
    # Log top loadings for each component
    logger.info("\nTop loadings by component:")
    for i, comp in enumerate(column_names):
        # Sort loadings by absolute value
        sorted_loadings = loadings_df[comp].abs().sort_values(ascending=False)
        top_vars = sorted_loadings.index[:3]  # Top 3 variables
        logger.info(f"  {comp}:")
        for var in top_vars:
            logger.info(f"    {var}: {loadings_df.loc[var, comp]:.4f}")
    
    # Create scatter plot of first two PCs if requested
    if plot_scatter and n_components >= 2:
        create_pca_scatter(
            pca_df,
            loadings_df,
            explained_variance,
            feature_names=dependent_vars,
            output_dir=output_dir,
            show_plot=show_plots
        )
    
    # Create loadings heatmap if requested
    if plot_loadings:
        create_loadings_heatmap(
            loadings_df,
            output_dir=output_dir,
            show_plot=show_plots
        )
    
    logger.info("\n=== PCA Analysis Complete ===")
    
    return pca_df, pca, loadings_df


def detect_pca_outliers(df, columns, n_components=None, threshold_factor=3.0, output_dir=None, show_plots=False):
    """
    Detects outliers using PCA and Mahalanobis distance.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The DataFrame to analyze
    columns : list
        List of columns to include in PCA
    n_components : int, optional
        Number of components to use (default: determined automatically)
    threshold_factor : float, default=3.0
        Factor to multiply with median Mahalanobis distance for threshold
    output_dir : str, optional
        Directory to save output files and plots
    show_plots : bool, default=False
        Whether to display plots
        
    Returns:
    --------
    dict
        Dictionary with outlier information
    """
    # Drop rows with NaN values in the selected columns
    df_clean = df[columns].dropna()
    
    if df_clean.shape[0] < 3:
        return {"outlier_indices": [], "message": "Insufficient data after removing NaN values"}
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_clean)
    
    # Perform PCA
    if n_components is None:
        # Use enough components to explain 95% of variance
        pca = PCA(n_components=0.95)
    else:
        pca = PCA(n_components=n_components)
    
    principal_components = pca.fit_transform(scaled_data)
    
    # Calculate Mahalanobis distances
    # For PCA space, this is equivalent to the normalized Euclidean distance
    distances = np.sum(np.square(principal_components), axis=1)
    
    # Determine threshold (using median and MAD for robustness)
    median_dist = np.median(distances)
    mad = np.median(np.abs(distances - median_dist))
    threshold = median_dist + threshold_factor * mad
    
    # Identify outliers
    outlier_mask = distances > threshold
    outlier_indices = df_clean.index[outlier_mask].tolist()
    
    # Create results dictionary
    results = {
        "outlier_indices": outlier_indices,
        "mahalanobis_distances": {idx: distances[i] for i, idx in enumerate(df_clean.index) if outlier_mask[i]},
        "threshold": threshold,
        "n_components": pca.n_components_,
        "explained_variance": pca.explained_variance_ratio_.sum()
    }
    
    # Create visualization if requested
    if output_dir and len(principal_components) > 0 and principal_components.shape[1] >= 2:
        try:
            # Create scatter plot of first two PCs with outliers highlighted
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Plot normal points
            ax.scatter(
                principal_components[~outlier_mask, 0],
                principal_components[~outlier_mask, 1],
                alpha=0.5, label='Normal'
            )
            
            # Plot outliers
            if np.any(outlier_mask):
                ax.scatter(
                    principal_components[outlier_mask, 0],
                    principal_components[outlier_mask, 1],
                    color='red', alpha=0.7, label='Outliers'
                )
            
            ax.set_title('PCA Outlier Detection')
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
            ax.legend()
            
            # Save plot
            os.makedirs(output_dir, exist_ok=True)
            plot_path = os.path.join(output_dir, 'pca_outliers.png')
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            
            if not show_plots:
                plt.close(fig)
                
            results['plot_path'] = plot_path
        except Exception as e:
            results['plot_error'] = str(e)
    
    return results