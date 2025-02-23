import numpy as np

# Compute Leading Edge Radius
def leading_edge_radius(points):
    """
    Compute the leading edge radius of the airfoil.
    
    @param points: Numpy array of (x, y) coordinates.
    @return: Estimated leading edge radius.
    """
    le_idx = np.argmin(points[:, 0])
    le_points = points[np.abs(points[:, 0] - points[le_idx, 0]) < 0.05]  # Close to LE
    
    x_fit, y_fit = le_points[:, 0], le_points[:, 1]
    fit_circle = np.polyfit(x_fit, y_fit, 2)  # Quadratic fit
    radius = 1 / (2 * fit_circle[0])  # Radius of curvature approximation
    
    return radius