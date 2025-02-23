import numpy as np

# Compute Trailing Edge Angle
def trailing_edge_angle(points):
    """
    Compute the trailing edge angle of the airfoil.
    
    @param points: Numpy array of (x, y) coordinates.
    @return: Trailing edge angle in degrees.
    """
    te_idx = np.argmax(points[:, 0])
    te_points = points[np.abs(points[:, 0] - points[te_idx, 0]) < 0.05]
    
    x_fit, y_fit = te_points[:, 0], te_points[:, 1]
    poly = np.polyfit(x_fit, y_fit, 1)  # Linear fit
    angle = np.arctan(poly[0]) * (180 / np.pi)  # Convert to degrees
    
    return angle