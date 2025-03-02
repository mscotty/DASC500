from scipy.interpolate import splprep, splev
import numpy as np

from scipy.interpolate import splprep, splev
import numpy as np

def interpolate_points(points, num_points=200, tolerance=1e-8):
    """Interpolates points along a curve to increase resolution."""
    if len(points) < 3:
        return points  # Not enough points for interpolation

    # Sort the points by x-coordinate
    sorted_points = points[np.argsort(points[:, 0])]

    # Remove duplicates and average close points
    unique_points = []
    i = 0
    while i < len(sorted_points):
        current_x = sorted_points[i, 0]
        close_points = []
        while i < len(sorted_points) and abs(sorted_points[i, 0] - current_x) < tolerance:
            close_points.append(sorted_points[i])
            i += 1
        if close_points:
            avg_y = np.mean([p[1] for p in close_points])
            unique_points.append([current_x, avg_y])

    unique_points = np.array(unique_points)

    if len(unique_points) < 3:
        return unique_points

    x, y = unique_points[:, 0], unique_points[:, 1]
    try:
        tck, u = splprep([x, y], s=0)  # s=0 for no smoothing
        u_new = np.linspace(0, 1, num_points)
        x_new, y_new = splev(u_new, tck)
        return np.column_stack((x_new, y_new))
    except ValueError as e:
        print(f"Error interpolating points: {e}")
        return unique_points #Return the points before the error.