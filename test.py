"""
================================================================================
shape_triangulator.py
================================================================================

This module provides a function to convert an extended object's contour
representation into a surface representation through polygon triangulation.

The core logic uses a robust Constrained Delaunay Triangulation (CDT)
via the 'triangle' library, ensuring correctness for complex concave polygons.

--------------------------------------------------------------------------------
Dependencies:
- numpy
- triangle (a wrapper for Shewchuk's robust C-based triangulator)
- matplotlib (for visualization example)
- pickle (for loading shape library)
--------------------------------------------------------------------------------
"""

import numpy as np
import triangle as tr
import matplotlib.pyplot as plt
from typing import List
import pickle
import os
import random

# ==============================================================================
# SECTION 1: CONVERSION FUNCTION
# ==============================================================================

def convert_contour_to_surface(contourShapeMatrix: np.ndarray) -> np.ndarray:
    """
    Converts an edge-based contour matrix into a triangle-based surface matrix.

    This function uses a robust Constrained Delaunay Triangulation (CDT)
    to ensure that the triangulation perfectly matches the polygon's surface,
    even for complex and highly concave shapes.

    The function performs the following steps:
    1. Extracts the unique, ordered vertices from the input contour matrix.
    2. Defines the polygon's edges (segments) as constraints.
    3. Uses the 'triangle' library to perform the constrained triangulation.
       
    4. Constructs the final surface matrix from the resulting triangles,
       where each column represents a valid triangle and its area.

    Args:
        contourShapeMatrix (np.ndarray): A 5xN matrix where each column
                                         represents an edge of the polygon.
                                         - Rows 0, 1: Coords of the start vertex.
                                         - Rows 2, 3: Coords of the end vertex.
                                         - Row 4: Length of the edge.

    Returns:
        np.ndarray: A 7xM matrix where M is the number of triangles in the
                    triangulation.
                    - Rows 0, 1: Coords of the first vertex (v0).
                    - Rows 2, 3: Coords of the second vertex (v1).
                    - Rows 4, 5: Coords of the third vertex (v2).
                    - Row 6: Area of the triangle.
    """
    # --- Step 1: Extract unique, ordered vertices ---
    vertices = contourShapeMatrix[:2, :].T
    if vertices.shape[0] < 3:
        return np.empty((7, 0)) # Not a valid polygon

    # --- Step 2: Define segments (the edges of the polygon) ---
    num_vertices = vertices.shape[0]
    segments = np.array([[i, i + 1] for i in range(num_vertices - 1)] + [[num_vertices - 1, 0]])

    # --- Step 3: Define the polygon structure for the 'triangle' library ---
    poly_data = {'vertices': vertices, 'segments': segments}

    # --- Step 4: Perform Constrained Delaunay Triangulation ---
    # The 'p' flag indicates a Planar Straight Line Graph (the polygon).
    # The 'q' flag requests quality triangulation (avoids skinny triangles).
    # The 'a' flag imposes a maximum triangle area constraint if needed, but we'll omit for now.
    triangulation = tr.triangulate(poly_data, 'p')
    
    # The result contains the new triangulation's vertices and triangle indices
    tri_vertices = triangulation['vertices']
    simplices = triangulation['triangles']
    
    num_valid_triangles = len(simplices)
    
    # --- Step 5: Construct the surfaceShapeMatrix ---
    surfaceShapeMatrix = np.zeros((7, num_valid_triangles))
    
    for i, simplex in enumerate(simplices):
        v0, v1, v2 = tri_vertices[simplex[0]], tri_vertices[simplex[1]], tri_vertices[simplex[2]]
        
        # Populate the vertex coordinates
        surfaceShapeMatrix[0:2, i] = v0
        surfaceShapeMatrix[2:4, i] = v1
        surfaceShapeMatrix[4:6, i] = v2
        
        # Calculate and populate the triangle area using the Shoelace formula
        area = 0.5 * abs(v0[0]*(v1[1] - v2[1]) + v1[0]*(v2[1] - v0[1]) + v2[0]*(v0[1] - v1[1]))
        surfaceShapeMatrix[6, i] = area
        
    return surfaceShapeMatrix

# ==============================================================================
# SECTION 2: EXAMPLE USAGE & VISUALIZATION
# ==============================================================================

def visualize_triangulation(vertices, surface_matrix, title):
    """Helper function to plot the polygon and its triangulation."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot the original polygon outline
    polygon_patch = plt.Polygon(vertices, edgecolor='red', facecolor='none', lw=2, label='Original Polygon')
    ax.add_patch(polygon_patch)

    # Plot the triangulation
    for i in range(surface_matrix.shape[1]):
        triangle_coords = surface_matrix[:6, i].reshape(3, 2)
        triangle_patch = plt.Polygon(triangle_coords, edgecolor='blue', facecolor='cyan', alpha=0.5)
        ax.add_patch(triangle_patch)

    ax.scatter(vertices[:, 0], vertices[:, 1], c='red', zorder=5)
    
    ax.set_title(title)
    ax.set_aspect('equal', adjustable='box')
    ax.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    # --- Step 1: Load the shape library ---
    SHAPE_LIBRARY_PATH = os.path.join('data', 'shape_library.pkl')
    
    if not os.path.exists(SHAPE_LIBRARY_PATH):
        print(f"Error: Shape library not found at '{SHAPE_LIBRARY_PATH}'")
        print("Please ensure the file exists and the script is run from the correct directory.")
    else:
        print(f"--- Loading shape library from '{SHAPE_LIBRARY_PATH}' ---")
        with open(SHAPE_LIBRARY_PATH, 'rb') as f:
            shape_library = pickle.load(f)
        
        # --- Step 2: Select a random shape to test ---
        num_shapes = len(shape_library)
        shape_to_test_idx = random.randint(0, num_shapes - 1)
        selected_shape = shape_library[shape_to_test_idx]
        
        print(f"\n--- Testing with randomly selected Shape #{shape_to_test_idx} ---")
        
        # --- Step 3: Extract contour matrix and generate surface matrix ---
        contour_matrix = selected_shape['contourShapeMatrix']
        surface_matrix = convert_contour_to_surface(contour_matrix)
        
        print(f"Input Vertices: {contour_matrix.shape[1]}")
        print(f"Output Triangles: {surface_matrix.shape[1]}")
        print("Output surfaceShapeMatrix shape:", surface_matrix.shape)

        # --- Step 4: Visualize the result ---
        # Extract the vertices from the contour matrix for plotting
        vertices_for_plot = contour_matrix[:2, :].T
        visualize_triangulation(vertices_for_plot, 
                                surface_matrix, 
                                f'Triangulation of Shape #{shape_to_test_idx} from Library (Robust Method)')

