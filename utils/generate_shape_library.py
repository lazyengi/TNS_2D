"""
================================================================================
generate_shape_library.py
================================================================================

This utility script is responsible for generating the 'shape_library.pkl' file,
which serves as the central database of extended object shapes for the TNS-2D simulator.

The script reads raw vertex data from a specified HDF5 file. For each entry,
it processes the vertices to construct a standardized 'contourShapeMatrix' and a
triangulated 'surfaceShapeMatrix'. Optionally, it can also compute a detailed
Line-of-Sight (LoS) visibility table.

The final output is a pickled list of dictionaries, where each dictionary
represents a single extended object shape and its associated data, including both
contour and surface representations.

--------------------------------------------------------------------------------
Author  : Matteo Tesori (matteotesori@gmail.com)
Date    : 2025-10-17
All wrongs reserved.
--------------------------------------------------------------------------------
"""

# ==============================================================================
# SECTION 0: PREAMBLE
# ==============================================================================

# --- Add the project root to the Python path ---
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# --- Imports ---
import h5py
import numpy as np
import pickle
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import triangle as tr

# --- Path Definitions ---
ROOT_DIR           = Path(__file__).resolve().parent.parent
DATA_DIR           = ROOT_DIR / "data"
H5_INPUT_PATH      = DATA_DIR / "raw_aircraft_shape_library.h5"
SHAPE_LIBRARY_PATH = DATA_DIR / "aircraft_shape_library.pkl"

# ==============================================================================
# SECTION 1: CONFIGURATION
# ==============================================================================

# --- Generation Flags ---
GENERATE_VISIBILITY_TABLES    = True                                                            # If True, computes LoS tables for each shape
FORCE_REGENERATION            = True                                                            # If False, loads existing library for validation
RUN_TRIANGULATION_VALIDATION  = True                                                            # If True, validates the generated triangulation for each shape
RUN_GRID_VALIDATION           = True                                                            # If True, validates the LoS grid point coverage before generation
RUN_PERFORMANCE_VALIDATION    = True                                                            # If True, runs a performance check after generation
RUN_VISUAL_VALIDATION         = True                                                            # If True, runs a visual LoS check after generation

# --- Visibility (LoS) Table Parameters ---
VISIBILITY_PARAMS = {
    'angleResolution': np.deg2rad(10.0),                                                        # Angular step size for the grid                                   [rad]
    'rangeResolution': 10.0,                                                                    # Radial step size for the grid                                    [m]
    'maximumRange'   : 75.0,                                                                    # Maximum distance from the extended object to compute visibility  [m]
    'minimumRange'   :  2.0,                                                                    # Minimum distance from the extended object to compute visibility  [m]
    'rangeDensity'   :  3.0                                                                     # Higher values create more circles closer to the min range
}

# ==============================================================================
# SECTION 2: HELPER FUNCTIONS
# ==============================================================================

#--- Contour Shape Matrix Generation ---

def unfold(foldedVertices):
    """Unfolds a symmetrical set of vertices defined by one half."""
    fv          = np.asarray(foldedVertices).ravel(order = 'F')                                 # Flatten the input vertices into a 1D array
    foldedVL    = fv.reshape((2, -1), order = 'F')                                              # Reshape into a 2xN matrix (left half)
    foldedVR    = np.fliplr(foldedVL * np.array([[1], [-1]]))                                   # Create the right half by flipping + inverting the y-coordinate
    combined    = np.hstack((foldedVL[:, :-1], foldedVR[:, :-1]))                               # Combine the left + right halves, avoiding duplicate center
    return combined.reshape((-1, 1), order = 'F')                                               # Return the full set of vertices as a single column vector

def circularClip(unfoldedVertices):
    """Creates a list of edges connecting vertices in a closed loop."""
    n       = unfoldedVertices.shape[0] // 2                                                    # Calculate the total number of vertices
    ears    = np.full((2, n), np.nan)                                                           # Initialize a matrix to store edge connections
    for i in range(n):                                                                          # Iterate through each vertex
        ears[0, i] = i                                                                          # Set the start of the edge to the current vertex index
        ears[1, i] = i + 1 if i < n - 1 else 0                                                  # Set the end of the edge to the next vertex, wrapping around
    return ears                                                                                 # Return the matrix of edge connections

def shapeVector(unfoldedVertices, ears):
    """Constructs a vector of edge start and end points."""
    tildeSv = np.empty((0, 1))                                                                  # Initialize an empty vector to store the edge data
    n       = ears.shape[1]                                                                     # Get the number of edges
    for i in range(n):                                                                          # Iterate through each edge connection.
        edgeSlice = np.full((4, 1), np.nan)                                                     # Temporary slice for the edge's coordinates [x1, y1, x2, y2]
        for j in range(2):                                                                      
            ij                           = int(ears[j, i])                                      # Get the vertex index for the start/end point
            edgeSlice[2 * j : 2 * j + 2] = unfoldedVertices[2 * ij : 2 * ij + 2]                # Extract the (x, y) coordinates from the vertex list
        tildeSv = np.vstack((tildeSv, edgeSlice))                                               # Append the current edge's coordinates to the main vector
    return tildeSv                                                                              # Return the complete vector of edge coordinates

def contourShapeMatrixGenerator(foldedVertices):
    """Generates the final 5xN contour shape matrix from folded vertex data."""
    unfoldedVertices    = unfold(foldedVertices)                                                # Create the full symmetrical set of vertices
    tildeSv             = shapeVector(unfoldedVertices, circularClip(unfoldedVertices))         # Define the edges and format them
    S4                  = np.asarray(tildeSv).reshape(-1, 4).T                                  # Reshape into a 4xN matrix [x1, y1, x2, y2] for each edge
    d                   = np.sqrt((S4[2] - S4[0])**2 + (S4[3] - S4[1])**2)                      # Calculate the length of each edge
    return np.vstack((S4, d))                                                                   # Stack the edge lengths to create the final 5xN matrix

def contourShapeMatrixGenerator(contourShapeMatrix: np.ndarray) -> np.ndarray:
    """Converts an edge-based contour matrix into a triangle-based surface matrix."""
    vertices = contourShapeMatrix[:2, :].T                                                      # Extract the unique vertices from the contour matrix   
    if vertices.shape[0] < 3: return np.empty((7, 0))                                           # Return empty if less than 3 vertices (no surface possible)

    num_vertices = vertices.shape[0]                                                            # Get the number of unique vertices       
    segments     = np.array([[i, (i + 1) % num_vertices] for i in range(num_vertices)])         # Define segments connecting each vertex in a closed loop
    poly_data    = {'vertices': vertices, 'segments': segments}                                 # Prepare data for triangulation
    
    triangulation = tr.triangulate(poly_data, 'p')                                              # Perform triangulation using the 'triangle' library
    tri_vertices, simplices = triangulation['vertices'], triangulation['triangles']             # Extract the triangulated vertices and simplices
    
    surfaceShapeMatrix = np.zeros((7, len(simplices)))                                          # Initialize the surface shape matrix
    for i, simplex in enumerate(simplices):
        v0                         = tri_vertices[simplex[0]]                                   # Get the vertices of the triangle
        v1                         = tri_vertices[simplex[1]]
        v2                         = tri_vertices[simplex[2]]
        surfaceShapeMatrix[0:2, i] = v0                                                         # Store the vertices in the matrix
        surfaceShapeMatrix[2:4, i] = v1 
        surfaceShapeMatrix[4:6, i] = v2
        surfaceShapeMatrix[6, i]   = 0.5 * abs(v0[0] * (v1[1] - v2[1]) +                        # Calculate the area of the triangle
                                               v1[0] * (v2[1] - v0[1]) + 
                                               v2[0] * (v0[1] - v1[1]))
        
    return surfaceShapeMatrix

# --- Visibility (LoS) Table Generation ---

def _generate_grid_points(visibilityParameters):
    """Generates a polar grid of points."""
    # --- Unpack visibility parameters ---
    angleResolution = visibilityParameters['angleResolution']                                   # Unpack angular resolution from parameters
    rangeResolution = visibilityParameters['rangeResolution']                                   # Unpack radial resolution from parameters
    maximumRange    = visibilityParameters['maximumRange']                                      # Unpack maximum range from parameters
    minimumRange    = visibilityParameters['minimumRange']                                      # Unpack minimum range from parameters
    rangeDensity    = visibilityParameters['rangeDensity']                                      # Unpack range density (circle clustering) parameter

    # --- Step 1: Generate Radii with Exponential Spacing ---
    nRadii  = int(np.round(maximumRange / rangeResolution))                                     # Calculate the number of concentric circles (radii)
    j       = np.arange(1, nRadii + 1)                                                          # Create an index for each circle
    radii   = minimumRange + (maximumRange - minimumRange) * \
            (np.exp(rangeDensity * j / nRadii) - 1) / (np.exp(rangeDensity) - 1)                # Exponential formula to cluster radii near the minimum range

    # --- Step 2: Generate Points with Constant Tangential Spacing ---
    all_points          = []                                                                    # Initialize a list to hold all grid points
    tangential_spacing  = minimumRange * angleResolution                                        # Target tangential spacing based on the smallest circle

    for r in radii:                                                                             # Iterate through each calculated radius
        num_points_on_circle = int(np.round(2 * np.pi * r / tangential_spacing))                # Calculate number of points for this circle to maintain spacing
        if num_points_on_circle < 1: continue                                                   # Skip if the circle is too small to contain any points

        angles  = np.linspace(0, 2 * np.pi, num_points_on_circle, endpoint = False)             # Generate evenly spaced angles for this circle
        x       = r * np.cos(angles)                                                            # Convert polar coordinates to Cartesian x
        y       = r * np.sin(angles)                                                            # Convert polar coordinates to Cartesian y
        all_points.append(np.column_stack([x, y]))                                              # Add the points for this circle to the master list

    if not all_points: return np.empty((0, 2))                                                  # If no points were generated, return an empty array
    return np.vstack(all_points)                                                                # Combine all points from all circles into a single array

def segments_intersect_point(p1, p2, q1, q2):
    """(Optimized) Determines if two line segments intersect and returns the intersection point."""
    r           = p2 - p1                                                                       # Direction vector for segment 1
    s           = q2 - q1                                                                       # Direction vector for segment 2
    r_cross_s   = r[0]*s[1] - r[1]*s[0]                                                         # Calculate the 2D cross product of the direction vectors
    if r_cross_s == 0: return None                                                              # If zero, lines are parallel or collinear; no unique intersection
    t = ((q1 - p1)[0]*s[1] - (q1 - p1)[1]*s[0]) / r_cross_s                                     # Calculate parameter 't' for the intersection point on segment 1
    u = ((q1 - p1)[0]*r[1] - (q1 - p1)[1]*r[0]) / r_cross_s                                     # Calculate parameter 'u' for the intersection point on segment 2
    if 0 <= t <= 1 and 0 <= u <= 1:                                                             # Check if intersection point lies within both segments 
        return p1 + t*r                                                                         # If it does, calculate and return the intersection point
    return None                                                                                 # Otherwise, the segments do not intersect

def compute_edge_LoS_flag(sensor_pos, edge_idx, contourShapeMatrix):
    """(Optimized) Computes if a single edge is visible from a sensor position."""
    n_edges     = contourShapeMatrix.shape[1]                                                   # Get the total number of edges in the shape
    edge_center = (contourShapeMatrix[:2, edge_idx] + contourShapeMatrix[2:4, edge_idx]) / 2    # Calculate the center point of the target edge

    for j in range(n_edges):                                                                    # Iterate through all other edges to check for occlusion
        if j == edge_idx: continue                                                              # Skip checking the edge against itself
        if segments_intersect_point(sensor_pos, edge_center, 
                                    contourShapeMatrix[:2, j], 
                                    contourShapeMatrix[2:4, j]) is not None:                    # Check if LoS to the target edge is blocked by another edge
            return 0                                                                            # If blocked, return a visibility flag of 0
    return 1                                                                                    # If no intersections are found, the edge is visible; return 1

def compute_all_LoS_flags(sensor_position, extended_object_contourShapeMatrix):
    """Computes visibility flags for all edges of an extended object from a given point."""
    n_edges = extended_object_contourShapeMatrix.shape[1]                                       # Get the total number of edges
    flags   = np.zeros(n_edges, dtype=int)                                                      # Initialize an array of flags, one for each edge
    for i in range(n_edges):                                                                    # Iterate through each edge
        flags[i] = compute_edge_LoS_flag(sensor_position, i, 
                                         extended_object_contourShapeMatrix)                    # Compute the visibility flag for the current edge and store it
    return flags                                                                                # Return the complete array of visibility flags

def generate_visibility_table(contourShapeMatrix, visibilityParameters):
    """Generates a lookup table for Line-of-Sight (LoS) visibility."""
    grid_points_flat = _generate_grid_points(visibilityParameters)                              # Generate the grid of points around the extended object
    n_points         = grid_points_flat.shape[0]                                                # Get the total number of grid points
    nEdges           = contourShapeMatrix.shape[1]                                              # Get the number of edges in the extended object shape

    LoSTable = {                                                                                # Initialize the LoSTable dictionary structure
        'coords' : grid_points_flat,                                                            # To store the flat array of grid point coordinates
        'flags'  : np.zeros((n_points, nEdges), dtype = np.uint8),                              # To store visibility flags for each point and edge
        'meta'   : {**visibilityParameters, 'totalGridPoints': n_points}                        # To store the generation parameters + grid point count 
    }

    for i in tqdm(range(n_points), desc = "  Generating LoS table", leave = False):             # Iterate through each grid point with a progress bar
        currentPoint             = grid_points_flat[i]                                          # Get the coordinates of the current grid point
        LoSTable['flags'][i, :]  = compute_all_LoS_flags(currentPoint, 
                                                         contourShapeMatrix[:4, :])             # Compute and store visibility flags for all edges from this point

    return LoSTable                                                                             # Return the completed LoS table

# ==============================================================================
# SECTION 3: TRIANGULATION VALIDATION
# ==============================================================================

def validate_triangulation_visuals(shape_library):
    """Interactively plots the triangulation over each shape to validate correctness."""
    print("-" * 60)                                                                             # Print a separator for clean output
    print("Starting Triangulation pre-validation...")                                           # Announce the start of the validation
    print("Press any key to cycle through the shapes.")                                         # Provide user instructions

    bg_color    = (38/255, 55/255, 59/255)                                                      # Define the background color for the dark theme
    text_color  = 'white'                                                                       # Define the text color for the dark theme
    plt.ion()                                                                                   # Turn on interactive mode for Matplotlib
    fig, ax = plt.subplots(figsize = (8, 8))                                                    # Create a new figure and axis for plotting
    fig.patch.set_facecolor(bg_color)                                                           # Set the figure's background color
    ax.set_facecolor(bg_color)                                                                  # Set the axis' background color
    fig.canvas.manager.set_window_title('Triangulation Visualizer')                             # Set the title of the plot window
    ax.tick_params(colors = text_color)                                                         # Set the color of the tick marks and labels
    for spine in ax.spines.values(): spine.set_color(text_color)                                # Set the color of the plot's bounding box (spines)

    for entry in shape_library:
        ax.clear()                                                                              # Clear the axis for the new plot 
        contour_matrix = entry['contourShapeMatrix']                                            # Get the contour shape matrix
        surface_matrix = entry['surfaceShapeMatrix']                                            # Get the surface shape matrix
        vertices       = contour_matrix[:2, :].T                                                # Extract the unique vertices from the contour matrix

        # --- Plot Triangulation ---
        for i in range(surface_matrix.shape[1]):                                                 
            triangle_coords = surface_matrix[:6, i].reshape(3, 2)                               # Get the vertices of the triangle
            triangle_patch  = plt.Polygon(triangle_coords, edgecolor = 'orange', 
                                          facecolor = 'red', alpha = 0.1, lw = 2.5)             # Create a semi-transparent triangle patch
            ax.add_patch(triangle_patch)                                                        # Add the triangle patch to the plot
            
        # --- Plot Contour on Top ---
        contour_patch = plt.Polygon(vertices, edgecolor = 'orange', 
                                    facecolor = 'none', lw = 2.5, label = 'Original Contour')   # Create the contour polygon patch
        ax.add_patch(contour_patch)                                                             # Add the contour patch to the plot
        
        # --- Final Plot Styling ---
        ax.set_aspect('equal', 'box')                                                           # Ensure the plot has a 1:1 aspect ratio
        ax.set_xlabel("x [m]", color = text_color, font = 'monospace')                          # Set the x-axis label
        ax.set_ylabel("y [m]", color = text_color, font = 'monospace')                          # Set the y-axis label
        ax.set_title(f"Shape: {entry['fullName']}", color = text_color, font = 'monospace')     # Set the plot title with the extended object's name
        ax.grid(True, linestyle = '--', alpha = 0.2)
        
        # --- Adaptive Axis ---
        x_coords = vertices[:, 0]                                                               # Extract x-coordinates for axis limits
        y_coords = vertices[:, 1]                                                               # Extract y-coordinates for axis limits
        margin   = np.mean(np.ptp(vertices, axis = 0)) * 0.1                                    # Calculate a margin based on the shape size
        ax.set_xlim(np.min(x_coords) - margin, np.max(x_coords) + margin)                       # Set x-axis limits with margin
        ax.set_ylim(np.min(y_coords) - margin, np.max(y_coords) + margin)                       # Set y-axis limits with margin
        
        fig.canvas.draw()                                                                       # Redraw the canvas with the new plot
        fig.canvas.flush_events()                                                               # Process any pending UI events    
        plt.waitforbuttonpress()                                                                # Pause execution and wait for a key press

    plt.ioff()                                                                                  # Turn off interactive mode
    plt.close(fig)                                                                              # Close the plot window
    print("Triangulation validation complete.")                                                 # Announce the end of the validation
    print("-" * 60)                                                                             # Print another separator

# ==============================================================================
# SECTION 4: GRID COVERAGE VALIDATION
# ==============================================================================

def validate_los_grid_coverage(shape_library, visibility_params):
    """Interactively plots the LoS grid points over each shape to validate coverage."""
    print("-" * 60)                                                                             # Print a separator for clean output
    print("Starting LoS grid coverage pre-validation...")                                       # Announce the start of the validation
    print("Press any key to cycle through the shapes.")                                         # Provide user instructions

    grid_points = _generate_grid_points(visibility_params)                                      # Generate the grid points using the specified parameters
    bg_color    = (38/255, 55/255, 59/255)                                                      # Define the background color for the dark theme
    text_color  = 'white'                                                                       # Define the text color for the dark theme
    plt.ion()                                                                                   # Turn on interactive mode for Matplotlib
    fig, ax = plt.subplots(figsize = (8, 8))                                                    # Create a new figure and axis for plotting
    fig.patch.set_facecolor(bg_color)                                                           # Set the figure's background color
    ax.set_facecolor(bg_color)                                                                  # Set the axis' background color
    fig.canvas.manager.set_window_title('LoS Grid Coverage Visualizer')                         # Set the title of the plot window
    ax.tick_params(colors = text_color)                                                         # Set the color of the tick marks and labels
    for spine in ax.spines.values(): spine.set_color(text_color)                                # Set the color of the plot's bounding box (spines)

    for entry in shape_library:                                                                 # Loop through each extended object shape in the library
        ax.clear()                                                                              # Clear the axis for the new plot
        
        # --- Plot Extended Object Shape ---
        shapeMat = entry['contourShapeMatrix']                                                  # Get the shape matrix for the current extended object
        for k in range(shapeMat.shape[1]):                                                      # Iterate through each edge in the shape matrix
            x_vals = [shapeMat[0, k], shapeMat[2, k]]                                           # Get the x-coordinates for the edge's start and end points
            y_vals = [shapeMat[1, k], shapeMat[3, k]]                                           # Get the y-coordinates for the edge's start and end points
            ax.plot(x_vals, y_vals, color = 'white', linewidth = 1.5)                           # Plot the edge as a white line
        
        # --- Plot LoS Grid Points ---
        ax.scatter(grid_points[:, 0], grid_points[:, 1], s = 1, color = 'yellow', alpha = 0.5)  # Plot all grid points as small, semi-transparent yellow dots

        # --- Final Plot Styling ---
        ax.set_aspect('equal', 'box')                                                           # Ensure the plot has a 1:1 aspect ratio
        ax.set_xlabel("x [m]", color = text_color, font = 'monospace')                          # Set the x-axis label
        ax.set_ylabel("y [m]", color = text_color, font = 'monospace')                          # Set the y-axis label
        ax.set_title(f"Shape: {entry['fullName']}", color = text_color, font = 'monospace')     # Set the plot title with the extended object's name
        ax.grid(True, linestyle = '--', alpha = 0.2)                                            # Add a faint grid for reference
        ax.axis([-120, 120, -120, 120])                                                         # Set fixed axis limits for consistent scale
        
        fig.canvas.draw()                                                                       # Redraw the canvas with the new plot
        fig.canvas.flush_events()                                                               # Process any pending UI events
        plt.waitforbuttonpress()                                                                # Pause execution and wait for a key press

    plt.ioff()                                                                                  # Turn off interactive mode
    plt.close(fig)                                                                              # Close the plot window
    print("Grid coverage validation complete.")                                                 # Announce the end of the validation
    print("-" * 60)                                                                             # Print another separator

# ==============================================================================
# SECTION 5: PERFORMANCE VALIDATION
# ==============================================================================

def validate_los_performance(shape_entry, num_sensors = 100):
    """Runs a performance and correctness check on the LoS flag computation."""
    print("Running LoS performance validation, please wait...")                                 # Announce the start of the performance test
    contourShapeMatrix = shape_entry['contourShapeMatrix']                                      # Get the shape matrix from the provided library entry
    sensor_positions   = np.random.uniform(low = -50, high = 50, size = (num_sensors, 2))       # Generate a set of random sensor positions for testing

    def segments_intersect_point_slow(p1, p2, q1, q2):                                          # Define a slow, robust implementation for segment intersection
        b, A = np.array([p2 - p1, q2 - q1]), np.column_stack(((p2 - p1), -(q2 - q1)))           # Set up the linear system to solve for intersection
        if np.linalg.matrix_rank(A) < 2: return None                                            # If matrix is singular, lines are parallel
        try: t = np.linalg.lstsq(A, q1 - p1, rcond = None)[0]                                   # Solve the linear system for parameters t and u
        except np.linalg.LinAlgError: return None                                               # Catch potential errors during solving
        if 0 <= t[0] <= 1 and 0 <= t[1] <= 1: return p1 + t[0] * b[0]                           # Check if intersection is within both segments and return point
        return None                                                                             # Otherwise, no valid intersection

    def compute_edge_LoS_flag_slow(sensor_pos, edge_idx, contourShapeMatrix):                   # Define a slow reference implementation for LoS flag computation
        n_edges     = contourShapeMatrix.shape[1]                                               # Get number of edges
        edge_center = (contourShapeMatrix[:2, edge_idx] + contourShapeMatrix[2:4, 
                                                                             edge_idx]) / 2     # Get edge centers
        for j in range(n_edges):                                                                # Iterate through all other edges
            if j == edge_idx: continue                                                          # Skip the target edge itself
            if segments_intersect_point_slow(sensor_pos, edge_center, 
                                             contourShapeMatrix[:2, j], 
                                             contourShapeMatrix[2:4, j]) is not None: return 0  # Check for intersection
        return 1                                                                                # Return 1 if no occlusions found

    start_time   = time.perf_counter()                                                          # Start the timer for the slow implementation
    results_slow = [np.array([compute_edge_LoS_flag_slow(pos, i, contourShapeMatrix)
                                for i in range(contourShapeMatrix.shape[1])
                            ]) for pos in sensor_positions]                                     # Run slow computation
    print(f"Slow implementation took: {time.perf_counter() - start_time:.4f} seconds")          # Print the elapsed time for the slow version

    start_time   = time.perf_counter()                                                          # Reset and start the timer for the fast implementation
    results_fast = [compute_all_LoS_flags(pos, contourShapeMatrix) for pos in sensor_positions] # Run the fast computation
    print(f"Fast implementation took: {time.perf_counter() - start_time:.4f} seconds")          # Print the elapsed time for the fast version

    all_match = all(np.array_equal(a, b) for a, b in zip(results_slow, results_fast))           # Ensure all results match between implementations
    print(f"All outputs match: {all_match}")                                                    # Print whether the outputs match
    print("-" * 60)                                                                             # Print a separator

# ==============================================================================
# SECTION 6: SHAPE LIBRARY VISUAL VALIDATION 
# ==============================================================================

def validate_shape_library_visuals(shape_library):
    """Interactively plots each shape to visually verify the LoS tables"""
    print("Starting visual validation of LoS flags...")                                         # Announce the start of the visual validation
    print("Press any key to cycle through the shapes.")                                         # Provide user instructions

    bg_color, text_color = (38/255, 55/255, 59/255), 'white'                                    # Define standard colors for the plot's dark theme
    plt.ion()                                                                                   # Turn on interactive mode for Matplotlib
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (16, 8))                                     # Create a figure with two side-by-side subplots
    fig.patch.set_facecolor(bg_color)                                                           # Set the background color for the entire figure
    fig.canvas.manager.set_window_title('Shape Library LoS Visualizer')                         # Set the title of the plot window

    for ax in [ax1, ax2]:                                                                       # Loop through both axes to apply common styling
        ax.set_facecolor(bg_color)                                                              # Set the axis background color
        ax.tick_params(colors = text_color)                                                     # Set the color of the tick marks and labels
        for spine in ax.spines.values():                                                        # Loop through the plot's bounding box (spines)
            spine.set_color(text_color)                                                         # Set the spine color to white

    for entry in shape_library:                                                                 # Loop through each extended object shape in the library
        ax1.clear()                                                                             # Clear the left subplot for the new drawing
        ax2.clear()                                                                             # Clear the right subplot for the new drawing
        shapeMat = entry['contourShapeMatrix']                                                  # Unpack the shape matrix for the current entry
        LoSTable = entry['LoSTable']                                                            # Unpack the LoS table for the current entry
        sensor_pos = np.random.uniform(low = -50, high = 50, size = (2,))                       # Generate a random sensor position for this validation instance

        grid_points         = LoSTable['coords']                                                # Get the grid point coordinates from the LoS table
        distances           = np.linalg.norm(grid_points - sensor_pos, axis=1)                  # Euclidean distance from the sensor to every grid point
        nearest_idx         = np.argmin(distances)                                              # Find the index of the grid point closest to the sensor
        nearest_grid_point  = grid_points[nearest_idx]                                          # Get the coordinates of that closest grid point
        visibility_flags    = LoSTable['flags'][nearest_idx, :]                                 # Get the pre-computed visibility flags for that grid point

        # --- LEFT PLOT (ax1): Grid and Sensor Position ---
        for k in range(shapeMat.shape[1]):                                                      # Loop through each edge of the extended object shape
            ax1.plot([shapeMat[0, k], shapeMat[2, k]], 
                     [shapeMat[1, k], shapeMat[3, k]], color = 'white', linewidth = 1.5)        # Plot the extended object shape outline
            
        ax1.scatter(grid_points[:, 0], grid_points[:, 1], 
                    s = 2, c = 'y', alpha = 0.3, label = 'LoS Grid')                            # Plot all LoS grid points as semi-transparent yellow dots
        ax1.plot(sensor_pos[0], sensor_pos[1], 'o', c = 'w', ms = 8, 
                 label = 'Sensor Position')                                                     # Plot the random sensor position as a white circle
        ax1.plot(nearest_grid_point[0], nearest_grid_point[1], 'X', 
                 c = 'r', ms = 10, label = 'Nearest Grid Point')                                # Highlight the nearest grid point with a red 'X'
        ax1.set_aspect('equal')                                                                 # Set the aspect ratio to be equal, preventing distortion
        ax1.set_xlabel("x [m]", color = text_color, fontname = 'monospace')                     # Set the x-axis label
        ax1.set_ylabel("y [m]", color = text_color, fontname = 'monospace')                     # Set the y-axis label
        ax1.set_title("Grid and Sensor Placement", color = text_color, fontname = 'monospace')  # Set the title for the left subplot
        ax1.legend(loc = 'upper right', facecolor = bg_color, edgecolor = 'white', 
                   labelcolor = 'white', prop = {'family': 'monospace'})                        # Add a legend with appropriate styling

        # --- RIGHT PLOT (ax2): Visibility Result ---
        for k in range(shapeMat.shape[1]):                                                      # Loop through each edge of the extended object shape again
            x_vals  = [shapeMat[0, k], shapeMat[2, k]]                                          # Get the x-coordinates for the current edge
            y_vals  = [shapeMat[1, k], shapeMat[3, k]]                                          # Get the y-coordinates for the current edge
            color   = 'green' if visibility_flags[k] > 0 else 'red'                             # Determine the color based on the visibility flag
            ax2.plot(x_vals, y_vals, color = color, linewidth = 2)                              # Plot the edge with the determined color
        
        ax2.set_aspect('equal', 'box')                                                          # Set the aspect ratio to be equal
        ax2.set_xlabel("x [m]", color = text_color, fontname = 'monospace')                     # Set the x-axis label
        ax2.set_ylabel("y [m]", color = text_color, fontname = 'monospace')                     # Set the y-axis label
        
        # --- Adaptive Axis for Right Plot ---
        x_coords = np.concatenate((shapeMat[0, :], shapeMat[2, :]))                             # Collect all x-coordinates of the shape's vertices
        y_coords = np.concatenate((shapeMat[1, :], shapeMat[3, :]))                             # Collect all y-coordinates of the shape's vertices
        margin   = 5.0                                                                          # Define a 5-meter margin for the plot
        ax2.set_xlim(np.min(x_coords) - margin, np.max(x_coords) + margin)                      # Set x-axis limits to fit the shape plus the margin
        ax2.set_ylim(np.min(y_coords) - margin, np.max(y_coords) + margin)                      # Set y-axis limits to fit the shape plus the margin
        
        ax2.set_title(f"Visibility for: {entry['shortName']}", 
                      color = text_color, fontname = 'monospace')                               # Set the title for the right subplot
        
        # --- Custom Legend for Right Plot ---
        legend_elements = [                                                                     # Create custom handles for the legend
            Line2D([0], [0], color = 'green', lw = 2, label = 'Visible Edges'),                 # Create a green line for 'Visible Edges'
            Line2D([0], [0], color = 'red', lw = 2, label = 'Occluded Edges'),                  # Create a red line for 'Occluded Edges'
            Line2D([0], [0], marker = 'o', color = 'white', label = 'Sensor Position', 
                   markersize = 8, linestyle = 'None')                                          # Create a white marker for 'Sensor Position'
        ]
        ax2.legend(handles = legend_elements, loc = 'upper right', facecolor = bg_color, 
                   edgecolor = 'white', labelcolor = 'white', prop = {'family': 'monospace'})   # Add the custom legend to the plot
        
        fig.suptitle(f"Shape: {entry['fullName']}", color = text_color, 
                     fontsize = 16, fontname = 'monospace')                                     # Set the main title for the entire figure
        fig.tight_layout(rect = [0, 0.03, 1, 0.95])                                             # Adjust subplot parameters for a tight layout
        fig.canvas.draw()                                                                       # Redraw the canvas
        fig.canvas.flush_events()                                                               # Process UI events
        plt.waitforbuttonpress()                                                                # Wait for a key press before continuing

    plt.ioff()                                                                                  # Turn off interactive mode
    plt.close(fig)                                                                              # Close the figure window
    print("Visual validation complete.")                                                        # Announce the end of the validation
    print("-" * 60)                                                                             # Print a separator

# ==============================================================================
# SECTION 7: EXECUTION
# ==============================================================================

if __name__ == "__main__":
    shape_library = []                                                                          # Initialize an empty list to hold the shape library data
    if FORCE_REGENERATION:                                                                      # Check if we need to regenerate the library from scratch
        print("Starting shape library generation (FORCE_REGENERATION=True)...")                 # Announce the start of the generation process

        if not H5_INPUT_PATH.exists():                                                          # Check if the required input HDF5 file exists
            print(f"Error: Input file not found at '{H5_INPUT_PATH}'")                          # Print an error message if the input file is missing
        else:
            # --- Step 1: Generate preliminary library (shape matrices only) ---
            prelim_library = []                                                                 # Initialize a temporary library to hold basic shape info
            with h5py.File(H5_INPUT_PATH, 'r') as f:                                            # Open the HDF5 file for reading
                for key in tqdm(list(f.keys()), desc='Generating Shape Matrices'):              # Iterate through each extended object shape in the file
                    csm   = contourShapeMatrixGenerator(f[key]["foldedVertices"][:])            # Generate the contour matrix first
                    ssm   = contourShapeMatrixGenerator(csm)                                    # Now generate the surface matrix from the contour matrix
                    entry = {                                                                   # Create a dictionary for the current shape
                        'fullName'           : str(f[key].attrs.get("name", "Unknown")),        # Get the full name from HDF5 attributes
                        'shortName'          : str(f[key].attrs.get("id", "N/A")),              # Get the short name/ID from HDF5 attributes
                        'contourShapeMatrix' : csm,                                             # Store the generated contour matrix
                        'surfaceShapeMatrix' : ssm                                              # Store the generated surface matrix
                    }
                    prelim_library.append(entry)                                                # Add the new entry to our preliminary library
            print(f"Processed {len(prelim_library)} shapes from HDF5 file.")                    # Print a summary of how many shapes were processed
            
            # --- Step 2: Pre-validation of triangulation (if enabled) ---
            if RUN_TRIANGULATION_VALIDATION:                                                    # Check if triangulation validation is enabled                                
                validate_triangulation_visuals(prelim_library)                                  # Run the triangulation pre-validation    

            # --- Step 3: Pre-validation of grid points (if enabled) ---    
            if RUN_GRID_VALIDATION and GENERATE_VISIBILITY_TABLES:                              # Check if grid validation is enabled
                validate_los_grid_coverage(prelim_library, VISIBILITY_PARAMS)                   # Run the grid coverage pre-check

            # --- Step 4: Generate LoS tables (the time-consuming part) ---
            generation_start_time = time.perf_counter()                                         # Start a timer specifically for the LoS table generation
            if GENERATE_VISIBILITY_TABLES:                                                      # Check if LoS table generation is enabled
                print("Generating Line-of-Sight tables...")                                     # Announce the start of this intensive process
                for entry in tqdm(prelim_library, desc='Processing shapes'):                    # Loop through the shapes with a progress bar
                    entry['LoSTable'] = generate_visibility_table(entry['contourShapeMatrix'], 
                                                                  VISIBILITY_PARAMS)            # Generate and add the LoS table to each entry
            shape_library = prelim_library                                                      # The preliminary library is now the final, complete library
            
            gen_duration = time.perf_counter() - generation_start_time                          # Calculate the duration of the LoS table generation
            if GENERATE_VISIBILITY_TABLES:                                                      # If tables were generated, print the time taken
                gen_minutes, gen_seconds = divmod(gen_duration, 60)                             # Convert the duration to minutes and seconds
                print(f"LoS table generation took: {int(gen_minutes)} minutes "
                      f"and {gen_seconds:.2f} seconds.")                                        # Print the formatted duration

            # --- Step 5: Save the final library to a file ---
            print(f"Saving library to '{SHAPE_LIBRARY_PATH}'...")                               # Announce that the library is being saved to a file
            with open(SHAPE_LIBRARY_PATH, 'wb') as f: pickle.dump(shape_library, f)             # Open the output file and save the library using pickle
            print("Shape library generation complete.")                                         # Confirm that the generation process is finished
    else:
        print("Skipping generation (FORCE_REGENERATION=False). Loading existing library...")    # If not regenerating, announce that we are loading a file
        if SHAPE_LIBRARY_PATH.exists():                                                         # Check if the pre-existing library file is found
            with open(SHAPE_LIBRARY_PATH, 'rb') as f:                                           # Open the file for reading in binary mode
                shape_library = pickle.load(f)                                                  # Load the shape library from the pickle file
            print(f"Successfully loaded {len(shape_library)} shapes from "
                  f"'{SHAPE_LIBRARY_PATH}'.")                                                   # Confirm successful loading and print the number of shapes
        else:
            print(f"Error: Shape library not found at '{SHAPE_LIBRARY_PATH}'.")                 # Print an error if the file doesn't exist
            print("Set FORCE_REGENERATION = True to generate it.")                              # Advise the user on how to fix the issue

    # --- Optional Validation Steps (Post-generation or Load) ---
    if shape_library:                                                                           # Proceed with validation only if a library exists
        has_los_tables = 'LoSTable' in shape_library[0]                                         # Check if the loaded/generated library contains LoS data
        if has_los_tables:                                                                      # If it does, proceed with LoS-dependent validations
            # Grid validation pre-check is handled above. This runs it on a loaded library.
            if RUN_GRID_VALIDATION and not FORCE_REGENERATION:                                  # Run grid validation if loading a file and flag is enabled
                validate_los_grid_coverage(shape_library, shape_library[0]['LoSTable']['meta'])
            if RUN_PERFORMANCE_VALIDATION:                                                      # Check if the performance test is enabled
                validate_los_performance(shape_library[-1])                                     # Run the performance validation on the last shape
            if RUN_VISUAL_VALIDATION:                                                           # Check if the visual validation is enabled
                validate_shape_library_visuals(shape_library)                                   # Run the main visual validation test
        elif RUN_GRID_VALIDATION or RUN_PERFORMANCE_VALIDATION or RUN_VISUAL_VALIDATION:        # If validation is on but there's no LoS data
            print("\nWarning: Validation tests were enabled, but the shape library")            # Print a warning message
            print("does not contain Line-of-Sight (LoS) tables. Skipping validation.")          # Explain that the tests are being skipped

