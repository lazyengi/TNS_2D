"""
================================================================================
generate_dataset.py
================================================================================

This script generates a dataset for a machine learning classification task.

The goal is to train a model that can infer an aircraft's shape index (its
class) based solely on a sensor point cloud of that aircraft.

To do this, the script iterates a specified number of times. In each iteration:
1.  A random aircraft shape index is chosen uniformly.
2.  The aircraft's position is chosen uniformly from a circle (fixed distance)
    or a ring (random distance) around the sensor.
3.  The aircraft's heading is chosen uniformly between 0 and 2 * pi radians.
4.  A single sensor, configured with specific noise and resolution, generates a
    point cloud at the origin.
5.  Based on a configuration flag, the point cloud is either kept in world
    coordinates or transformed into the aircraft's local coordinate system.
6.  The final data example, including relevant metadata, is saved.

The final dataset is saved as a pickled dictionary to be used in a separate model
training script.

--------------------------------------------------------------------------------
Author  : Matteo Tesori (matteotesori@gmail.com)
Date    : 2025-10-11
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

# --- Path Definitions ---
ROOT_DIR            = Path(__file__).resolve().parent.parent                         
DATA_DIR            = ROOT_DIR   / "data"
OUTPUT_DIR          = ROOT_DIR   / "output"
SHAPE_LIBRARY_PATH  = DATA_DIR   / "shape_library.pkl"
DATASET_PATH        = OUTPUT_DIR / "full_dataset.pkl"

# --- Imports ---
import numpy as np                                      
import pickle                                          
import random                                          
from pathlib import Path                                
from tqdm import tqdm                                   
import time                                             
import matplotlib.pyplot as plt                         
from matplotlib.patches import Polygon, Circle

from src.tns_2d.simulation.data_structures  import Aircraft, Sensor                     
from src.tns_2d.simulation.simulator        import Simulator    

# ==============================================================================
# SECTION 1: CONFIGURATION
# ==============================================================================

# --- Dataset Settings ---
USE_RANDOM_DISTANCE         = False                                                     # If True, aircraft distance is randomized in [MIN, 2*POSE_DISTANCE]
USE_AIRCRAFT_COORDINATES    = True                                                      # If True, point clouds are in aircraft frame; if False, in world frame
NUM_EXAMPLES                = 1000                                                      # Defines the total number of data samples to generate
POSE_DISTANCE               = 50.0                                                      # The fixed or maximum distance between aircraft and sensor         [m]
MIN_DISTANCE                = 15.0                                                      # The minimum distance if randomization is used                     [m]
DELTA_T                     = 1e-6                                                      # Small time step to get an instant snapshot                        [s]

# --- Sensor Imperfection Settings ---
USE_LOS                     = True                                                      # Toggles Line-of-Sight simulation (self-occlusions) on or off
SENSOR_RESOLUTION           = 2.0                                                       # Number of detections per unit length of an aircraft's edge        [m^-1]
SENSOR_NOISE_STD            = 0.1                                                       # Standard deviation of Gaussian noise added to measurements        [m]

# --- Validation Settings ---
VALIDATE_AFTER_GENERATION   = True                                                      # Flag to run the validation plotting function after generation
NUM_VALIDATION_EXAMPLES     = 10                                                        # Number of random examples to show during validation

# ==============================================================================
# SECTION 2: HELPER FUNCTIONS 
# ==============================================================================

def apply_inverse_roto_translation(points_world, aircraft_position, aircraft_heading):     
    """Transforms points from the world frame to the aircraft's local frame."""
    points_world_2d = np.atleast_2d(points_world)                                       # Ensures the input points array is at least 2D

    # Translation 
    translated_points = points_world_2d - aircraft_position                             # Step 1: Translates points by subtracting the aircraft's position

    # Rotation matrix construction
    h               = aircraft_heading                                                  # Step 2: Prepares for rotation by getting the heading
    cos_h, sin_h    = np.cos(h), np.sin(h)                                              # Calculates cosine and sine of the heading for the rotation matrix
    inv_rot_matrix  = np.array([[cos_h, sin_h],                                         # Defines the 2D inverse rotation matrix
                                 [-sin_h, cos_h]])
    
    # Inverse roto-translation
    points_aircraft = np.dot(inv_rot_matrix, translated_points.T).T                     # Applies the inverse rotation to the translated points

    return points_aircraft.squeeze() if points_world.ndim == 1 else points_aircraft     # Returns the transformed points, squeezing to 1D if the input was 1D

def apply_forward_roto_translation(points_local, aircraft_position, aircraft_heading):
    """Transforms points from the aircraft's local frame to the world frame."""
    points_local_2d = np.atleast_2d(points_local)                                       # Ensures the input points array is at least 2D
    
    # Rotation matrix construction
    h             = aircraft_heading                                                    # Step 1: Prepares for rotation by getting the heading
    cos_h, sin_h  = np.cos(h), np.sin(h)                                                # Calculates cosine and sine for the rotation matrix
    rot_matrix    = np.array([[cos_h, -sin_h],                                          # Defines the 2D forward rotation matrix
                              [sin_h, cos_h]])

    # Forward roto-translation
    rotated_points = np.dot(rot_matrix, points_local_2d.T).T                            # Applies the forward rotation
    points_world   = rotated_points + aircraft_position                                 # Translates the rotated points to the aircraft's world position
    
    return points_world.squeeze() if points_local.ndim == 1 else points_world           # Returns the transformed points

def generate_single_example(shape_library):                                       
    """Generates a single, complete data example for the dataset."""
    # STEP 1. Randomly select an aircraft shape
    num_shapes          = len(shape_library)                                            # Number of available aircraft shapes
    random_shape_idx    = np.random.randint(0, num_shapes)                              # Selects a shape index uniformly at random

    # STEP 2. Position the aircraft (fixed or random distance on a circle/ring)
    random_angle = np.random.uniform(0, 2 * np.pi)                                      # Sample a random angle for the position

    if USE_RANDOM_DISTANCE:
        # Sample uniformly from the area of the ring
        r_squared_min       = MIN_DISTANCE**2
        r_squared_max       = (2 * POSE_DISTANCE)**2
        random_r_squared    = np.random.uniform(r_squared_min, r_squared_max)
        radius              = np.sqrt(random_r_squared)
    else:
        # Use a fixed distance (uniform on the circle's circumference)
        radius = POSE_DISTANCE

    aircraft_position = np.array([radius * np.cos(random_angle), 
                                  radius * np.sin(random_angle)])                       # Calculate the (x, y) position
    
    # STEP 3: Selects a random heading between 0 and 2*pi radians
    aircraft_heading = np.random.uniform(0, 2 * np.pi)                            

    # STEP 4. Configure and run a simulation to generate the point cloud
    aircraft = Aircraft(                                                                # Creates an Aircraft              
        identifier          = 0,                                                       
        shape_idx           = random_shape_idx,                                 
        t_init              = 0.0,                                              
        initial_position    = aircraft_position,                                
        initial_heading     = aircraft_heading,                                 
        initial_speed       = 0,                                                
        long_vals           = np.array([0]),                                    
        long_times          = np.array([DELTA_T]),                              
        centr_vals          = np.array([0]),                                    
        centr_times         = np.array([DELTA_T]),                              
        dt                  = DELTA_T                                           
    )

    sensor = Sensor(                                                                    # Creates a Sensor 
        identifier          = 0,                                                
        position            = np.array([0.0, 0.0]),                             
        heading             = 0.0,                                              
        max_range           = 1000.0,                                           
        min_range           = 0.0,                                              
        min_bearing         = -np.pi,                                           
        max_bearing         =  np.pi,                                           
        pd                  = 1.0,                                              
        clutter_rate        = 0.0,                                              
        dt_sample           = DELTA_T,                                          
        t_offset            = 0.0,                                              
        alpha               = None,                                             
        resolution          = SENSOR_RESOLUTION,                                
        sigma_v             = SENSOR_NOISE_STD                                  
    )

    simulator = Simulator(                                                              # Initializes the Simulator with the scenario objects
        aircraft        = [aircraft],                                           
        sensors         = [sensor],                                             
        shape_library   = shape_library,                                        
        spine_length    = 60.0,                                                 
        global_dt       = DELTA_T,                                              
        max_duration    = DELTA_T * 2,                                          
        use_dropouts    = False,                                                
        use_los         = USE_LOS,                                              
        use_oor         = False,                                                
        verbose         = False                                                 
    )
    simulator.run()                                                                     # Executes the simulation

    sensor_dataset = simulator.sensors[0].dataset                                       # Extracts the point cloud from the sensor
    if not sensor_dataset['measurements']:                                              # Checks if any measurements were generated
        return None                                                                     # If not, returns None to indicate failure

    point_cloud_world = sensor_dataset['measurements'][0]                               # Retrieves the point cloud from the first measurement scan
    
    if USE_AIRCRAFT_COORDINATES:
        # STEPS 5 & 6. Express point cloud and sensor position in aircraft coordinates
        point_cloud_aircraft = apply_inverse_roto_translation(
            point_cloud_world, aircraft_position, aircraft_heading
        )
        sensor_pos_ac = apply_inverse_roto_translation(
            np.array([0.0, 0.0]), aircraft_position, aircraft_heading
        )
        return {
            'point_cloud'   : point_cloud_aircraft,
            'shape_idx'     : random_shape_idx,
            'sensor_pos_ac' : sensor_pos_ac
        }
    else:
        # Keep all data in world coordinates
        return {
            'point_cloud'       : point_cloud_world,
            'shape_idx'         : random_shape_idx,
            'aircraft_position' : aircraft_position,
            'aircraft_heading'  : aircraft_heading,
            'sensor_position'   : np.array([0.0, 0.0])
        }

def validate_dataset(dataset_path, shape_library_path):                                 # Defines the function for visually validating the dataset
    """Loads a generated dataset and plots random examples for visual validation."""
    print("\nStarting dataset validation...")                                           # Prints a status message to the console
    print("Press any key to cycle through random examples.")                            # Prints instructions for the user

    # --- Load Dataset and Shape Library ---
    with open(dataset_path, 'rb') as f:                                                 # Opens the dataset file in binary read mode
        full_dataset = pickle.load(f)                                                   # Loads the pickled dataset from the file
    with open(shape_library_path, 'rb') as f:                                           # Opens the shape library file in binary read mode
        shape_library = pickle.load(f)                                                  # Loads the pickled shape library from the file
    
    metadata          = full_dataset['metadata']                                        # Extracts the metadata dictionary
    dataset           = full_dataset['data']                                            # Extracts the list of data examples from the loaded dictionary
    in_aircraft_frame = metadata.get('use_aircraft_coordinates', True)                  # Checks the coordinate system from metadata
    
    # --- Randomly Sample Examples to Show ---
    num_to_show       = min(NUM_VALIDATION_EXAMPLES, len(dataset))                      # Determines how many examples to show (up to the configured max)
    selected_examples = random.sample(dataset, num_to_show)                             # Randomly samples the specified number of examples from the dataset

    # --- Plotting Setup ---
    fig, ax = plt.subplots(figsize=(8,8))                                               # Creates a new figure and axes for plotting with a specific size
    fig.patch.set_facecolor((38/255, 55/255, 59/255))                                   # Sets the background color of the figure to a dark gray-blue
    ax.set_facecolor((38/255, 55/255, 59/255))                                          # Sets the background color of the plot area to match the figure
    ax.tick_params(colors = 'white')                                                    # Sets the color of the tick marks and labels to white
    for spine in ax.spines.values():                                                    # Loops through all the spines (the plot borders)
        spine.set_color('white')                                                        # Sets the color of each spine to white

    for i, example in enumerate(selected_examples):                                     # Loops through the randomly selected examples
        ax.cla()                                                                        # Clears the axes to prepare for the next plot

        shape_idx       = example['shape_idx']
        shape_matrix    = shape_library[shape_idx]['shapeMatrix']
        base_vertices   = shape_matrix[:2, :].T
        pc              = example['point_cloud']

        if in_aircraft_frame:
            # --- PLOT IN AIRCRAFT COORDINATES ---
            sensor_pos = example['sensor_pos_ac']                                       # Unpacks the sensor position from the example
            
            # Plot underlying geometry based on distance setting
            if metadata.get('use_random_distance', False):                              # Checks if random distance was used
                outer = Circle((0, 0), 2 * POSE_DISTANCE,                               # Creates the outer circle of the ring
                               color = 'c', alpha = 0.1, zorder = 0)
                inner = Circle((0, 0), MIN_DISTANCE,                                    # Creates the inner circle (hole) of the ring
                               color = ax.get_facecolor(), zorder = 0)
                ax.add_patch(outer); ax.add_patch(inner)                                # Adds the two circles to create a ring effect
            else:
                ax.add_patch(Circle((0, 0), POSE_DISTANCE, color = 'c',                 # Adds a dashed circle for the fixed distance case
                                    fill = False, linestyle = '--', zorder = 0))

            # Plot aircraft shape
            ax.add_patch(Polygon(base_vertices,                                         # Creates a transparent polygon for the aircraft's body
                                 facecolor='white', alpha = 0.1, zorder = 1))
            for edge_idx in range(shape_matrix.shape[1]):                               # Loops through each edge of the aircraft shape
                start, end = shape_matrix[:2, edge_idx], shape_matrix[2:4, edge_idx]    # Gets the start and end points of the current edge
                ax.plot([start[0], end[0]], [start[1], end[1]], 'r-',                   # Plots the edge in red
                        lw = 1.0, zorder = 2, 
                        label = 'True Shape' if edge_idx == 0 else "")                  # Adds a label only to the first edge to avoid duplicates

            ax.set_xlabel("X coordinate (Aircraft Frame) [m]",                          # Sets the x-axis label
                          color = 'white', fontname = 'monospace')
            ax.set_ylabel("Y coordinate (Aircraft Frame) [m]",                          # Sets the y-axis label
                          color = 'white', fontname = 'monospace')
            ax.set_xlim([-60, 60]); ax.set_ylim([-60, 60])                              # Sets fixed axis limits for the aircraft frame view
        
        else:
            # --- PLOT IN WORLD COORDINATES ---
            ac_pos     = example['aircraft_position']                                   # Unpacks the aircraft's world position
            ac_heading = example['aircraft_heading']                                    # Unpacks the aircraft's world heading
            sensor_pos = example['sensor_position']                                     # Unpacks the sensor's world position
            
            # Roto-translate aircraft shape to its world position
            world_vertices = apply_forward_roto_translation(base_vertices,              # Calculates the world coordinates of the aircraft's vertices
                                                            ac_pos, ac_heading)
            
            # --- Intelligent Axis Scaling ---
            all_plot_points = [sensor_pos]                                              # Initializes a list of all points to consider for scaling
            if pc.size > 0:                                                             # Checks if there is a point cloud
                all_plot_points.append(pc)                                              # Adds the point cloud to the list
            all_plot_points.append(world_vertices)                                      # Adds the aircraft's vertices to the list
            
            all_points_np = np.vstack(all_plot_points)                                  # Stacks all points into a single NumPy array

            min_coords = all_points_np.min(axis = 0)                                    # Finds the minimum x and y coordinates across all points
            max_coords = all_points_np.max(axis = 0)                                    # Finds the maximum x and y coordinates

            center  = (min_coords + max_coords) / 2                                     # Calculates the center of the bounding box
            max_dim = np.max(max_coords - min_coords)                                   # Finds the largest dimension (width or height) of the data
            
            padding     = 10.0                                                          # Defines a padding amount in meters
            half_size   = (max_dim / 2) + padding                                       # Calculates half the size for the new square plot area
            
            ax.set_xlim(center[0] - half_size, center[0] + half_size)                   # Sets the x-axis limits to be centered on the data
            ax.set_ylim(center[1] - half_size, center[1] + half_size)                   # Sets the y-axis limits to match, creating a square view

            ax.add_patch(Polygon(world_vertices, facecolor = 'white',                   # Adds the aircraft's polygon to the world plot
                                 alpha = 0.1, zorder = 1))
            for edge_idx in range(world_vertices.shape[0]):                             # Loops through the vertices to draw the edges
                start = world_vertices[edge_idx]                                        # Gets the current vertex
                end   = world_vertices[(edge_idx + 1) % world_vertices.shape[0]]        # Gets the next vertex, wrapping around at the end
                ax.plot([start[0], end[0]], [start[1], end[1]], 'r-', lw = 1.0,         # Plots the edge connecting the vertices
                        zorder = 2, label = 'True Shape' if edge_idx == 0 else "")

            ax.set_xlabel("X coordinate (World Frame) [m]",                             # Sets the x-axis label for the world frame view
                          color = 'white', fontname = 'monospace')
            ax.set_ylabel("Y coordinate (World Frame) [m]",                             # Sets the y-axis label for the world frame view
                          color = 'white', fontname = 'monospace')
        
        # --- Plot Common Elements (Sensor Data & Position) ---
        if pc.size > 0:                                                                 # Checks if the point cloud has data
            ax.scatter(pc[:,0], pc[:,1], c = 'yellow', marker = 's',                    # Plots the point cloud detections
                       s = 10, label = 'Point Cloud', alpha = 1.0, zorder = 10)
        ax.plot(sensor_pos[0], sensor_pos[1], 'co',                                     # Plots the sensor's position
                markersize = 10, label = 'Sensor Position')

        # --- Final Plot Adjustments ---
        title_text = f"Validation Example {i+1}/{num_to_show} (Shape Index: {shape_idx})" 
        ax.set_title(title_text, color = 'white', fontname = 'monospace')               # Sets the plot title
        legend = ax.legend(loc='upper right')                                           # Creates the legend in the top-right corner
        for text in legend.get_texts(): text.set_color('white')                         # Sets the legend text to white
        legend.get_frame().set_facecolor((48/255, 65/255, 69/255))                      # Sets the legend background color
        legend.get_frame().set_edgecolor('white')                                       # Sets the legend border color
        
        ax.set_aspect('equal', adjustable = 'box')                                      # Enforces a square aspect ratio by adjusting the plot box
        plt.pause(0.1)                                                                  # Pauses briefly to allow the plot to render

        if not plt.waitforbuttonpress():                                                # Waits for the user to press a key
            print("Plot window closed. Exiting validation.")                            # If the window is closed, prints an exit message
            break                                                                       # Exits the validation loop
    
    plt.close()                                                                         # Closes the figure after validation is complete
    print("Validation complete.")                                                       # Prints a final completion message

# ==============================================================================
# SECTION 3: EXECUTION
# ==============================================================================

if __name__ == "__main__":                                                              # Standard Python entry point
    start_time = time.time()                                                            # Records the starting time of the script

    print("Loading shape library...")                                                   # Prints a status message
    with open(SHAPE_LIBRARY_PATH, 'rb') as f:                                           # Opens the shape library file
        shape_library = pickle.load(f)                                                  # Loads the shape library

    print(f"Generating {NUM_EXAMPLES} examples for the dataset...")                     # Prints a status message with the number of examples
    dataset = []                                                                        # Initializes an empty list to store the generated data

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)                                       # Creates the output directory if it doesn't exist

    for _ in tqdm(range(NUM_EXAMPLES)):                                                 # Starts the main generation loop with a progress bar
        example = generate_single_example(shape_library)                                # Calls the function to generate one example
        if example is not None:                                                         # Checks if the generation was successful
            dataset.append(example)                                                     # If so, appends the example to the dataset list

    metadata = {                                                                        # Creates a dictionary to store metadata
        'num_examples'              : NUM_EXAMPLES,                                             
        'pose_distance'             : POSE_DISTANCE,                                              
        'use_random_distance'       : USE_RANDOM_DISTANCE,
        'min_distance'              : MIN_DISTANCE,
        'delta_t'                   : DELTA_T,                                                  
        'sensor_resolution'         : SENSOR_RESOLUTION,                                        
        'sensor_noise_std'          : SENSOR_NOISE_STD,                                         
        'use_los'                   : USE_LOS,
        'use_aircraft_coordinates'  : USE_AIRCRAFT_COORDINATES
    }

    full_dataset = {                                                                    # Creates a final dictionary to hold both data and metadata
        'metadata'  : metadata,                                                     
        'data'      : dataset                                                       
    }

    end_time    = time.time()                                                           # Records the ending time of the generation process
    duration    = end_time - start_time                                                 # Calculates the total duration in seconds
    minutes     = int(duration // 60)                                                   # Calculates the whole number of minutes
    seconds     = int(duration % 60)                                                    # Calculates the remaining seconds
    summary_message = f"Generated {len(dataset)} examples in {minutes}m {seconds}s."
    print(summary_message)                                                              # Prints the total time taken

    print(f"Saving dataset and metadata to {DATASET_PATH}...")                          # Prints a status message indicating the save operation
    with open(DATASET_PATH, 'wb') as f:                                                 # Opens the output file in binary write mode
        pickle.dump(full_dataset, f)                                                    # Saves the complete dataset dictionary to the file using pickle

    if VALIDATE_AFTER_GENERATION:                                                       # Checks if the validation flag is set to True
        validate_dataset(DATASET_PATH, SHAPE_LIBRARY_PATH)                              # If so, calls the validation function

