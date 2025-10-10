"""
================================================================================
generate_dataset.py
================================================================================

This script generates a dataset for a machine learning classification task.

The goal is to train a model that can infer an aircraft's shape index (its
class) based solely on a sensor point cloud of that aircraft.

To do this, the script iterates a specified number of times. In each iteration:
1.  A random aircraft shape index is chosen uniformly.
2.  A point is chosen uniformly on a circle of a fixed radius.
3.  The aircraft's heading is set to be tangent to the circle at that point.
4.  A single sensor, configured with specific noise and resolution, generates a
    point cloud at the origin.
5.  The point cloud is transformed from world coordinates to the aircraft's
    local coordinate system by applying an inverse roto-translation.
6.  The sensor's position is also transformed into the aircraft's coordinates.
7.  The tuple (point cloud, shape index, sensor position) is saved.

The final dataset is saved as a pickled list to be used in a separate model
training script.

--------------------------------------------------------------------------------
Author  : Matteo Tesori (matteotesori@gmail.com)
Date    : 2025-10-11
All wrongs reserved.
--------------------------------------------------------------------------------
"""
import numpy as np                                      
import pickle                                          
import random                                          
from pathlib import Path                                
from tqdm import tqdm                                   
import time                                             
import matplotlib.pyplot as plt                         
from matplotlib.patches import Polygon                  


from src.tns_2d.simulation.data_structures  import Aircraft, Sensor                     
from src.tns_2d.simulation.simulator        import Simulator                           

# ==============================================================================
# SECTION 1: CONFIGURATION
# ==============================================================================

# --- Dataset Settings ---
NUM_EXAMPLES                = 1000                                                      # Defines the total number of data samples to generate
POSE_RADIUS                 = 50.0                                                      # Sets the radius of the circle for placing aircraft [m]
DELTA_T                     = 1e-6                                                      # Sets a small time step for the simulation to get an instant snapshot [s]

# --- Sensor Imperfection Settings ---
SENSOR_RESOLUTION           = 2.0                                                       # Sets the number of detections per meter of an aircraft's edge.
SENSOR_NOISE_STD            = 0.1                                                       # Sets the standard deviation of Gaussian noise added to measurements [m]
USE_LOS                     = True                                                      # Toggles Line-of-Sight simulation (self-occlusions) on or off.

# --- Validation Settings ---
VALIDATE_AFTER_GENERATION   = True                                                      # Flag to run the validation plotting function after generation

# --- Path Definitions ---
ROOT_DIR           = Path(__file__).resolve().parent                         
DATA_DIR           = ROOT_DIR   / "data"                                                
OUTPUT_DIR         = ROOT_DIR   / "output"                                              
SHAPE_LIBRARY_PATH = DATA_DIR   / "shape_library.pkl"                                   
DATASET_PATH       = OUTPUT_DIR / "full_dataset.pkl"                                    

# ==============================================================================
# SECTION 2: HELPER FUNCTIONS & DATA GENERATION
# ==============================================================================

def apply_inverse_roto_translation(points_world, aircraft_position, aircraft_heading):     
    """Transforms points from the world frame to the aircraft's local frame."""
    points_world_2d = np.atleast_2d(points_world)                                       # Ensures the input points array is at least 2D.

    # Translation 
    translated_points = points_world_2d - aircraft_position                             # Step 1: Translates points by subtracting the aircraft's position.

    # Rotation matrix construction
    h               = aircraft_heading                                                  # Step 2: Prepares for rotation by getting the heading.
    cos_h, sin_h    = np.cos(h), np.sin(h)                                              # Calculates cosine and sine of the heading for the rotation matrix.
    inv_rot_matrix  = np.array([[cos_h, sin_h],                                         # Defines the 2D inverse rotation matrix.
                                 [-sin_h, cos_h]])
    
    # Inverse roto-translation
    points_aircraft = np.dot(inv_rot_matrix, translated_points.T).T                     # Applies the inverse rotation to the translated points.

    return points_aircraft.squeeze() if points_world.ndim == 1 else points_aircraft     # Returns the transformed points, squeezing to 1D if the input was 1D.

def generate_single_example(shape_library):                                       
    """Generates a single, complete data example for the dataset."""
    # STEP 1. Randomly select an aircraft shape and pose.
    num_shapes          = len(shape_library)                                            # Number of available aircraft shapes.
    random_shape_idx    = np.random.randint(0, num_shapes)                              # Selects a shape index uniformly at random.

    # STEP 2. Randomly position the aircraft 
    random_angle        = np.random.uniform(0, 2 * np.pi)                               # Selects a random angle for positioning on the circle.
    aircraft_position   = np.array([                                                    # Calculates the aircraft's (x, y) position on the circle.
                            POSE_RADIUS * np.cos(random_angle),                         # x-coordinate of the position.
                            POSE_RADIUS * np.sin(random_angle)                          # y-coordinate of the position.
    ])

    # STEP 3: Selects a random heading between 0 and 2*pi radians.
    aircraft_heading = np.random.uniform(0, 2 * np.pi)                            

    # STEP 4. Configure and run a simulation to generate the point cloud.
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

    simulator = Simulator(                                                              # Initializes the Simulator with the scenario objects.
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
    simulator.run()                                                                     # Executes the simulation.

    sensor_dataset = simulator.sensors[0].dataset                                       # Extracts the point cloud from the sensor.
    if not sensor_dataset['measurements']:                                              # Checks if any measurements were generated.
        return None                                                                     # If not, returns None to indicate failure.

    point_cloud_world_xy = sensor_dataset['measurements'][0]                            # Retrieves the point cloud from the first measurement scan.
    sensor_pos_world     = np.array([0.0, 0.0])                                         # Defines the sensor's position in the world frame (the origin).

    # STEPS 5. Express the point cloud in aircraft coordinates
    point_cloud_aircraft_xy = apply_inverse_roto_translation(                           # Transforms the point cloud to the aircraft's frame.
        point_cloud_world_xy, aircraft_position, aircraft_heading                       # Passes the point cloud and aircraft pose to the helper function.
    )

    # STEP 6. Express the sensor position in aircraft coordinates
    sensor_pos_ac = apply_inverse_roto_translation(                                     # Transforms the sensor's position to the aircraft's frame.
        sensor_pos_world, aircraft_position, aircraft_heading                           # Passes the sensor position and aircraft pose to the helper function.
    )

    return {                                                                            # Returns the final data example as a dictionary.
        'point_cloud'   : point_cloud_aircraft_xy,                                      # The point cloud in aircraft coordinates.
        'shape_idx'     : random_shape_idx,                                             # The ground truth shape index (the label).
        'sensor_pos_ac' : sensor_pos_ac                                                 # The sensor's position in aircraft coordinates (metadata).
    }

def validate_dataset(dataset_path, shape_library_path):                                 # Defines the function for visually validating the dataset.
    """Loads a generated dataset and plots random examples for visual validation."""
    print("\nStarting dataset validation...")                                           # Prints a status message to the console.
    print("Press any key to cycle through random examples.")                            # Prints instructions for the user.

    # --- Load Dataset and Shape Library ---
    with open(dataset_path, 'rb') as f:                                                 # Opens the dataset file in binary read mode.
        full_dataset = pickle.load(f)                                                   # Loads the pickled dataset from the file.
    with open(shape_library_path, 'rb') as f:                                           # Opens the shape library file in binary read mode.
        shape_library = pickle.load(f)                                                  # Loads the pickled shape library from the file.
    dataset           = full_dataset['data']                                            # Extracts the list of data examples from the loaded dictionary.
    
    # --- Randomly Sample Examples to Show ---
    num_to_show       = min(10, len(dataset))                                           # Determines how many examples to show (up to 10).
    selected_examples = random.sample(dataset, num_to_show)                             # Randomly samples the specified number of examples from the dataset.

    # --- Plotting Setup ---
    fig, ax = plt.subplots(figsize=(8,8))                                               # Creates a new figure and axes for plotting with a specific size.
    fig.patch.set_facecolor((38/255, 55/255, 59/255))                                   # Sets the background color of the figure to a dark gray-blue.
    ax.set_facecolor((38/255, 55/255, 59/255))                                          # Sets the background color of the plot area to match the figure.
    ax.tick_params(colors='white')                                                      # Sets the color of the tick marks and labels to white.
    for spine in ax.spines.values():                                                    # Loops through all the spines (the plot borders).
        spine.set_color('white')                                                        # Sets the color of each spine to white.

    for i, example in enumerate(selected_examples):                                     # Loops through the randomly selected examples.
        ax.cla()                                                                        # Clears the axes to prepare for the next plot.

        # --- Plot Aircraft Shape ---
        shape_idx       = example['shape_idx']                                          # Unpacks the shape index from the example dictionary.
        sensor_pos      = example['sensor_pos_ac']                                      # Unpacks the sensor position from the example dictionary.
        shape_matrix    = shape_library[shape_idx]['shapeMatrix']                       # Retrieves the ground truth shape data from the library using the index.
        vertices = shape_matrix[:2, :].T                                                # Extracts the vertices of the aircraft shape.
        polygon  = Polygon(vertices, facecolor = 'white', alpha = 0.1, zorder = 1)      # Creates a transparent white polygon for the aircraft body.
        ax.add_patch(polygon)                                                           # Adds the polygon patch to the plot.

        for edge_idx in range(shape_matrix.shape[1]):                                   # Loops through each edge of the aircraft shape.
            start_point = shape_matrix[:2, edge_idx]                                    # Gets the start point of the current edge.
            end_point   = shape_matrix[2:4, edge_idx]                                   # Gets the end point of the current edge.
            ax.plot([start_point[0], end_point[0]], 
                    [start_point[1], end_point[1]], 'r-', 
                    lw = 2, label = 'True Shape' if edge_idx == 0 else "", zorder=2)    # Plots the edge in red.

        # --- Plot Sensor Data ---
        pc              = example['point_cloud']                                        # Unpacks the point cloud from the example dictionary.
        if pc.size > 0:                                                                 # Checks if the point cloud is not empty.
            ax.scatter(pc[:,0], pc[:,1], c = 'yellow', 
                       marker = 's', s = 10, label = 'Point Cloud', 
                       alpha = 1.0, zorder = 10)                                        # Plots the point cloud as small yellow squares.

        # --- Plot Sensor Position ---
        ax.plot(sensor_pos[0], sensor_pos[1], 'co', 
                markersize = 10, label = 'Sensor Position')                             # Plots the sensor's position as a cyan circle.

        # --- Final Plot Adjustments ---
        title_text = f"Validation Example {i+1}/{num_to_show} (Shape Index: {shape_idx})"
        ax.set_title(title_text, color='white', fontname='monospace')                   # Sets the plot title.
        ax.set_xlabel("X coordinate (Aircraft Frame) [m]", 
                      color='white', fontname='monospace')                              # Sets the x-axis label.
        ax.set_ylabel("Y coordinate (Aircraft Frame) [m]", 
                      color='white', fontname='monospace')                              # Sets the y-axis label.

        legend = ax.legend()                                                            # Gets the legend object.
        for text in legend.get_texts():                                                 # Loops through the text elements in the legend.
            text.set_color('white')                                                     # Sets the legend text color to white.
        legend.get_frame().set_facecolor((48/255, 65/255, 69/255))                      # Sets the legend background color.
        legend.get_frame().set_edgecolor('white')                                       # Sets the legend border color.

        ax.axis('equal')                                                                # Ensures the x and y axes have the same scale.
        ax.set_xlim([-60, 60])                                                          # Sets the x-axis limits.
        ax.set_ylim([-60, 60])                                                          # Sets the y-axis limits.
        plt.pause(0.1)                                                                  # Pauses briefly to ensure the plot is drawn.

        # --- Wait for User Input to Proceed ---
        if not plt.waitforbuttonpress():                                                # Waits for a key press from the user.
            print("Plot window closed. Exiting validation.")                            # If the window is closed, prints a message.
            break                                                                       # Breaks the loop to exit validation.

    plt.close()                                                                         # Closes the plot window after the loop finishes.
    print("Validation complete.")                                                       # Prints a completion message.

# ==============================================================================
# SECTION 3: EXECUTION
# ==============================================================================

if __name__ == "__main__":                                                              # Standard Python entry point.
    start_time = time.time()                                                            # Records the starting time of the script.

    print("Loading shape library...")                                                   # Prints a status message.
    with open(SHAPE_LIBRARY_PATH, 'rb') as f:                                           # Opens the shape library file.
        shape_library = pickle.load(f)                                                  # Loads the shape library.

    print(f"Generating {NUM_EXAMPLES} examples for the dataset...")                     # Prints a status message with the number of examples.
    dataset = []                                                                        # Initializes an empty list to store the generated data.

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)                                       # Creates the output directory if it doesn't exist.

    for _ in tqdm(range(NUM_EXAMPLES)):                                                 # Starts the main generation loop with a progress bar.
        example = generate_single_example(shape_library)                                # Calls the function to generate one example.
        if example is not None:                                                         # Checks if the generation was successful.
            dataset.append(example)                                                     # If so, appends the example to the dataset list.

    metadata = {                                                                        # Creates a dictionary to store metadata.
        'num_examples'      : NUM_EXAMPLES,                                             
        'pose_radius'       : POSE_RADIUS,                                              
        'delta_t'           : DELTA_T,                                                  
        'sensor_resolution' : SENSOR_RESOLUTION,                                        
        'sensor_noise_std'  : SENSOR_NOISE_STD,                                         
        'use_los'           : USE_LOS                                                   
    }

    full_dataset = {                                                                    # Creates a final dictionary to hold both data and metadata.
        'metadata'  : metadata,                                                     
        'data'      : dataset                                                       
    }

    end_time    = time.time()                                                           # Records the ending time of the generation process.
    duration    = end_time - start_time                                                 # Calculates the total duration in seconds.
    minutes     = int(duration // 60)                                                   # Calculates the whole number of minutes.
    seconds     = int(duration % 60)                                                    # Calculates the remaining seconds.
    summary_message = f"Generated {len(dataset)} examples in {minutes}m {seconds}s."
    print(summary_message)                                                              # Prints the total time taken.

    print(f"Saving dataset and metadata to {DATASET_PATH}...")                          # Prints a status message indicating the save operation.
    with open(DATASET_PATH, 'wb') as f:                                                 # Opens the output file in binary write mode.
        pickle.dump(full_dataset, f)                                                    # Saves the complete dataset dictionary to the file using pickle.

    if VALIDATE_AFTER_GENERATION:                                                       # Checks if the validation flag is set to True.
        validate_dataset(DATASET_PATH, SHAPE_LIBRARY_PATH)                              # If so, calls the validation function.

