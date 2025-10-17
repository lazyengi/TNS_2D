"""
================================================================================
simulator.py
================================================================================

This module contains the core logic for the TNS-2D simulation project.

It includes the standalone functions for generating trajectories, dropouts,
and sensor measurements (including clutter), as well as the main `Simulator`
class, which acts as the central orchestrator for the entire data generation
process.

The design philosophy is to keep the core algorithms as pure, reusable
functions and to use the `Simulator` class to manage the state and flow of
the simulation from start to finish.

--------------------------------------------------------------------------------
Author  : Matteo Tesori (matteotesori@gmail.com)
Date    : 2025-10-10
All wrongs reserved.
--------------------------------------------------------------------------------
"""

import numpy as np
import pandas as pd
import time
from .data_structures import ExtendedObject, Sensor

# ==============================================================================
# SECTION 1: SIMULATOR CLASS
# ==============================================================================

class Simulator:
    """The central orchestrator for the TNS-2D simulation."""
    def __init__(self, extended_objects: list[ExtendedObject], sensors: list[Sensor], shape_library: dict, 
                 global_dt: float, spine_length: float, max_duration: float,
                 use_dropouts: bool = True, use_los: bool = True, use_oor: bool = True,
                 verbose: bool = True):
        """
        Initializes the Simulator with the complete scenario configuration.

        Args:
            extended_objects (list[ExtendedObject]) : List of configured ExtendedObject objects.
            sensors (list[Sensor])                  : List of configured Sensor objects.
            shape_library (dict)                    : Library mapping shape indices to shape data.
            global_dt (float)                       : Master time step for the simulation grid [s].
            spine_length (float)                    : Half-length of the simulation area.
            max_duration (float)                    : Maximum allowed duration for the simulation [s]
            use_dropouts (bool, optional)           : If True, simulates sensor dropouts. 
            use_los (bool, optional)                : If True, simulates Line-of-Sight blockages. 
            use_oor (bool, optional)                : If True, simulates Out-of-Range conditions.
            verbose (bool, optional)                : If True, prints progress information. 
        """
        # --- Core Simulation Objects ---
        self.extended_objects = extended_objects
        self.sensors          = sensors
        self.shape_library    = shape_library

        # --- Global Simulation Parameters ---
        self.global_dt      = global_dt
        self.spine_length   = spine_length
        self.max_duration   = max_duration

        # --- Debugging Flags ---
        self.use_dropouts   = use_dropouts
        self.use_los        = use_los
        self.use_oor        = use_oor
        self.verbose        = verbose

        # --- Internal State ---
        self.time_grid      = None

    def run(self):
        """Executes the full simulation pipeline."""
        t_start_sim = time.time()                                                                       # Start timer
        if self.verbose:
            print("Simulation started...")

        if self.verbose:
            print("Step 1: Generating raw trajectories...")
        for eo in self.extended_objects:
            _generate_raw_trajectory(eo, self.max_duration)                                             # Generate trajectories up to the max possible duration

        if self.verbose:
            print("Step 2: Aligning trajectories to master time grid...")
        self._align_all_tracks()                                                                        # Align all tracks to the master time grid

        if self.verbose:
            print("Step 3: Generating sensor datasets...")
        self._generate_all_datasets()                                                                   # Generate sensor datasets         

        t_end_sim   = time.time()                                                                       # End timer
        duration    = t_end_sim - t_start_sim                                                           # Calculate simulation duration
        if self.verbose:
            print(f"Simulation complete in {duration:.2f} seconds.")

    def _align_all_tracks(self):
        """Creates the master time grid based on actual events and aligns all trajectories."""
        # --- Step 1: Mask Raw Tracks and Find Terminal Exit Times ---
        terminal_exit_times = []                                                                        # Holds the last in-bounds time for each extended object
        for eo in self.extended_objects:
            # Create a mask for any data points that have left the simulation area
            mask                    = (np.abs(eo.raw_track[0, :]) > self.spine_length) | \
                                       (np.abs(eo.raw_track[1, :]) > self.spine_length)
            eo.raw_track[:, mask]   = np.nan                                                            # Nullify the out-of-bounds data points
            
            valid_times = eo.raw_time[~np.isnan(eo.raw_track[0,:])]                                     # Get the timestamps for all in-bounds points       
            if valid_times.size > 0:
                terminal_exit_times.append(np.max(valid_times))                                         # Record the extended object's final valid timestamp

        # --- Step 2: Determine True Simulation Duration ---
        t_start             = min((np.nanmin(eo.raw_time) for eo in self.extended_objects 
                                   if eo.raw_time is not None and eo.raw_time.size > 0), default = 0)   # Find the earliest start time across all extended objects
        true_event_duration = max(terminal_exit_times) if terminal_exit_times else t_start              # Find the latest exit time across all extended objects
        final_t_end         = min(true_event_duration, self.max_duration)                               # Clamp to the maximum allowed duration

        # --- Step 3: Create the Master Time Grid ---
        self.time_grid = np.arange(t_start, final_t_end + self.global_dt, self.global_dt)               # Create the master time grid
        K              = len(self.time_grid)                                                            # Total number of time steps

        # --- Step 4: Align Each Extended Object's Track to the Grid ---
        for eo in self.extended_objects:
            eo.aligned_track    = np.full((4, K), np.nan)                                               # Initialize aligned track with NaNs
            valid_mask          = ~np.isnan(eo.raw_track[0, :])                                         # Find all valid (in-bounds) points in the raw track
            if not np.any(valid_mask): continue                                                         # Skip this extended object if it has no valid data
            
            raw_times_valid                     = eo.raw_time[valid_mask]                               # Get the timestamps of the valid raw data
            grid_indices                        = np.searchsorted(self.time_grid, 
                                                                raw_times_valid, side = 'left')         # Find where each raw timestamp fits into the master grid
            grid_indices                        = np.clip(grid_indices, 0, K - 1)                       # Prevent indices from going out of bounds
            unique_indices, unique_raw_map      = np.unique(grid_indices, return_index = True)          # Get unique grid indices and their raw data indices
            eo.aligned_track[:, unique_indices] = eo.raw_track[:, valid_mask][:, unique_raw_map]        # Place the valid raw data onto the master grid

            # --- Step 4a: Housekeeping ---
            df = pd.DataFrame(eo.aligned_track.T)                                                       # Use a DataFrame for convenient interpolation  
            df.interpolate(method = 'linear', limit = 1, axis = 0, inplace = True)                      # Linearly interpolate gaps (up to 1 consecutive NaN)
            eo.aligned_track = df.to_numpy().T                                                          # Convert back to NumPy array
            
            # Normalize the heading angle to ensure it stays within the [-pi, pi) range
            heading_row                 = eo.aligned_track[2, :]                                        # Extract the heading row   
            valid_headings              = ~np.isnan(heading_row)                                        # Identify valid (non-NaN) headings
            heading_row[valid_headings] = np.arctan2(np.sin(heading_row[valid_headings]), 
                                                     np.cos(heading_row[valid_headings]))               # Normalize valid headings

    def _generate_all_datasets(self):
        """Generates datasets for all sensors in the simulation."""
        for sensor in self.sensors:
            # --- Robustly Calculate Sample Indices to Avoid Floating Point Errors ---
            start_index    = int(round(sensor.t_offset / self.global_dt))                               # Starting index in the global time grid
            step_index     = int(round(sensor.dt_sample / self.global_dt))                              # Step size in indices for this sensor's sampling period
            end_index      = len(self.time_grid)                                                        # End index (exclusive)
            sample_indices = np.arange(start_index, end_index, step_index)                              # All sample indices for this sensor
            num_samples    = len(sample_indices)                                                        # Total number of samples for this sensor

            # --- Generate Dropout Sequence ---
            if self.use_dropouts:                                                                       
                sensor.dropout_sequence = _generate_dropout_sequence(sensor, num_samples)               # Generate dropout sequence if enabled
            else:
                sensor.dropout_sequence = np.ones(num_samples)                                          # Otherwise, assume always active

            # --- Generate Measurements for Each Sample Time ---
            measurements, labels, times = [], [], []                                                    # Lists to hold measurements, labels, and timestamps
            for i, k in enumerate(sample_indices):
                if k >= len(self.time_grid): continue                                                   # Safety check
                
                dropout_status = sensor.dropout_sequence[i]                                             # Get the dropout status for this specific sample event
                
                if dropout_status > 0.5:                                                                # Only generate measurements if the sensor is active 
                    # --- Generate Clutter Points ---                                                  
                    clutter_points, clutter_labels  = _generate_clutter(sensor)                         # Generate clutter points for this scan
                    all_points, all_labels          = [clutter_points], [clutter_labels]                # Start with clutter points and labels
                    
                    # --- Generate True Detections from All Extended Objects ---
                    for extended_object in self.extended_objects:
                        if not np.isnan(extended_object.aligned_track[0, k]):                           # Only consider extended objects that are in-bounds at this time
                            extended_object_pose  = extended_object.aligned_track[:3, k]                # Extract the extended object's pose at this time
                            extended_object_id    = extended_object.identifier                          # Extract extended object identifier
                            extended_object_shape = self.shape_library[extended_object.shape_idx]       # Extract extended object shape information
                            
                            if sensor.generation_mode == "contour":                                     # Generate detections on the contour      
                                det_points, det_labels = _generate_detections_on_contour(
                                    extended_object_pose, extended_object_id, sensor, 
                                    extended_object_shape, dropout_status,
                                    self.use_los, self.use_oor
                                )
                            elif sensor.generation_mode == "surface":                                   # Generate detections on the surface    
                                det_points, det_labels = _generate_detections_on_surface(
                                    extended_object_pose, extended_object_id, sensor, 
                                    extended_object_shape, dropout_status,
                                    self.use_los, self.use_oor
                                )
                            elif sensor.generation_mode == "center":                                    # Generate detection at the center   
                                det_points, det_labels = _generate_detections_on_center(
                                    extended_object_pose, extended_object_id, sensor, 
                                    extended_object_shape, dropout_status,
                                    self.use_los, self.use_oor
                                )
                            else:
                                print(f"Warning: Unknown generation_mode '{sensor.generation_mode}' "
                                      f"for sensor {sensor.identifier}. No detections will be generated.")
                                det_points, det_labels = np.empty((0, 2)), np.empty(0, dtype=int)

                            if det_points.size > 0:                                                     # Check if there are detection points
                                all_points.append(det_points)                                           # Append detection points
                                all_labels.append(det_labels)                                           # Append detection labels      
                    
                    if any(p.size > 0 for p in all_points):                                             # Only store if there are any points at all
                        measurements.append(np.vstack(all_points))                                      # Stack all points into a single list
                        labels.append(np.concatenate(all_labels))                                       # Stack all labels into a single list
                        times.append(self.time_grid[k])                                                 # Stack all timestamps into a single list

            sensor.dataset = {'times': np.array(times), 
                              'measurements': measurements, 
                              'labels': labels}                                                         # Store the final dataset in the sensor object

# ==============================================================================
# SECTION 2: CORE ALGORITHMS 
# ==============================================================================

def _generate_raw_trajectory(extended_object: ExtendedObject, t_end: float):
    """Generates a raw, high-resolution trajectory for a single extended object."""
    # --- Generate Time Vector ---
    dt          = extended_object.dt
    time_vector = np.arange(0, t_end + dt, dt)                                                          # Time vector for raw trajectory generation

    # --- Generate Control Inputs ---
    u_lon_vals  = extended_object.long_vals
    u_lon_times = extended_object.long_times
    u_cen_vals  = extended_object.centr_vals
    u_cen_times = extended_object.centr_times
    u_signal    = _convert_control_inputs(time_vector, u_lon_vals, 
                                          u_lon_times, u_cen_vals, u_cen_times)                         # Convert event-based controls to continuous signal

    # --- Initialize State and Storage ---
    x0 = np.array([extended_object.initial_position[0], extended_object.initial_position[1], 
                   extended_object.initial_heading, extended_object.initial_speed])                     # Initial state vector [x, y, heading, speed]

    # --- Integrate Dynamics Using ZOH ---
    K       = len(time_vector)                                                                          # Total number of time steps
    x       = np.zeros((4, K))                                                                          # State storage
    x[:, 0] = x0                                                                                        # Set initial state
    x[2, 0] = np.arctan2(np.sin(x[2, 0]), np.cos(x[2, 0]))                                              # Wrap initial heading to [-pi, pi)

    for k in range(K - 1):
        x[:, k + 1], _ = _zoh_integrator(x[:, k], u_signal[:, k], dt)                                   # Perform ZOH integration step  

    extended_object.raw_time  = time_vector + extended_object.t_init                                    # Store the raw time vector, adjusted for initial time
    extended_object.raw_track = x                                                                       # Store the raw trajectory

def _zoh_integrator(x_k, u_k, dt):
    """Performs one step of Zero-Order Hold (ZOH) integration."""
    # --- Unpack State and Control Vectors ---
    _, _, h_k, v_k = x_k                                                                                # State  vector: position (x,y), heading, speed
    a_lon, a_cen   = u_k                                                                                # Control vector: longitudinal, centripetal accelerations

    # --- Compute Next State Using ZOH ---
    v_plus  = v_k + dt * a_lon                                                                          # Update speed
    v_avg   = (v_k + v_plus) / 2.0                                                                      # Trapezoidal integration for speed

    # --- Handle Heading Wrap-around and Avoid Division by Zero ---
    h_dot   = a_cen / v_k if np.abs(v_k) > 1e-6 else 0.0                                                # Update heading rate (avoid division by zero)
    h_plus  = h_k + dt * h_dot                                                                          # Update heading
    h_plus  = np.arctan2(np.sin(h_plus), np.cos(h_plus))                                                # Wrap heading to [-pi, pi)
    h_avg   = (h_k + h_plus) / 2.0                                                                      # Trapezoidal integration for heading

    # --- Update Position Using Unicycle Model ---
    x_plus  = x_k[0] + dt * v_avg * np.cos(h_avg)                                                       # Update x position
    y_plus  = x_k[1] + dt * v_avg * np.sin(h_avg)                                                       # Update y position
    
    # --- Pack Next State and State Derivative ---
    x_k_plus    = np.array([x_plus, y_plus, h_plus, v_plus])                                            # Next state vector
    dot_x_k     = np.array([v_k * np.cos(h_k), v_k * np.sin(h_k), h_dot, a_lon])                        # State derivative vector

    return x_k_plus, dot_x_k

def _convert_control_inputs(time_vector, u_lon_vals, u_lon_times, u_cen_vals, u_cen_times):
    """Converts event-based control inputs into a continuous signal."""
    u_lon = _zoh_interpolator(time_vector, u_lon_times, u_lon_vals)                                     # Interpolate longitudinal acceleration   
    u_cen = _zoh_interpolator(time_vector, u_cen_times, u_cen_vals)                                     # Interpolate centripetal acceleration    
    return np.vstack([u_lon, u_cen])

def _zoh_interpolator(t_query, t_steps, u_values):
    """Performs Zero-Order Hold interpolation."""
    indices = np.searchsorted(t_steps, t_query, side='right') - 1                                       # Find indices for ZOH
    indices = np.clip(indices, 0, len(u_values) - 1)                                                    # Clamp indices to valid range
    return u_values[indices]

def _generate_dropout_sequence(sensor: Sensor, num_samples: int, seed: int = None):
    """Generates an event-based dropout sequence for a single sensor."""
    rng = np.random.default_rng(seed)                                                                   # Initialize random number generator
    K   = num_samples                                                                                   # Total number of samples
    
    if sensor.alpha is None:
        # --- Generate Dropout Sequence Based on Bernoulli Model ---
        p = [1 - sensor.pd, sensor.pd]
        return rng.choice([0.0, 1.0], size = K, p = p)                                                  # Dropout sequence with Bernoulli (drop prob = 1 - pd)
    else:
        # --- Generate Dropout Sequence Based on Markov Chain Model ---
        expression  = (1 - 2*sensor.pd + sensor.pd*sensor.alpha) / (1 - sensor.pd)                      # Intermediate expression for beta calculation
        beta        = expression if (1-sensor.pd) > 1e-9 else 0.0                                       # Probability of recovery from dropout 
        drops       = np.zeros(K)                                                                       # Initialize dropout sequence
        if K > 0:
            drops[0] = 1.0 if rng.random() < sensor.pd else 0.0                                         # Initialize first drop state based on pd
            for k in range(K - 1):
                if drops[k] == 1.0:                                                                     # If currently active...
                    drops[k + 1] = 1.0 if rng.random() < sensor.alpha else 0.0                          # ....stay active with prob alpha, else dropout
                else:                                                                                   # Otherwise...
                    drops[k + 1] = 0.0 if rng.random() < beta else 1.0                                  # ...stay dropout with prob beta, else recover
        return drops

def _generate_clutter(sensor: Sensor):
    """Generates a set of clutter points for a single sensor scan."""
    # --- Calculate Expected Number of Clutter Points ---
    radial_component = 0.5 * (sensor.max_range**2 - sensor.min_range**2)                                # Radial component of the FoV area
    fov_area         = radial_component * (sensor.max_bearing - sensor.min_bearing)                     # Total FoV area in m^2
    lambda_clutter   = sensor.clutter_rate * fov_area                                                   # Expected number of clutter points
    n_clutter        = np.random.poisson(lambda_clutter)                                                # Actual number of clutter points (Poisson distributed)

    # --- Generate Clutter Points Uniformly Within the Sensor's FoV ---
    if n_clutter == 0: return np.empty((0, 2)), np.empty(0, dtype=int)                                  # Return empty arrays if no clutter points    

    range_sq_diff       = sensor.max_range**2 - sensor.min_range**2                                     # Difference in squared ranges
    r_sq                = range_sq_diff * np.random.rand(n_clutter) + sensor.min_range**2               # Sample squared ranges uniformly
    r                   = np.sqrt(r_sq)                                                                 # Convert to actual ranges
    angular_span        = sensor.max_bearing - sensor.min_bearing                                       # Total angular span of the FoV
    theta               = angular_span * np.random.rand(n_clutter) + sensor.min_bearing                 # Sample bearings uniformly
    x_sens              = r * np.cos(theta)                                                             # Convert to Cartesian x-coordinates in sensor frame
    y_sens              = r * np.sin(theta)                                                             # Convert to Cartesian y-coordinates in sensor frame

    sensor_pose         = (sensor.position[0], sensor.position[1], sensor.heading)                      # Sensor pose for transformation
    points_sensor_frame = np.vstack([x_sens, y_sens]).T                                                 # Clutter points in sensor frame
    points_world        = _roto_translate(points_sensor_frame, sensor_pose)                             # Transform clutter points to world frame
    labels              = np.full(n_clutter, -1, dtype=int)                                             # Clutter point labels (-1 indicates clutter)

    return points_world, labels

# ==============================================================================
# SECTION 3: CONTOUR DETECTION GENERATION FUNCTIONS
# ==============================================================================

def _generate_detections_on_contour(extended_object_pose: np.ndarray, extended_object_id: int, 
                         sensor: Sensor, extended_object_shape: dict, 
                         dropout_status: float, use_los: bool, use_oor: bool):
    """Generates a point cloud of true detections on an extended object's contour."""
    # --- Unpack Extended Object Shape Information ---
    contourShapeMatrix  = extended_object_shape['contourShapeMatrix']                                   # Shape matrix defining the extended object's geometry
    los_table           = extended_object_shape.get('LoSTable')                                         # Line-of-Sight table
    n_edges             = contourShapeMatrix.shape[1]                                                   # Number of edges in the shape
    
    # --- Compute Edge Positions in World Frame ---
    edge_starts_local   = contourShapeMatrix[:2, :].T                                                   # Edge start points in extended object frame
    edge_ends_local     = contourShapeMatrix[2:4, :].T                                                  # Edge end points in extended object frame
    edge_lengths        = contourShapeMatrix[4, :]                                                      # Edge lengths

    # --- Compute Flags and Detection Rates ---
    sensor_pose_vec     = np.array([sensor.position[0], sensor.position[1], sensor.heading])            # Sensor pose in world frame
    edge_starts_world   = _roto_translate(edge_starts_local, extended_object_pose)                      # Edge start points in world frame
    edge_ends_world     = _roto_translate(edge_ends_local, extended_object_pose)                        # Edge end points in world frame
    edge_centers_world  = (edge_starts_world + edge_ends_world) / 2                                     # Edge center points in world frame
    edge_centers_sensor = _roto_translate(edge_centers_world, sensor_pose_vec, inverse=True)            # Edge center points in sensor frame

    if use_oor:                                                                                         # If OoR is enabled...
        oor_flags = _compute_oor_flags(edge_centers_sensor, sensor.min_range, 
                                        sensor.max_range, sensor.min_bearing, 
                                        sensor.max_bearing)                                             # ...compute Out-of-Range flags
    else: oor_flags = np.ones(n_edges, dtype = int)                                                     # otherwise assume all edges are in range
    
    if use_los and los_table is not None:                                                               # If LoS is enabled and the table is provided...                         
        los_flags = _compute_los_flags(extended_object_pose, sensor_pose_vec, los_table)                # ...compute Line-of-Sight flags
    else: los_flags = np.ones(n_edges, dtype = int)                                                     # otherwise assume all edges are in LoS

    lambda_i            = sensor.resolution * edge_lengths * los_flags * oor_flags * dropout_status     # Expected detections per edge
    total_lambda        = np.sum(lambda_i)                                                              # Total expected detections across all edges

    # --- Sample Detections Based on Poisson Process ---
    if total_lambda == 0: return np.empty((0, 2)), np.empty(0, dtype = int)                             # Return empty arrays if no detections possible

    n_measurements = np.random.poisson(total_lambda)                                                    # Actual number of detections (Poisson distributed)
    if n_measurements == 0: return np.empty((0, 2)), np.empty(0, dtype = int)                           # Return empty arrays if no detections sampled

    edge_probs      = lambda_i / total_lambda                                                           # Probability of selecting each edge
    selected_edges  = np.random.choice(n_edges, size = n_measurements, p = edge_probs)                  # Sample edges based on probabilities
    points          = [                                                                                 # Sample points along each selected edges              
                        _sample_measurement_on_contour(edge_starts_world[i], 
                                                       edge_ends_world[i], sensor.sigma_v)          
                        for i in selected_edges
                        ]    
    labels          = np.full(n_measurements, extended_object_id, dtype = int)                          # Labels for each detection point (extended object ID)

    return np.array(points), labels

def _compute_los_flags(extended_object_pose, sensor_pose, los_table):
    """Computes Line-of-Sight flags using a pre-computed visibility table."""
    # --- Find Grid Point Closest to Sensor  ---
    sensor_pos_ac   = _roto_translate(sensor_pose[:2].reshape(1, -1), 
                                      extended_object_pose, inverse = True)[0]                          # Sensor position in extended object frame
    # --- Find Closest Point in the Flat Grid ---
    diffs           = los_table['coords'] - sensor_pos_ac                                               # Differences to all grid points (now a 2D array)
    dists           = np.linalg.norm(diffs, axis = 1)                                                   # Euclidean distances to all grid points (now a 1D array)
    idx             = np.argmin(dists)                                                                  # Index of the closest grid point in the flat list
    return los_table['flags'][idx, :].astype(int)                                                       # Return LoS flags for all edges from the closest point

def _compute_oor_flags(feature_centers_sensor, min_r, max_r, min_b, max_b):
    """Computes Out-of-Range flags based on sensor FoV for a set of feature centers."""
    ranges   = np.linalg.norm(feature_centers_sensor, axis = 1)                                         # Ranges to feature centers
    bearings = np.arctan2(feature_centers_sensor[:, 1], feature_centers_sensor[:, 0])                   # Bearings to feature centers
    return ((ranges >= min_r) & (ranges <= max_r) & \
            (bearings >= min_b) & (bearings <= max_b)).astype(int)                                      # Return OOR flags for all features

def _sample_measurement_on_contour(edge_start: np.ndarray, edge_end: np.ndarray, sigma_v: float):
    """Samples a single noisy measurement point along a specified edge."""
    u               = np.random.uniform()                                                               # Sample a uniform [0, 1]
    point_on_edge   = (1 - u) * edge_start + u * edge_end                                               # Interpolate point along the edge 
    noise           = np.random.normal(scale = sigma_v, size = 2)                                       # Add Gaussian noise to the point
    return point_on_edge + noise

# ==============================================================================
# SECTION 4: SURFACE DETECTION GENERATION FUNCTIONS
# ==============================================================================

def _generate_detections_on_surface(extended_object_pose: np.ndarray, extended_object_id: int, 
                                    sensor: Sensor, extended_object_shape: dict, 
                                    dropout_status: float, use_los: bool, use_oor: bool):
    """Generates a point cloud of true detections on an extended object's surface."""
    # (1) Unpack shape information
    surfaceShapeMatrix = extended_object_shape.get('surfaceShapeMatrix')
    if surfaceShapeMatrix is None or surfaceShapeMatrix.size == 0:
        return np.empty((0, 2)), np.empty(0, dtype=int)
    
    n_triangles  = surfaceShapeMatrix.shape[1]
    tri_areas    = surfaceShapeMatrix[6, :]
    
    # (2) Compute triangle positions in world coordinates
    v0_local = surfaceShapeMatrix[0:2, :].T
    v1_local = surfaceShapeMatrix[2:4, :].T
    v2_local = surfaceShapeMatrix[4:6, :].T

    v0_world = _roto_translate(v0_local, extended_object_pose)
    v1_world = _roto_translate(v1_local, extended_object_pose)
    v2_world = _roto_translate(v2_local, extended_object_pose)

    # (3) Compute triangle centers in sensor coordinates
    sensor_pose_vec      = np.array([sensor.position[0], sensor.position[1], sensor.heading])
    tri_centers_world    = (v0_world + v1_world + v2_world) / 3.0
    tri_centers_sensor   = _roto_translate(tri_centers_world, sensor_pose_vec, inverse=True)
    
    # (4) Compute OOR flags for each triangle
    if use_oor:
        oor_flags = _compute_oor_flags(tri_centers_sensor, sensor.min_range, 
                                       sensor.max_range, sensor.min_bearing, 
                                       sensor.max_bearing)
    else:
        oor_flags = np.ones(n_triangles, dtype=int)

    # (5) Compute expected detections per triangle (ignoring LoS)
    lambda_i     = sensor.resolution * tri_areas * oor_flags * dropout_status
    total_lambda = np.sum(lambda_i)

    # (6) Use Poisson sampling on the expected detections
    if total_lambda == 0:
        return np.empty((0, 2)), np.empty(0, dtype=int)

    n_measurements = np.random.poisson(total_lambda)
    if n_measurements == 0:
        return np.empty((0, 2)), np.empty(0, dtype=int)

    triangle_probs   = lambda_i / total_lambda
    selected_tris    = np.random.choice(n_triangles, size=n_measurements, p=triangle_probs)
    
    # --- Sample points within each selected triangle ---
    points = [
        _sample_measurement_on_triangle(v0_world[i], v1_world[i], v2_world[i], sensor.sigma_v)
        for i in selected_tris
    ]
    labels = np.full(n_measurements, extended_object_id, dtype=int)

    return np.array(points), labels

def _sample_measurement_on_triangle(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray, sigma_v: float):
    """Samples a single noisy measurement point uniformly within a specified triangle."""
    # --- Generate two uniform random numbers for barycentric coordinates ---
    u1, u2 = np.random.uniform(size=2)

    # --- Sample a point uniformly within the triangle using Kraemer's method ---
    # This ensures a statistically uniform distribution of points.
    point_on_triangle = (1 - np.sqrt(u1)) * v0 + \
                        (np.sqrt(u1) * (1 - u2)) * v1 + \
                        (np.sqrt(u1) * u2) * v2

    # --- Add Gaussian measurement noise ---
    noise = np.random.normal(scale=sigma_v, size=2)
    return point_on_triangle + noise

# ==============================================================================
# SECTION 5: CENTER DETECTION GENERATION FUNCTIONS
# ==============================================================================

def _generate_detections_on_center(extended_object_pose: np.ndarray, extended_object_id: int, 
                                   sensor: Sensor, extended_object_shape: dict, 
                                   dropout_status: float, use_los: bool, use_oor: bool):
    """Generates a single point detection on an extended object's center."""
    # --- Unpack Shape Info and Compute Center of Geometry ---
    contourShapeMatrix  = extended_object_shape['contourShapeMatrix']
    edge_starts_local   = contourShapeMatrix[:2, :].T
    edge_ends_local     = contourShapeMatrix[2:4, :].T
    edge_lengths        = contourShapeMatrix[4, :]
    total_length        = np.sum(edge_lengths)

    if total_length == 0:
        return np.empty((0, 2)), np.empty(0, dtype=int)

    weights             = edge_lengths / total_length
    edge_centers_local  = (edge_starts_local + edge_ends_local) / 2.0
    center_local        = np.sum(edge_centers_local * weights[:, np.newaxis], axis=0)

    # --- Transform Center to World and Sensor Frames ---
    sensor_pose_vec = np.array([sensor.position[0], sensor.position[1], sensor.heading])
    center_world    = _roto_translate(center_local.reshape(1, -1), extended_object_pose)[0]
    center_sensor   = _roto_translate(center_world.reshape(1, -1), sensor_pose_vec, inverse=True)

    # --- Compute OOR Flag for the Center Point ---
    if use_oor:
        oor_flag = _compute_oor_flags(center_sensor, sensor.min_range, 
                                      sensor.max_range, sensor.min_bearing, 
                                      sensor.max_bearing)[0]
    else:
        oor_flag = 1

    # --- Generate a single measurement if conditions are met ---
    if oor_flag * dropout_status == 1:
        noise = np.random.normal(scale=sensor.sigma_v, size=2)
        point = center_world + noise
        return np.array([point]), np.array([extended_object_id], dtype=int)
    else:
        return np.empty((0, 2)), np.empty(0, dtype = int)

# ==============================================================================
# SECTION 6: HELPER FUNCTIONS
# ==============================================================================

def _roto_translate(points, pose, inverse = False):
    """Applies a 2D roto-translation to a set of points."""
    # --- Unpack Pose ---
    x, y, h     = pose                                                                                  # Pose components: position (x,y), heading           
    
    # --- Define Rotation Matrix ---
    c, s        = np.cos(h), np.sin(h)                                                                  # Precompute cosine and sine of heading
    R           = np.array([[c, -s], [s, c]])                                                           # 2D rotation matrix

    # --- Apply Transformation ---
    points_T    = points.T
    if inverse:
        return (R.T @ (points_T - np.array([[x], [y]]))).T                                              # Inverse transformation    
    else:
        return (R @ points_T).T + np.array([x, y])                                                      # Forward transformation
