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
from .data_structures import Aircraft, Sensor

# ==============================================================================
# SECTION 1: THE SIMULATOR CLASS
# ==============================================================================

class Simulator:
    """The central orchestrator for the TNS-2D simulation."""
    def __init__(self, aircraft: list[Aircraft], sensors: list[Sensor], shape_library: dict, 
                 global_dt: float, spine_length: float, max_duration: float,
                 use_dropouts: bool = True, use_los: bool = True, use_oor: bool = True,
                 verbose: bool = True):
        """
        Initializes the Simulator with the complete scenario configuration.

        Args:
            aircraft (list[Aircraft])       : List of configured Aircraft objects.
            sensors (list[Sensor])          : List of configured Sensor objects.
            shape_library (dict)            : Library mapping shape indices to shape data.
            global_dt (float)               : Master time step for the simulation grid [s].
            spine_length (float)            : Half-length of the simulation area.
            max_duration (float)            : Maximum allowed duration for the simulation [s]
            use_dropouts (bool, optional)   : If True, simulates sensor dropouts. 
            use_los (bool, optional)        : If True, simulates Line-of-Sight blockages. 
            use_oor (bool, optional)        : If True, simulates Out-of-Range conditions.
            verbose (bool, optional)        : If True, prints progress information. 
        """
        # --- Core Simulation Objects ---
        self.aircraft       = aircraft
        self.sensors        = sensors
        self.shape_library  = shape_library

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
        for ac in self.aircraft:
            _generate_raw_trajectory(ac, self.max_duration)                                             # Generate trajectories up to the max possible duration

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
        terminal_exit_times = []                                                                        # Holds the last in-bounds time for each aircraft
        for ac in self.aircraft:
            # Create a mask for any data points that have left the simulation area
            mask                    = (np.abs(ac.raw_track[0, :]) > self.spine_length) | \
                                       (np.abs(ac.raw_track[1, :]) > self.spine_length)
            ac.raw_track[:, mask]   = np.nan                                                            # Nullify the out-of-bounds data points
            
            valid_times = ac.raw_time[~np.isnan(ac.raw_track[0,:])]                                     # Get the timestamps for all in-bounds points       
            if valid_times.size > 0:
                terminal_exit_times.append(np.max(valid_times))                                         # Record the aircraft's final valid timestamp

        # --- Step 2: Determine True Simulation Duration ---
        t_start             = min((np.nanmin(ac.raw_time) for ac in self.aircraft 
                                   if ac.raw_time is not None and ac.raw_time.size > 0), default = 0)   # Find the earliest start time across all aircraft
        true_event_duration = max(terminal_exit_times) if terminal_exit_times else t_start              # Find the latest exit time across all aircraft
        final_t_end         = min(true_event_duration, self.max_duration)                               # Clamp to the maximum allowed duration

        # --- Step 3: Create the Master Time Grid ---
        self.time_grid = np.arange(t_start, final_t_end + self.global_dt, self.global_dt)               # Create the master time grid
        K              = len(self.time_grid)                                                            # Total number of time steps

        # --- Step 4: Align Each Aircraft's Track to the Grid ---
        for ac in self.aircraft:
            ac.aligned_track    = np.full((4, K), np.nan)                                               # Initialize aligned track with NaNs
            valid_mask          = ~np.isnan(ac.raw_track[0, :])                                         # Find all valid (in-bounds) points in the raw track
            if not np.any(valid_mask): continue                                                         # Skip this aircraft if it has no valid data
            
            raw_times_valid                     = ac.raw_time[valid_mask]                               # Get the timestamps of the valid raw data
            grid_indices                        = np.searchsorted(self.time_grid, 
                                                                raw_times_valid, side = 'left')         # Find where each raw timestamp fits into the master grid
            grid_indices                        = np.clip(grid_indices, 0, K - 1)                       # Prevent indices from going out of bounds
            unique_indices, unique_raw_map      = np.unique(grid_indices, return_index = True)          # Get unique grid indices and their raw data indices
            ac.aligned_track[:, unique_indices] = ac.raw_track[:, valid_mask][:, unique_raw_map]        # Place the valid raw data onto the master grid

            # --- Step 4a: Housekeeping ---
            df = pd.DataFrame(ac.aligned_track.T)                                                       # Use a DataFrame for convenient interpolation  
            df.interpolate(method = 'linear', limit = 1, axis = 0, inplace = True)                      # Linearly interpolate small gaps (up to 1 consecutive NaN)
            ac.aligned_track = df.to_numpy().T                                                          # Convert back to NumPy array
            
            # Normalize the heading angle to ensure it stays within the [-pi, pi) range
            heading_row                 = ac.aligned_track[2, :]                                        # Extract the heading row   
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
                    
                    # --- Generate True Detections from All Aircraft ---
                    for aircraft in self.aircraft:
                        if not np.isnan(aircraft.aligned_track[0, k]):                                  # Only consider aircraft that are in-bounds at this time
                            aircraft_pose  = aircraft.aligned_track[:3, k]                              # Extract the aircraft's pose at this time
                            aircraft_id    = aircraft.identifier                                        # Extract aircraft identifier
                            aircraft_shape = self.shape_library[aircraft.shape_idx]                     # Extract aircraft shape information
                            
                            det_points, det_labels = _generate_detections(
                                aircraft_pose, aircraft_id, sensor, aircraft_shape, dropout_status,
                                self.use_los, self.use_oor
                            )                                                                           # Generate detections for this aircraft

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
# SECTION 2: CORE ALGORITHMS (HELPER FUNCTIONS)
# ==============================================================================

def _generate_raw_trajectory(aircraft: Aircraft, t_end: float):
    """Generates a raw, high-resolution trajectory for a single aircraft."""
    # --- Generate Time Vector ---
    dt          = aircraft.dt
    time_vector = np.arange(0, t_end + dt, dt)                                                          # Time vector for raw trajectory generation

    # --- Generate Control Inputs ---
    u_lon_vals  = aircraft.long_vals
    u_lon_times = aircraft.long_times
    u_cen_vals  = aircraft.centr_vals
    u_cen_times = aircraft.centr_times
    u_signal    = _convert_control_inputs(time_vector, u_lon_vals, 
                                          u_lon_times, u_cen_vals, u_cen_times)                         # Convert event-based controls to continuous signal

    # --- Initialize State and Storage ---
    x0 = np.array([aircraft.initial_position[0], aircraft.initial_position[1], 
                   aircraft.initial_heading, aircraft.initial_speed])                                   # Initial state vector [x, y, heading, speed]

    # --- Integrate Dynamics Using ZOH ---
    K       = len(time_vector)                                                                          # Total number of time steps
    x       = np.zeros((4, K))                                                                          # State storage
    x[:, 0] = x0                                                                                        # Set initial state
    x[2, 0] = np.arctan2(np.sin(x[2, 0]), np.cos(x[2, 0]))                                              # Wrap initial heading to [-pi, pi)

    for k in range(K - 1):
        x[:, k + 1], _ = _zoh_integrator(x[:, k], u_signal[:, k], dt)                                   # Perform ZOH integration step  

    aircraft.raw_time  = time_vector + aircraft.t_init                                                  # Store the raw time vector, adjusted for initial time
    aircraft.raw_track = x                                                                              # Store the raw trajectory

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

def _generate_detections(aircraft_pose: np.ndarray, aircraft_id: int, 
                         sensor: Sensor, aircraft_shape: dict, 
                         dropout_status: float, use_los: bool, use_oor: bool):
    """Generates a point cloud of true detections for a single aircraft."""
    # --- Unpack Aircraft Shape Information ---
    shape_matrix        = aircraft_shape['shapeMatrix']                                                 # Shape matrix defining the aircraft's geometry
    los_table           = aircraft_shape.get('LoSTable')                                                # Line-of-Sight table
    n_edges             = shape_matrix.shape[1]                                                         # Number of edges in the shape
    
    # --- Compute Edge Positions in World Frame ---
    edge_starts_local   = shape_matrix[:2, :].T                                                         # Edge start points in aircraft frame
    edge_ends_local     = shape_matrix[2:4, :].T                                                        # Edge end points in aircraft frame
    edge_lengths        = shape_matrix[4, :]                                                            # Edge lengths

    # --- Compute Flags and Detection Rates ---
    sensor_pose_vec     = np.array([sensor.position[0], sensor.position[1], sensor.heading])            # Sensor pose in world frame
    edge_starts_world   = _roto_translate(edge_starts_local, aircraft_pose)                             # Edge start points in world frame
    edge_ends_world     = _roto_translate(edge_ends_local, aircraft_pose)                               # Edge end points in world frame
    edge_centers_world  = (edge_starts_world + edge_ends_world) / 2                                     # Edge center points in world frame
    edge_centers_sensor = _roto_translate(edge_centers_world, sensor_pose_vec, inverse=True)            # Edge center points in sensor frame

    if use_oor:                                                                                         # If OOR is enabled...
        oor_flags = _compute_oor_flags(edge_centers_sensor, sensor.min_range, 
                                        sensor.max_range, sensor.min_bearing, 
                                        sensor.max_bearing)                                             # ...compute Out-of-Range flags
    else: oor_flags = np.ones(n_edges, dtype = int)                                                     # otherwise assume all edges are in range
    
    if use_los and los_table is not None:                                                               # If LoS is enabled and the table is provided...                         
        los_flags = _compute_los_flags(aircraft_pose, sensor_pose_vec, los_table)                       # ...compute Line-of-Sight flags
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
                        _sample_measurement(i, edge_starts_world, edge_ends_world, sensor.sigma_v)          
                        for i in selected_edges
                        ]    
    labels          = np.full(n_measurements, aircraft_id, dtype = int)                                 # Labels for each detection point (aircraft ID)

    return np.array(points), labels

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

def _compute_los_flags(aircraft_pose, sensor_pose, los_table):
    """Computes Line-of-Sight flags using a pre-computed visibility table."""
    # --- Find Grid Point Closest to Sensor  ---
    sensor_pos_ac   = _roto_translate(sensor_pose[:2].reshape(1, -1), 
                                      aircraft_pose, inverse = True)[0]                                 # Sensor position in aircraft frame
    diffs           = los_table['coords'] - sensor_pos_ac                                               # Differences to all grid points
    dists           = np.linalg.norm(diffs, axis = 2)                                                   # Euclidean distances to all grid points
    idx             = np.unravel_index(np.argmin(dists), dists.shape)                                   # Index of closest grid point  
    return los_table['flags'][idx[0], idx[1], :].astype(int)                                            # Return LoS flags for all edges 

def _compute_oor_flags(edge_centers_sensor, min_r, max_r, min_b, max_b):
    """Computes Out-of-Range flags based on sensor FoV."""
    ranges   = np.linalg.norm(edge_centers_sensor, axis = 1)                                            # Ranges to edge centers
    bearings = np.arctan2(edge_centers_sensor[:, 1], edge_centers_sensor[:, 0])                         # Bearings to edge centers
    return ((ranges >= min_r) & (ranges <= max_r) & \
            (bearings >= min_b) & (bearings <= max_b)).astype(int)                                      # Return OOR flags for all edges

def _sample_measurement(edge_idx, edge_starts, edge_ends, sigma_v):
    """Samples a single noisy measurement point along a specified edge."""
    u               = np.random.uniform()                                                               # Sample a uniform [0, 1]
    point_on_edge   = (1 - u) * edge_starts[edge_idx] + u * edge_ends[edge_idx]                         # Interpolate point along the edge 
    noise           = np.random.normal(scale = sigma_v, size = 2)                                       # Add Gaussian noise to the point
    return point_on_edge + noise








