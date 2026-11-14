"""
================================================================================
rendering_engine.py
================================================================================

This module contains the new, clean rendering engine for the TNS-2D project.

It provides a high-performance animation function that takes the completed
simulator object and a configuration dictionary to generate a high-quality
MP4 video. It is designed for clarity, performance, and maintainability.

--------------------------------------------------------------------------------
Author  : Matteo Tesori (matteotesori@gmail.com)
Date    : 2025-10-10
All wrongs reserved.
--------------------------------------------------------------------------------
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Wedge
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np
from tqdm import tqdm
import time
from typing import TYPE_CHECKING, List, Dict, Any
from dataclasses import dataclass

# Use a forward reference for the Simulator class to avoid circular imports
if TYPE_CHECKING:
    from ..simulation.simulator import Simulator

# ==============================================================================
# SECTION 1: CONFIGURATION CONSTANTS
# ==============================================================================

SEG_UPDATE_PERIOD           = 3                         # Defines how often (in frames) to perform expensive track maintenance
TITLE_UPDATE_PERIOD         = 1                         # Defines how often (in frames) to update the animation title
SIMPLIFY_THRESHOLD          = 300                       # The number of segments before track simplification is triggered
POINTCLOUD_FADE_DURATION_S  = 1.5                       # Duration of the point cloud fade-out effect in seconds
POINTCLOUD_DECAY_RATE       = 3.0                       # Exponential decay rate for the fade-out. Higher is faster

# ==============================================================================
# SECTION 2: RENDER STATE DATA CLASS
# ==============================================================================

@dataclass
class _RenderState:
    """A container for all data and state required by the animation callbacks."""
    ax                            : plt.Axes                  # Matplotlib axes object where the animation is drawn
    time_grid                     : np.ndarray                # Master time grid array for the entire simulation
    poses                         : np.ndarray                # 3D numpy array holding all extended object poses over time (4 x K x N)
    sample_indices                : List[int]                 # List of indices into the time_grid, one for each animation frame
    extended_object_artists       : List[Dict]                # List of dictionaries, each holding all matplotlib artists for one extended object
    sensor_list                   : List[Any]                 # List of sensor configuration objects from the simulator.
    pointcloud_states             : List[Dict]                # List of state dictionaries for rendering sensor point clouds (clutter/detections)
    sensor_data_map               : List[Dict]                # Pre-processed map for fast lookup of sensor data by time index
    config                        : Dict                      # Main configuration dictionary containing all rendering settings
    num_sensors                   : int                       # Total number of sensors in the simulation
    start_time                    : float                     # Simulation start time in seconds, used for the title display.
    n_frames                      : int                       # Total number of frames to be rendered in the final animation.
    extended_object_id_to_idx_map : Dict[int, int]            # Mapping from a unique extended object ID to its corresponding index in data arrays
    sensor_alphas_np              : np.ndarray                # Numpy array of alpha (transparency) values, one for each sensor
    extended_object_alphas_np     : np.ndarray                # Numpy array of alpha (transparency) values, one for each extended object

# ==============================================================================
# SECTION 3: MAIN RENDERING ENGINE
# ==============================================================================

def render_animation(simulator: 'Simulator', config: dict):
    """
    Generates a high-performance animation from the completed simulation results.
    """
    t_start_render = time.time()                                                                    # Record the start time for performance tracking
    print("Rendering engine started...")

    # --- Step 1: Unpack Data and Config ---
    extended_object_list = simulator.extended_objects                                               # Extract the list of extended object objects from the simulator
    sensor_list          = simulator.sensors                                                        # Extract the list of sensor objects from the simulator
    shape_library        = simulator.shape_library                                                  # Extract the library of extended object polygon shapes
    time_grid            = simulator.time_grid                                                      # Extract the master time grid for the entire simulation

    print(f"Rendering {len(extended_object_list)} extended objects and {len(sensor_list)} sensors.")

    # --- Step 2: Prepare Data for Animation ---
    num_extended_objects = len(extended_object_list)                                                # Get the total number of extended objects in the simulation
    num_sensors          = len(sensor_list)                                                         # Get the total number of sensors in the simulation
    K                    = len(time_grid)                                                           # Get the total number of time steps in the simulation
    
    poses = np.full((4, K, num_extended_objects), np.nan)                                           # Initialize the main poses array with NaNs for missing data
    for n, eo in enumerate(extended_object_list):                                                   # Iterate through each extended object to populate the main poses array
        L               = eo.aligned_track.shape[1]                                                 # Get the length of the current extended object's aligned track
        poses[:, :L, n] = eo.aligned_track                                                          # Copy the aligned track into the main poses array
    
    censor_map                    = {'null': 1.0, 'mild': 0.3, 'total': 0.0}                        # Define a mapping from censorship levels to transparency values
    sensor_alphas                 = [censor_map.get(level, 1.0) 
                                        for level in config['sensor_censor_levels']]                # Create a list of alpha values, one for each sensor
    extended_object_alphas        = [censor_map.get(level, 1.0) * config.get('alpha', 1.0) 
                                        for level in config['extended_object_censor_levels']]       # Create a list of alpha values, one for each extended object
    extended_object_id_to_idx_map = {eo.identifier: i for i, eo in enumerate(extended_object_list)} # Create a mapping from extended object ID to index in the poses array



    # --- Step 3: Figure and Axes Setup ---
    fig, ax = plt.subplots(figsize = config.get('FIGSIZE', (16, 9)), dpi = config.get('DPI', 120))  # Create the main figure and axes for the animation
    _setup_axes(ax, config['SPINE_LENGTH'])                                                         # Configure the axes properties 

    # --- Step 4: Create All Visual Artists ---
    extended_object_artists = [_create_extended_object_artists(ax, eo, poses[:, 0, n], 
                                                               shape_library, 
                                                               config['extended_object_colors'][n], 
                                                               extended_object_alphas[n], 
                                                               config.get('linewidth', 3.0), K)
                               for n, eo in enumerate(extended_object_list)]                        # Create all artists (polygons, lines) for each extended object
    pointcloud_states       = [_create_pointcloud_artists(ax) for _ in sensor_list]                 # Create artists and state containers for sensor data 

    for s_idx, sensor in enumerate(sensor_list):                                                    # Iterate through sensors to create their Field of View artists
        if sensor_alphas[s_idx] > 0:                                                                # Only create an artist if the sensor is not fully transparent
            _create_sensor_fov_artist(ax, sensor, config['sensor_colors'][s_idx], 
                                      sensor_alphas[s_idx])

    # --- Step 5: Setup Animation Loop ---
    DELTA_T         = time_grid[1] - time_grid[0] if len(time_grid) > 1 else 0                      # Calculate the time step of the master simulation grid
    FPS             = config.get('FPS', 60)                                                         # Get the desired frames per second for the output video
    DURATION_S      = config.get('DURATION_S', (time_grid[-1] - time_grid[0]))                      # Get the desired total duration of the output video in seconds
    steps_per_frame = max(1, int(round((1.0 / FPS) / DELTA_T))) if DELTA_T > 0 else 1               # Calculate how many simulation steps to skip per animation frame
    n_frames        = min((K - 1) // steps_per_frame + 1, int(FPS * DURATION_S) + 1)                # Determine the total number of frames for the final animation
    sample_indices  = [min(frame * steps_per_frame, K - 1) for frame in range(n_frames)]            # List of time indices to sample for each animation frame
    
    # --- Step 6: Create the RenderState Object ---
    render_state = _RenderState(                                                                    # Instantiate the render state dataclass with all necessary data
        ax                            = ax,                                                         # Matplotlib Axes object
        time_grid                     = time_grid,                                                  # Master time grid
        poses                         = poses,                                                      # 3D array of all extended object poses
        sample_indices                = sample_indices,                                             # List of time indices for each frame
        extended_object_artists       = extended_object_artists,                                    # List of extended object artist dictionaries
        sensor_list                   = sensor_list,                                                # List of sensor objects
        pointcloud_states             = pointcloud_states,                                          # List of point cloud artist state containers
        sensor_data_map               = _create_sensor_data_map(sensor_list, time_grid, K),         # Pre-processed sensor data map for fast lookup
        config                        = config,                                                     # Configuration dictionary
        num_sensors                   = num_sensors,                                                # Total number of sensors
        start_time                    = time_grid[0],                                               # Simulation start time
        n_frames                      = n_frames,                                                   # Total number of frames to render
        sensor_alphas_np              = np.array(sensor_alphas),                                    # Numpy array of sensor alpha values    
        extended_object_alphas_np     = np.array(extended_object_alphas),                           # Numpy array of extended object alpha values
        extended_object_id_to_idx_map = extended_object_id_to_idx_map                               # Mapping from extended object IDs to their indices
    )

    anim = FuncAnimation(fig, _update_frame, frames = n_frames, fargs = (render_state,),
                         init_func = lambda: _init_artists(render_state), 
                         interval = 1000 / FPS, blit = True)                                        # Create the main animation object

    # --- Step 7: Save Animation ---
    writer = FFMpegWriter(fps = FPS, codec = 'libx264',
                          extra_args = ['-pix_fmt', str(config.get('ENCODER_PIX_FMT', 'yuv420p')),
                                        '-crf', str(int(config.get('ENCODER_CRF', 18))),
                                        '-preset', str(config.get('ENCODER_PRESET', 'veryfast'))])
    with tqdm(total = n_frames, desc = "Rendering frames", unit = "frame") as pbar:                 # Use tqdm to show a progress bar during the rendering process
        anim.save(config['MP4_PATH'], writer = writer, dpi = config.get('DPI', 120), 
                    progress_callback = lambda i, n: pbar.update(1))                                # Save the animation to the specified MP4 file

    t_end_render    = time.time()                                                                   # Record the end time for performance tracking
    duration        = t_end_render - t_start_render                                                 # Calculate the total rendering duration
    print(f"Rendering complete in {duration:.2f} seconds.")
    print(f"Animation saved to {config['MP4_PATH']}")

# ==============================================================================
# SECTION 4: ANIMATION CALLBACKS AND HIGH-LEVEL UPDATE FUNCTIONS
# ==============================================================================

def _init_artists(state: _RenderState):
    """Initializes all artists for blitting."""
    artists = [state.ax.title]                                                                      # Start the list of artists with the plot title
    for art in state.extended_object_artists: artists.extend([art['poly'], art['lc']])              # Add all extended object polygons and line collections to the list
    for s in state.pointcloud_states:                                                               # Add all sensor data artists to the list
        artists.extend([s['clutter_artist'], s['detection_artist']])
    return tuple(artists)                                                                           # Return a tuple of all artists to be managed by the animation

def _update_frame(frame, state: _RenderState):
    """The main animation callback function, executed for each frame."""
    sample_idx   = state.sample_indices[frame]                                                      # Get the corresponding time index for this frame
    
    _update_all_extended_objects(state, sample_idx, frame)                                          # Update the position and track of every extended object
    _update_all_pointclouds(state, sample_idx)                                                      # Update the point clouds for every sensor
    _update_title(state.ax, frame, state.n_frames,                                                  # Update the title with current time and frame info
                  state.start_time + sample_idx * (state.time_grid[1] - state.time_grid[0]))

    return _init_artists(state)                                                                     # Return the tuple of all artists to be redrawn in this frame

# ==============================================================================
# SECTION 5: ARTIST CREATION HELPERS
# ==============================================================================

def _create_sensor_fov_artist(ax, sensor, color, alpha):
    """Creates and adds the artists for a single sensor's FoV and position."""
    center      = sensor.position                                                                   # Get the sensor's [x, y] position
    heading_deg = np.rad2deg(sensor.heading)                                                        # Convert the sensor's heading from radians to degrees
    theta1      = heading_deg + np.rad2deg(sensor.min_bearing)                                      # Calculate the start angle of the FoV wedge
    theta2      = heading_deg + np.rad2deg(sensor.max_bearing)                                      # Calculate the end angle of the FoV wedge
    fov_wedge   = Wedge(center, sensor.max_range, theta1, theta2,                                   # Create the Matplotlib Wedge artist for the FoV
                        width = sensor.max_range - sensor.min_range,
                        facecolor = _color_to_rgba(color, 0.15 * alpha),
                        edgecolor = None, zorder = 1)
    ax.add_patch(fov_wedge)                                                                         # Add the FoV wedge to the plot axes
    ax.plot(center[0], center[1], 'o', markersize = 5, 
            color = color, zorder = 2, alpha = alpha)                                               # Add a marker for the sensor's physical position

def _create_extended_object_artists(ax, extended_object, init_pose, shape_library, color, alpha, lw, K):
    """Creates a dictionary of artists and state for a single extended object."""
    poly_verts, _   = _get_extended_object_frame(init_pose, 
                                                 shape_library[extended_object.shape_idx])          # Get the initial polygon vertices for the extended object shape
    poly            = Polygon(poly_verts.T, closed = True, 
                              facecolor = color, alpha = alpha, zorder = 10)                        # Create the main Polygon artist for the extended object body
    lc              = LineCollection([], colors = [(*color, alpha)], linewidth = lw, zorder = 5)    # Create an empty LineCollection for the extended object's track
    ax.add_patch(poly)                                                                              # Add the polygon to the plot axes
    ax.add_collection(lc)                                                                           # Add the line collection to the plot axes
    return {
        'poly'                  : poly,                                                             # The Polygon artist for the extended object's body
        'lc'                    : lc,                                                               # The LineCollection artist for the track
        'extended_object_shape' : shape_library[extended_object.shape_idx],                         # The raw vertex data for the extended object's shape
        'buffer'                : np.zeros((K * 2, 2), dtype = float),                              # A pre-allocated buffer to store track points efficiently
        'buf_len'               : 0,                                                                # The current number of points in the buffer
        'segments'              : [],                                                               # A list to hold finalized track segments
        'was_visible'           : not np.isnan(init_pose[0])                                        # A flag to track visibility state across frames
    }

def _create_pointcloud_artists(ax):
    """Creates a state dictionary for a sensor's point cloud artists."""
    return { 
        'clutter_artist'        : ax.scatter([], [], marker='.', s =  4, zorder = 19),              # Artist for rendering clutter points (small dots)
        'detection_artist'      : ax.scatter([], [], marker='s', s = 24, zorder = 20),              # Artist for rendering true detections (larger squares)
        'last_update_time'      : -np.inf,                                                          # Timestamp of the last data update for fading calculations
        'last_clutter_rgba'     : np.empty((0, 4)),                                                 # Stores the RGBA color of the last clutter points
        'last_clutter_pts'      : np.empty((0, 2)),                                                 # Stores the [x, y] coordinates of the last clutter points
        'last_detection_rgba'   : np.empty((0, 4)),                                                 # Stores the RGBA color of the last detection points
        'last_detection_pts'    : np.empty((0, 2)),                                                 # Stores the [x, y] coordinates of the last detection points
    }


# ==============================================================================
# SECTION 6: ARTIST UPDATE HELPERS
# ==============================================================================

def _update_all_extended_objects(state: _RenderState, sample_idx, frame):
    """Updates all extended object polygons and tracks for the current frame."""
    for n, art in enumerate(state.extended_object_artists):
        pose        = state.poses[:, sample_idx, n]                                                 # Get the current pose [x, y, heading, speed] for this extended object
        is_visible  = not np.isnan(pose[0])                                                         # Determine if the extended object is visible (not NaN)
        
        _update_extended_object_track(art, pose[:2], is_visible)                                    # Update the extended object's track data with the new position
        
        if is_visible:
            _update_extended_object_polygon(art, pose)                                              # If visible, update the polygon's position and orientation
        else:
            art['poly'].set_visible(False)                                                          # If not visible, hide the polygon artist
        art['was_visible'] = is_visible                                                             # Store the current visibility state for the next frame's logic
        
        if frame > 0 and frame % SEG_UPDATE_PERIOD == 0:                                            # Periodically maintain and simplify track segments
            _maintain_track_segments(art, state.num_sensors, 
                                     state.config.get('min_linewidth', 0.3), 
                                     state.config.get('linewidth', 3.0))

def _update_extended_object_polygon(art, pose):
    """Updates a single extended object's polygon to a new pose."""
    if not art['poly'].get_visible():                                                               # If the polygon was previously hidden, make it visible
        art['poly'].set_visible(True)
    verts, _ = _get_extended_object_frame(pose, art['extended_object_shape'])                       # Calculate the new vertex positions based on the current pose
    art['poly'].set_xy(verts.T)                                                                     # Set the new vertices for the polygon artist

def _update_extended_object_track(art, point, is_visible):
    """Updates the track data for a single extended object, handling fragmentation."""
    if is_visible:
        if not art['was_visible']:                                                                  # If the extended object just reappeared after being hidden...
            art['segments'].append((-1, -1, 0))                                                     # ...add a "break" segment to create a gap in the track line
        
        idx = art['buf_len']                                                                        # Get the index for the new point in the buffer.
        art['buffer'][idx] = point                                                                  # Add the new [x, y] point to the track buffer.
        art['buf_len'] += 1                                                                         # Increment the buffer length.
        
        if not art['segments'] or art['segments'][-1][0] == -1:                                     # If this is the start of a new, continuous track segment...
            art['segments'].append((idx, idx, 0))                                                   # ...create a new segment entry.
        else:
            s0, e0, ac0 = art['segments'][-1]                                                       # Otherwise, extend the end index of the current active segment
            art['segments'][-1] = (s0, idx, ac0)

def _update_all_pointclouds(state: _RenderState, sample_idx):
    """Updates all sensor point cloud artists for the current frame with a fading effect."""
    current_time = state.time_grid[sample_idx]                                                      # Get the simulation time for the current frame

    for s_idx, pc_state in enumerate(state.pointcloud_states):
        # --- Step 1: Ingest new data if available ---
        if sample_idx in state.sensor_data_map[s_idx]:                                              # Check if there is new sensor data for this time index
            pc_state['last_update_time']    = current_time                                          # If so, update the timestamp to reset the fading effect
            meas_idx                        = state.sensor_data_map[s_idx][sample_idx]              # Get the index of the measurement within the sensor's dataset
            dataset                         = state.sensor_list[s_idx].dataset                      # Get the full dataset for the current sensor
            points, labels                  = dataset['measurements'][meas_idx], \
                                                dataset['labels'][meas_idx]                         # Unpack the measurement points and their corresponding labels
            
            clutter_pts   = points[labels == -1]                                                    # Filter out points that are labeled as clutter (-1)
            detection_pts = points[labels >= 0]                                                     # Filter out points that are labeled as true detections (>= 0)
            
            pc_state['last_clutter_pts']   = clutter_pts                                            # Store the current clutter points in the state
            pc_state['last_detection_pts'] = detection_pts                                          # Store the current detection points in the state

            if clutter_pts.size > 0:                                                                # If there are any clutter points to render...
                color = state.config.get('clutter_color_override') \
                        or state.config['sensor_colors'][s_idx]                                     # Override color if specified, otherwise use native color.
                alpha = 0.7 * state.sensor_alphas_np[s_idx]                                         # Set the base alpha for clutter points
                rgba  = _color_to_rgba(color, alpha)                                                # Convert the color and alpha to an RGBA tuple
                pc_state['last_clutter_rgba'] = np.tile(rgba, (clutter_pts.shape[0], 1))            # Create an array of this RGBA color for all clutter points.
                pc_state['clutter_artist'].set_offsets(clutter_pts)                                 # Update the artist's point positions
            else:
                pc_state['last_clutter_rgba'] = np.empty((0, 4))                                    # If no clutter, clear the stored color array

            if detection_pts.size > 0:                                                              # If there are any detection points to render...
                color = state.config['sensor_colors'][s_idx]                                        # Use the sensor's native color
                alpha = 1.0 * state.sensor_alphas_np[s_idx]                                         # Set the base alpha for detection points
                rgba  = _color_to_rgba(color, alpha)                                                # Convert to an RGBA tuple
                pc_state['last_detection_rgba'] = np.tile(rgba, (detection_pts.shape[0], 1))        # Create a color array for all detection points
                pc_state['detection_artist'].set_offsets(detection_pts)                             # Update the artist's point positions
            else:
                pc_state['last_detection_rgba'] = np.empty((0, 4))                                  # If no detections, clear the stored color array

        # --- Step 2: Apply fading logic and render ---
        age = current_time - pc_state['last_update_time']                                           # Calculate how long ago the last data packet was received
        
        should_render = age <= POINTCLOUD_FADE_DURATION_S                                           # Determine if the points are still young enough to be visible
        if should_render:
            alpha_frac = np.exp(-POINTCLOUD_DECAY_RATE * (age / POINTCLOUD_FADE_DURATION_S))        # Calculate the transparency fraction using an exponential decay
            if alpha_frac < 0.01:                                                                   # If the points have become almost fully transparent...
                should_render = False                                                               # ...mark them as not renderable to hide them completely

        # Render Clutter
        clutter_artist = pc_state['clutter_artist']
        if should_render and pc_state['last_clutter_pts'].size > 0:                                 # If the points should be rendered and exist...
            faded_colors        = pc_state['last_clutter_rgba'].copy()                              # Create a copy of the base colors to modify
            faded_colors[:, 3] *= alpha_frac                                                        # Apply the calculated fade to the alpha channel
            clutter_artist.set_facecolors(faded_colors)                                             # Set the new, faded colors on the artist
            if not clutter_artist.get_visible():                                                    # If the artist was previously hidden, make it visible
                clutter_artist.set_visible(True)
        else:
            if clutter_artist.get_visible():                                                        # Otherwise, if the artist is currently visible, hide it
                clutter_artist.set_visible(False)
                
        # Render Detections
        detection_artist = pc_state['detection_artist']
        if should_render and pc_state['last_detection_pts'].size > 0:                               # Repeat the same rendering logic for the detection points
            faded_colors        = pc_state['last_detection_rgba'].copy()
            faded_colors[:, 3] *= alpha_frac
            detection_artist.set_facecolors(faded_colors)
            if not detection_artist.get_visible():
                detection_artist.set_visible(True)
        else:
            if detection_artist.get_visible():
                detection_artist.set_visible(False)

def _update_title(ax, frame, n_frames, time):
    """Updates the animation title with frame and time info."""
    frame_width = len(str(n_frames - 1))                                                            # Calculate the string width for alignment
    time_width  = len(f"{(n_frames * (1/60)):.2f}")                                                 # Calculate the string width for alignment
    
    frame_str   = f"Frame: {frame:>{frame_width}d}/{n_frames - 1}"                                  # Format the frame counter part of the title
    time_str    = f"Time: {time:>{time_width}.2f} [s]"                                              # Format the time part of the title
    title       = f"{frame_str}    {time_str}"                                                      # Combine the two parts to form the final title string

    ax.set_title(title, fontfamily='monospace')                                                     # Monospace font for alignment

# ==============================================================================
# SECTION 7: TRACK MANAGEMENT HELPERS
# ==============================================================================

def _maintain_track_segments(art, num_sensors, min_lw, full_lw):
    """Orchestrates track compaction and simplification for performance."""
    _compact_segments(art)                                                                          # Merge any consecutive small segments into larger ones
    if len(art['segments']) > SIMPLIFY_THRESHOLD:                                                   # If the number of segments exceeds a threshold...
        _simplify_old_segments(art)                                                                 # ...simplify the older segments to reduce complexity
    _update_linecollection_from_segments(art, num_sensors, min_lw, full_lw)                         # Update the visual line artist with the new segment data

def _compact_segments(art):
    """Merges consecutive track segments, respecting breaks."""
    if not art['segments']: return                                                                  # If there are no segments, do nothing
    merged = []                                                                                     # Initialize a list to hold the new, compacted segments
    for s, e, ac in art['segments']:                                                                # Iterate through each existing segment
        if s == -1:                                                                                 # If the segment is a "break" marker (start_idx == -1)...
            merged.append((s, e, ac))                                                               # ...add it directly to the merged list to preserve the gap
            continue
        if (merged and merged[-1][2] == ac and                                                      # If the last segment in merged has the same activity code...
            s == merged[-1][1] + 1 and merged[-1][0] != -1):
            merged[-1] = (merged[-1][0], e, ac)                                                     # ...extend the last segment's end_idx 
        else:
            merged.append((s, e, ac))                                                               # Otherwise, it's a new, distinct segment, so append it
    art['segments'] = merged                                                                        # Replace the old segments list with the newly compacted one

def _simplify_old_segments(art, eps = 0.5):
    """Simplifies old track segments individually using RDP algorithm to preserve fragmentation."""
    if len(art['segments']) <= 2: return                                                            # Don't simplify if the track is very short
    n                   = len(art['segments']) // 2                                                 # Determine the midpoint to split old segments from recent ones
    old, remaining      = art['segments'][:n], art['segments'][n:]                                  # Split the segments list into two parts
    newly_simplified    = []                                                                        # Initialize a list for the newly created simplified segments
    for s, e, ac in old:                                                                            # Iterate through only the old segments
        if s == -1:                                                                                 # Preserve break markers
            newly_simplified.append((s, e, ac))
            continue
        if e < s: continue                                                                          # Skip any invalid or empty segments
        
        points          = art['buffer'][s:e + 1]                                                    # Extract the vertex coordinates for the current old segment
        simple          = _rdp(points, eps)                                                         # Apply the RDP algorithm to simplify the points
        start_idx, m    = art['buf_len'], len(simple)                                               # Determine where to place the simplified points in the buffer
        
        if start_idx + m >= len(art['buffer']):                                                     # If the main buffer is too small to hold the simplified points
            new_size                        = int((len(art['buffer']) + m) * 1.5) + 1               # ...calculate a new, larger size
            new_buffer                      = np.zeros((new_size, 2))                               # Create a new, larger buffer
            new_buffer[:len(art['buffer'])] = art['buffer']                                         # Copy the old buffer's contents into the new one
            art['buffer']                   = new_buffer                                            # Replace the old buffer

        art['buffer'][start_idx:start_idx + m] = simple                                             # Copy the simplified points to the end of the buffer
        art['buf_len'] += m                                                                         # Update the buffer's current length
        newly_simplified.append((start_idx, start_idx + m - 1, ac))                                 # Create a new segment tuple pointing to the simplified points
        
    art['segments'] = newly_simplified + remaining                                                  # Combine newly simplified old segments with untouched segments

def _update_linecollection_from_segments(art, num_sensors, min_lw, full_lw):
    """Updates the LineCollection artist from the current segment data."""
    segs, buf = art['segments'], art['buffer']                                                      # Get shortcuts to the segments list and the points buffer
    if not segs: art['lc'].set_segments([]); return                                                 # If there are no segments, clear the artist and exit
    seg_arrays = []                                                                                 # Initialize a list to hold the line segments as arrays
    for s, e, ac in segs:                                                                           # Iterate through the segment tuples
        if s == -1: continue                                                                        # Skip break markers as they are not drawn
        if e < s: continue                                                                          # Skip invalid or empty segments
        seg_arrays.append(buf[s:e + 1])                                                             # Extract the segment from the buffer and add to the list
    art['lc'].set_segments(seg_arrays)                                                              # Set the segments on the LineCollection artist

# ==============================================================================
# SECTION 8: UTILITIES
# ==============================================================================

def _setup_axes(ax, spine_length, bg_color=(0.15, 0.21, 0.23), text_color='white'):
    """Configures the appearance of the Matplotlib axes for a dark, technical look."""
    ax.set_facecolor(bg_color)                                                                      # Set the background color of the plotting area
    ax.figure.patch.set_facecolor(bg_color)                                                         # Set the background color of the figure, outside the plot
    ax.set_aspect('equal', 'box')                                                                   # Ensure that pixels are square and the plot is not distorted
    ax.set_xlim(-spine_length, spine_length)                                                        # Set the x-axis limits
    ax.set_ylim(-spine_length, spine_length)                                                        # Set the y-axis limits
    ax.tick_params(colors = text_color)                                                             # Set the color of the tick marks and their labels
    ax.set_xlabel("[m]", color = text_color, fontfamily = 'monospace')                              # Set the x-axis label
    ax.set_ylabel("[m]", color = text_color, fontfamily = 'monospace')                              # Set the y-axis label
    ax.title.set_color(text_color)                                                                  # Set the color of the main plot title
    for spine in ax.spines.values(): spine.set_edgecolor(text_color)                                # Set the color of the plot borders (the "spines")

def _get_extended_object_frame(pose, extended_object_shape):
    """Computes the world coordinates of an extended object's polygon vertices."""
    x, y, heading, _                = pose                                                          # Unpack the extended object's current position and heading
    contourShapeMatrix              = extended_object_shape['contourShapeMatrix']                   # Get the local-frame shape matrix for the extended object
    verts                           = np.empty((2, contourShapeMatrix.shape[1] * 2), dtype = float) # Pre-allocate an array to hold the vertices.
    verts[:, 0::2], verts[:, 1::2]  = contourShapeMatrix[0:2, :], contourShapeMatrix[2:4, :]        # Unpack the shape matrix into a standard (2, N) vertex array
    c, s                            = np.cos(heading), np.sin(heading)                              # Calculate the cosine and sine of the heading for rotation
    R                               = np.array([[c, -s], [s, c]])                                   # Construct the 2D rotation matrix
    rotated_verts                   = R @ verts                                                     # Rotate the local-frame vertices
    world_verts                     = rotated_verts + np.array([[x], [y]])                          # Translate rotated vertices to extended object's world position
    return world_verts, np.array([x, y])                                                            # Return final world-frame vertices and center position

def _color_to_rgba(color, alpha):
    """Converts a color to an RGBA tuple, handling various color formats."""
    if isinstance(color, (tuple, list)) and len(color) == 3:                                        # If the color is already an RGB tuple/list...
        return (*color, alpha)                                                                      # ...simply append the alpha value
    if isinstance(color, str):                                                                      # If the color is a string (e.g., 'red', '#FF0000')...
        from matplotlib import colors                                                               # ...use Matplotlib's robust color converter
        return colors.to_rgba(color, alpha)                                                         # Convert the string to an RGBA tuple with the specified alpha
    return color                                                                                    # Fallback: return the color as-is if the format is unrecognized

def _rdp(points, eps):
    """Ramer-Douglas-Peucker line simplification algorithm."""
    # Base case: if the line has 2 or fewer points, it can't be simplified.
    if len(points) < 3:
        if len(points) > 0:
            return np.array([points[0], points[-1]]) 
        else:
            return np.empty((0,2))

    # Get the start and end points of the line segment.
    start, end = points[0], points[-1]                                                          
    if np.all(start == end): return start.reshape(1,2)                                          # Handle the edge case where the start and end points are the same
    
    # Calculate the perpendicular distance of each point from the line connecting start and end.
    with np.errstate(divide = 'ignore', invalid = 'ignore'):                                    # Suppress potential divide-by-zero warnings for zero-length lines
        line_vec    = end - start                                                               # Vector from start to end point
        point_vecs  = points[1:-1] - start                                                      # Vectors from start point to all intermediate points
        numerator   = np.abs(np.cross(line_vec, point_vecs))                                    # Area of parallelogram formed by line_vec and point_vecs
        denominator = np.linalg.norm(line_vec)                                                  # Length of the line segment
        dists       = numerator / denominator                                                   # Perpendicular distances from points to the line

    if not np.any(dists) or not np.all(np.isfinite(dists)): return np.array([start, end])       # If distances are invalid, return the original endpoints
    
    idx = np.nanargmax(dists)                                                                   # Find the index of the point that is farthest from the line
    if dists[idx] <= eps:                                                                       # If the farthest point is within the tolerance 'eps'...
        return np.array([start, end])                                                           # ...then we can simplify this segment to a straight line
    
    idx    += 1                                                                                 # Adjust index to match the original 'points' array slicing
    left    = _rdp(points[:idx + 1], eps)                                                       # Recursively simplify the line segment to the left
    right   = _rdp(points[idx:], eps)                                                           # Recursively simplify the line segment to the right
    return np.vstack([left[:-1], right])                                                        # Combine the results, removing the duplicate middle point

def _create_sensor_data_map(sensor_list, time_grid, K):
    """Creates a fast lookup map from simulation time index to measurement index for each sensor."""
    sensor_data_map = []                                                                        # Initialize the list that will hold a map for each sensor
    for sensor in sensor_list:                                                                  # Iterate through each sensor in the simulation
        index_map = {}                                                                          # Initialize a dictionary to hold the mapping for this sensor
        
        if hasattr(sensor, 'dataset') and sensor.dataset and 'times' in sensor.dataset:         # Check if the sensor has produced a dataset with timing information
            indices = np.searchsorted(time_grid, sensor.dataset['times'], side='left')          # Find indices in time grid that correspond to each measurement time
            for meas_idx, sim_idx in enumerate(indices):                                        # Iterate through the found simulation indices
                if sim_idx < K: index_map[sim_idx] = meas_idx                                   # If the index is valid, add the mapping
        sensor_data_map.append(index_map)                                                       # Add the completed map for this sensor to the main list
    return sensor_data_map                                                                      # Return the list of lookup maps


