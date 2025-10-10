"""
================================================================================
run_simulation.py
================================================================================

This is the main entry point for the TNS-2D simulation project.

This script defines a specific simulation scenario by configuring the aircraft
and sensor objects. It then instantiates the `Simulator` class, runs the full
data generation pipeline, and calls the rendering engine to produce the final
animated video.

--------------------------------------------------------------------------------
Author  : Matteo Tesori (matteotesori@gmail.com)
Date    : 2025-10-10
All wrongs reserved.
--------------------------------------------------------------------------------
"""

import numpy as np
import pickle
from pathlib import Path


from src.tns_2d.simulation.data_structures  import Aircraft, Sensor
from src.tns_2d.simulation.simulator        import Simulator
from src.tns_2d.graphics.rendering_engine   import render_animation


# ==============================================================================
# SECTION 1: HELPER FUNCTIONS AND PATH DEFINITIONS
# ==============================================================================

def    g2mss(arr,    g =   9.81): return np.asarray(arr) *    g
def mach2mss(arr, Mach = 343.00): return np.asarray(arr) * Mach

# --- Path Definitions ---
ROOT_DIR           = Path(__file__).resolve().parent
DATA_DIR           = ROOT_DIR   / "data"
OUTPUT_DIR         = ROOT_DIR   / "output"
SHAPE_LIBRARY_PATH = DATA_DIR   / "shape_library.pkl"
MP4_PATH           = OUTPUT_DIR / "simulation_output.mp4"

# ==============================================================================
# SECTION 2: SCENARIO CONFIGURATION
# ==============================================================================

# --- Load Shape Library ---
with open(SHAPE_LIBRARY_PATH, 'rb') as f:
    shape_library = pickle.load(f)

# --- Define Constants ---
DELTA_T         = 1.0 / 120.0
SPINE_LENGTH    = 50.0
R               = SPINE_LENGTH * 0.98
y_offset        = -0.25 * R 

# --- Aircraft Definitions (Data-Driven) ---
aircraft_definitions = [
    {
        'identifier'       : 0,
        'shape_idx'        : 2,
        't_init'           : 2.0,
        'initial_position' : np.array([SPINE_LENGTH, 10.0]),
        'initial_heading'  : np.deg2rad(180),
        'initial_speed'    : mach2mss(0.05),
        'long_vals'        : g2mss([0.02]),
        'long_times'       : np.array([30.00]),
        'centr_vals'       : g2mss([-0.50]),
        'centr_times'      : np.array([30.00]),
        'dt'               : DELTA_T
    },
    {
        'identifier'       : 1,
        'shape_idx'        : 1,
        't_init'           : 4.0,
        'initial_position' : np.array([-SPINE_LENGTH, -10.0]),
        'initial_heading'  : np.deg2rad(0),
        'initial_speed'    : mach2mss(0.06),
        'long_vals'        : g2mss([0.00]),
        'long_times'       : np.array([30.00]),
        'centr_vals'       : g2mss([0.40]),
        'centr_times'      : np.array([35.00]),
        'dt'               : DELTA_T
    },
    {
        'identifier'       : 2,
        'shape_idx'        : 4,
        't_init'           : 5.0,
        'initial_position' : np.array([0.0, -SPINE_LENGTH]),
        'initial_heading'  : np.deg2rad(99),
        'initial_speed'    : mach2mss(0.04),
        'long_vals'        : g2mss([0.03]),
        'long_times'       : np.array([15.00]),
        'centr_vals'       : g2mss([0.00]),
        'centr_times'      : np.array([30.00]),
        'dt'               : DELTA_T
    }
]

# --- Sensor Definitions (Data-Driven) ---
sensor_definitions = [
    {
        'identifier'   : 0,
        'position'     : np.array([0.0, R + y_offset]),
        'heading'      : np.deg2rad(-90),
        'min_range'    : 5.0, 
        'max_range'    : 50.0, 
        'min_bearing'  : -np.deg2rad(35), 
        'max_bearing'  :  np.deg2rad(35),
        'resolution'   : 2.0, 
        'pd'           : 0.75,
        'clutter_rate' : 0.05, 
        'sigma_v'      : 0.2,
        'dt_sample'    : 1.0, 
        't_offset'     : 0.0,
        'alpha'        : 0.8
    },
    {
        'identifier'   : 1,
        'position'     : np.array([R * np.cos(np.deg2rad(210)), R * np.sin(np.deg2rad(210)) + y_offset]), 
        'heading'      : np.deg2rad(30), 
        'min_range'    : 5.0, 
        'max_range'    : 50.0,
        'min_bearing'  : -np.deg2rad(35), 
        'max_bearing'  :  np.deg2rad(35),
        'resolution'   : 2.0, 
        'pd'           : 0.75,
        'clutter_rate' : 0.05, 
        'sigma_v'      : 0.15,
        'dt_sample'    : 1.0, 
        't_offset'     : 0.0,
        'alpha'        : 0.8
    },
    {
        'identifier'   : 2,
        'position'     : np.array([R * np.cos(np.deg2rad(330)), R * np.sin(np.deg2rad(330)) + y_offset]), 
        'heading'      : np.deg2rad(150), 
        'min_range'    : 5.0, 
        'max_range'    : 50.0,
        'min_bearing'  : -np.deg2rad(35), 
        'max_bearing'  :  np.deg2rad(35),
        'resolution'   : 2.0, 
        'pd'           : 0.75,
        'clutter_rate' : 0.05, 
        'sigma_v'      : 0.25,
        'dt_sample'    : 1.0, 
        't_offset'     : 0.0,
        'alpha'        : 0.8
    }
]

# --- Create Objects from Definitions ---
aircraft_list = [Aircraft(**params) for params in aircraft_definitions]
sensor_list   = [  Sensor(**params) for params in   sensor_definitions]

# ==============================================================================
# SECTION 3: SIMULATION CONFIGURATION
# ==============================================================================

simulation_config = {
    'aircraft'      : aircraft_list,
    'sensors'       : sensor_list,
    'shape_library' : shape_library,
    'global_dt'     : DELTA_T,
    'spine_length'  : SPINE_LENGTH,
    'max_duration'  : 60.0,  
    'use_dropouts'  : True,
    'use_los'       : True,
    'use_oor'       : True,
    'verbose'       : True
}

# ==============================================================================
# SECTION 4: RENDERING CONFIGURATION
# ==============================================================================

rendering_config = {
    'aircraft_colors'        : [(1.0, 1.0, 1.0), (0.0, 1.0, 1.0), (1.0, 0.0, 1.0)], #  White,  Cyan,    Magenta
    'sensor_colors'          : [(1.0, 0.5, 0.0), (0.0, 1.0, 0.0), (0.0, 0.5, 1.0)], # Orange, Green, Light Blue
    'aircraft_censor_levels' : ['null', 'null', 'null'],
    'sensor_censor_levels'   : ['null', 'null', 'null'],
    'alpha'                  : 1.0,
    'linewidth'              : 3.0,
    'min_linewidth'          : 0.3,
    'clutter_color_override' : None,
    'MP4_PATH'               : MP4_PATH,
    'DURATION_S'             : simulation_config['max_duration'],
    'SPINE_LENGTH'           : simulation_config['spine_length'],
    'FIGSIZE'                : (16, 9),
    'DPI'                    : 120,
    'FPS'                    : 60,
}

# ==============================================================================
# SECTION 5: EXECUTION
# ==============================================================================

if __name__ == "__main__":
    
    # --- Simulation ---
    print("Configuring and running simulation...")
    simulator = Simulator(**simulation_config)
    simulator.run()

    # --- Rendering ---
    render_animation(
        simulator = simulator,
        config    = rendering_config
    )

