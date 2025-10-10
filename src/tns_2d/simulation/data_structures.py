"""
================================================================================
data_structures.py
================================================================================

This module defines the core data structures for the TNS-2D simulation project.

It uses Python's `dataclass` feature to create clean, modern, and type-hinted
classes that serve as blueprints for the primary entities in the simulation:
Aircraft and Sensors.

These classes are designed to be simple data containers, separating the "what"
(the data and configuration) from the "how" (the simulation logic, which is
handled by the Simulator class).

--------------------------------------------------------------------------------
Author  : Matteo Tesori (matteotesori@gmail.com)
Date    : 2025-10-09
All wrongs reserved.
--------------------------------------------------------------------------------
"""
import numpy as np
from dataclasses import dataclass, field

@dataclass
class Aircraft:
    """
    Represents a single dynamic object within the simulation environment.

    This class serves as a blueprint for an aircraft, holding all its initial
    configuration parameters (like its flight plan and physical shape) as well as
    the results of the trajectory simulation (raw and time-aligned tracks).
    It is designed as a pure data container; all calculations are performed
    externally by the Simulator.
    """
    # --- Configuration ---
    identifier          : int                                                   # (1) Unique identifier for the aircraft
    shape_idx           : int                                                   # (2) Index to look up the aircraft's shape in a global dictionary
    t_init              : float                                                 # (3) Simulation time at which the aircraft first appears [s]
    initial_position    : np.ndarray                                            # (4) Initial position as a NumPy array [x, y] in meters
    initial_heading     : float                                                 # (5) Initial heading in radians
    initial_speed       : float                                                 # (6) Initial speed in meters per second

    # Flight plan defined by longitudinal and centripetal accelerations over time
    long_vals           : np.ndarray                                            # (7) Values for longitudinal acceleration [m/s^2]
    long_times          : np.ndarray                                            # (8) Times at which longitudinal acceleration changes [s]
    centr_vals          : np.ndarray                                            # (9) Values for centripetal acceleration [m/s^2]
    centr_times         : np.ndarray                                            # (10) Times at which centripetal acceleration changes [s]

    # Simulation time step for this specific aircraft's raw (high-resolution) trajectory
    dt                  : float                                                 # (11) Time step for raw trajectory generation [s]

    # --- Simulation Results (to be populated by the Simulator class) ---
    raw_track           : np.ndarray = field(init = False, default = None)      # High-resolution trajectory, not aligned to global time grid
    raw_time            : np.ndarray = field(init = False, default = None)      # Time vector corresponding to the raw track
    aligned_track       : np.ndarray = field(init = False, default = None)      # Trajectory aligned to the global simulation time grid


@dataclass
class Sensor:
    """
    Represents a single static sensor within the simulation environment.

    This class acts as a blueprint for a measurement device, containing all its
    physical properties (position, field of view), performance characteristics
    (detection probability, clutter rate), and timing parameters. The final
    generated dataset, including all measurements and labels, is stored within
    this object after the simulation is run.
    """
    # --- Configuration ---
    identifier          : int                                                   # (1) Unique identifier for the sensor
    position            : np.ndarray                                            # (2) Sensor's static position as a NumPy array [x, y] in meters
    heading             : float                                                 # (3) Sensor's orientation (boresight direction) in radians

    # Field of View (FoV) parameters
    min_range           : float                                                 # (4) Minimum detection range [m]
    max_range           : float                                                 # (5) Maximum detection range [m]
    min_bearing         : float                                                 # (6) Minimum bearing relative to sensor heading [rad]
    max_bearing         : float                                                 # (7) Maximum bearing relative to sensor heading [rad]

    # Performance Parameters
    pd                  : float                                                 # (8) Probability of Detection (0.0 to 1.0)
    clutter_rate        : float                                                 # (9) Average number of false alarms (clutter) per square meter
    sigma_v             : float                                                 # (10) Standard deviation of measurement noise   [m]
    resolution          : float                                                 # (11) Detection per unit length                 [m^-1]

    # Timing
    dt_sample           : float                                                 # (12) Time between sensor measurements (sampling period)    [s]
    t_offset            : float                                                 # (13) Initial time delay for the first measurement          [s]

    # Dropout model parameters (optional, for stateful dropout)
    alpha               : float = None                                          # Markov chain transition probability for dropouts

    # --- Simulation Results (to be populated by the Simulator class) ---
    dataset             : dict = field(init = False, default_factory = dict)    # Dictionary to store all generated measurements and labels
    dropout_sequence    : np.ndarray = field(init = False, default = None)      # The raw sequence of dropouts (1=active, 0=dropout) over the global time grid

