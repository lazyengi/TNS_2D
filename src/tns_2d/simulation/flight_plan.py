"""
Random flight plans that stay inside a vehicle's kinematic envelope.

The unicycle integrator is unchanged. This module only chooses the initial
state and the accelerations fed to it. Longitudinal acceleration is clipped
so speed stays in ``[speed_min, speed_max]``. Centripetal acceleration is
clipped by the profile ceiling, by ``v² / min_turn_radius`` and by
``max_yaw_rate * v``. With ``allow_reverse: false``, ``speed_min`` is ≥ 0
and the speed never goes below it.
"""

from dataclasses import dataclass

import numpy as np

from .simulator import _zoh_integrator


@dataclass(frozen=True)
class VehicleProfile:
    """Kinematic envelope, plus the arena that envelope was sized for."""

    name: str
    shape_idx: int
    speed_min: float
    speed_max: float
    cruise_speed: float
    long_accel_min: float
    long_accel_max: float
    centr_accel_max: float
    min_turn_radius: float
    max_yaw_rate: float
    n_waypoints_min: int
    n_waypoints_max: int
    min_time_in_arena: float
    allow_reverse: bool
    reduce_speed_in_turn: bool
    spine_length: float
    max_duration: float
    global_dt: float


@dataclass
class FlightPlan:
    """Open-loop plan consumed by ``ExtendedObject``."""

    profile_name: str
    shape_idx: int
    t_init: float
    initial_position: np.ndarray
    initial_heading: float
    initial_speed: float
    long_vals: np.ndarray
    long_times: np.ndarray
    centr_vals: np.ndarray
    centr_times: np.ndarray
    dt: float


def sample_flight_plan(
    profile: VehicleProfile,
    rng: np.random.Generator,
    *,
    spine_length: float,
    max_duration: float,
    dt: float,
    t_init: float = 0.0,
    occupied: list[np.ndarray] | None = None,
    max_attempts: int = 40,
) -> FlightPlan:
    """Draw a plan that respects ``profile`` and remains inside the square long enough."""
    _check_arena(profile, spine_length, max_duration)
    best_time = -1.0
    for _ in range(max_attempts):
        plan, preview, time_vector = _draw_once(
            profile, rng, spine_length, max_duration, dt, t_init, occupied or []
        )
        inside = _time_inside(preview, time_vector, spine_length)
        if inside > best_time:
            best_time = inside
        if inside + 1e-9 >= profile.min_time_in_arena:
            _assert_limits(preview, plan, profile, dt)
            return plan
    raise RuntimeError(
        f"Nessuna traiettoria di '{profile.name}' resta nel quadrato per almeno "
        f"{profile.min_time_in_arena:.1f} s (meglio ottenuto: {best_time:.1f} s in "
        f"{max_attempts} tentativi). Allarga l'arena o allenta speed_max / "
        f"min_turn_radius nel profilo."
    )


def _check_arena(profile: VehicleProfile, spine_length: float, max_duration: float) -> None:
    needed = 1.3 * profile.min_turn_radius
    if spine_length < needed:
        raise ValueError(
            f"Il profilo '{profile.name}' ha raggio di virata minimo "
            f"{profile.min_turn_radius:.0f} m: spine_length deve essere almeno "
            f"{needed:.0f} m, ora è {spine_length:.0f} m. "
            f"Con use_profile_arena: true viene usata l'arena del profilo "
            f"({profile.spine_length:.0f} m)."
        )
    if profile.min_time_in_arena > max_duration + 1e-9:
        raise ValueError(
            f"min_time_in_arena ({profile.min_time_in_arena:.1f} s) supera "
            f"max_duration ({max_duration:.1f} s) per il profilo '{profile.name}'."
        )


def _draw_once(profile, rng, spine_length, max_duration, dt, t_init, occupied):
    margin = min(spine_length * 0.15, max(profile.min_turn_radius * 0.2, 2.0))
    margin = min(margin, spine_length * 0.4)
    lo, hi = -spine_length + margin, spine_length - margin
    n_wp = int(rng.integers(profile.n_waypoints_min, profile.n_waypoints_max + 1))
    min_sep = max(4.0, min(profile.min_turn_radius, hi - lo) * 0.5)
    position = _sample_start(rng, lo, hi, occupied, min_sep)
    cruise_floor = profile.speed_min if profile.speed_min > 0 else min(profile.cruise_speed, profile.speed_max)
    cruise = float(np.clip(profile.cruise_speed * rng.uniform(0.9, 1.1), cruise_floor, profile.speed_max))
    waypoints = _sample_waypoints(rng, n_wp, lo, hi, position, profile, cruise)
    heading = _bearing(position, waypoints[0])

    time_vector = np.arange(0.0, max_duration + dt, dt)
    if len(time_vector) < 2:
        raise ValueError("max_duration deve contenere almeno un passo di integrazione.")

    state = np.zeros((4, len(time_vector)))
    state[:, 0] = (position[0], position[1], _wrap(heading), cruise)
    a_lon = np.zeros(len(time_vector) - 1)
    a_cen = np.zeros(len(time_vector) - 1)
    wp_index = 0
    capture = _capture_radius(profile, spine_length)

    for k in range(len(time_vector) - 1):
        pos = state[0:2, k]
        heading_k = float(state[2, k])
        speed_k = float(state[3, k])
        if np.linalg.norm(waypoints[wp_index] - pos) < capture:
            wp_index = (wp_index + 1) % len(waypoints)
        target, recovering = _select_target(
            pos, heading_k, speed_k, waypoints[wp_index], profile, spine_length
        )
        a_lon[k], a_cen[k] = _accelerations(
            pos, heading_k, speed_k, target, cruise, dt, profile, recovering
        )
        state[:, k + 1], _ = _zoh_integrator(state[:, k], np.array([a_lon[k], a_cen[k]]), dt)

    plan = FlightPlan(
        profile_name=profile.name,
        shape_idx=profile.shape_idx,
        t_init=float(t_init),
        initial_position=state[0:2, 0].copy(),
        initial_heading=float(state[2, 0]),
        initial_speed=float(state[3, 0]),
        long_vals=a_lon,
        long_times=time_vector[:-1].copy(),
        centr_vals=a_cen,
        centr_times=time_vector[:-1].copy(),
        dt=float(dt),
    )
    return plan, state, time_vector


def _accelerations(pos, heading, speed, target, cruise, dt, profile: VehicleProfile, recovering: bool):
    desired = _bearing(pos, target)
    error = _wrap(desired - heading)
    v_floor = profile.speed_min if profile.speed_min > 0.0 else 0.0
    if profile.reduce_speed_in_turn or recovering:
        v_turn = v_floor + 0.45 * (cruise - v_floor)
        blend = max(0.0, np.cos(error))
        v_target = v_turn + (cruise - v_turn) * blend
    else:
        v_target = cruise
    if recovering:
        v_target = min(v_target, max(v_floor, 0.75 * cruise))

    a_lon = float(np.clip(
        (v_target - speed) / max(dt * 4.0, 0.5),
        profile.long_accel_min,
        profile.long_accel_max,
    ))
    a_lim = _centr_limit(speed, profile)
    yaw_max = a_lim / speed if abs(speed) > 1e-3 else 0.0
    yaw_rate = float(np.clip(1.2 * error, -yaw_max, yaw_max))
    a_cen = yaw_rate * speed if abs(speed) > 1e-3 else 0.0
    return _clip_command(speed, a_lon, a_cen, dt, profile)


def _clip_command(speed, a_lon, a_cen, dt, profile: VehicleProfile):
    a_lon = float(np.clip(a_lon, profile.long_accel_min, profile.long_accel_max))
    v_next = speed + dt * a_lon
    if v_next > profile.speed_max:
        a_lon = (profile.speed_max - speed) / dt
    elif v_next < profile.speed_min:
        a_lon = (profile.speed_min - speed) / dt
    a_lon = float(np.clip(a_lon, profile.long_accel_min, profile.long_accel_max))
    a_lim = _centr_limit(speed, profile)
    a_cen = float(np.clip(a_cen, -a_lim, a_lim))
    return a_lon, a_cen


def _centr_limit(speed: float, profile: VehicleProfile) -> float:
    if abs(speed) <= 1e-3:
        return 0.0
    return float(min(
        profile.centr_accel_max,
        (speed * speed) / profile.min_turn_radius,
        profile.max_yaw_rate * abs(speed),
    ))


def _select_target(pos, heading, speed, waypoint, profile: VehicleProfile, spine_length: float):
    """Steer back to the centre before the turn radius no longer fits in the box."""
    along_track = _distance_to_boundary(pos[0], pos[1], heading, spine_length)
    clearance = spine_length - max(abs(pos[0]), abs(pos[1]))
    braking = (speed * speed) / (2.0 * max(abs(profile.long_accel_min), 1e-3))
    margin = min(0.30 * spine_length, max(0.50 * profile.min_turn_radius, braking))
    if clearance < margin or along_track < margin:
        return np.array([0.0, 0.0]), True
    return waypoint, False


def _distance_to_boundary(x: float, y: float, heading: float, spine: float) -> float:
    if abs(x) > spine or abs(y) > spine:
        return 0.0
    direction_x, direction_y = np.cos(heading), np.sin(heading)
    hits = []
    if abs(direction_x) > 1e-12:
        wall_x = spine if direction_x > 0 else -spine
        t_hit = (wall_x - x) / direction_x
        if t_hit > 0 and abs(y + t_hit * direction_y) <= spine + 1e-6:
            hits.append(t_hit)
    if abs(direction_y) > 1e-12:
        wall_y = spine if direction_y > 0 else -spine
        t_hit = (wall_y - y) / direction_y
        if t_hit > 0 and abs(x + t_hit * direction_x) <= spine + 1e-6:
            hits.append(t_hit)
    return float(min(hits)) if hits else 0.0


def _sample_waypoints(rng, count, lo, hi, origin, profile: VehicleProfile, cruise: float):
    """Chain of waypoints. Each leg is a few seconds long, with a real course change.

    Independent points in the box leave consecutive targets on opposite sides,
    and the pursuit between them is a long straight. A chain keeps the next
    turn close.
    """
    span = float(hi - lo)
    # A slow yaw rate needs a long arc to change course. If the next waypoint
    # arrives before that arc is finished, the command stays saturated and the
    # path collapses to a circle. The leg is the arc plus a bit of alignment.
    max_turn = float(np.clip(profile.max_yaw_rate * 18.0, np.deg2rad(25.0), np.deg2rad(90.0)))
    min_turn = 0.5 * max_turn
    arc = float(cruise) * max_turn / profile.max_yaw_rate
    max_leg = min(0.30 * span, max(2.0 * arc, 8.0))
    min_leg = min(max_leg, max(1.15 * arc, 0.55 * max_leg))

    points = []
    pos = np.asarray(origin, dtype=float)
    heading = float(rng.uniform(-np.pi, np.pi))
    for index in range(count):
        pos, heading = _next_waypoint(
            rng, pos, heading, lo, hi, min_leg, max_leg,
            0.0 if index == 0 else min_turn,
            max_turn,
        )
        points.append(pos.copy())
    return np.stack(points)


def _next_waypoint(rng, pos, heading, lo, hi, min_leg, max_leg, min_deflect, max_deflect):
    for attempt in range(48):
        if min_deflect <= 0.0:
            new_heading = heading if attempt == 0 else float(rng.uniform(-np.pi, np.pi))
        else:
            deflect = float(rng.uniform(min_deflect, max_deflect))
            if rng.random() < 0.5:
                deflect = -deflect
            new_heading = _wrap(heading + deflect)
        distance = float(rng.uniform(min_leg, max_leg))
        candidate = pos + distance * np.array([np.cos(new_heading), np.sin(new_heading)])
        if lo <= candidate[0] <= hi and lo <= candidate[1] <= hi:
            return candidate, new_heading
    new_heading = _wrap(_bearing(pos, np.zeros(2)) + float(rng.uniform(-0.8, 0.8)))
    distance = 0.5 * float(rng.uniform(min_leg, max_leg))
    candidate = np.clip(
        pos + distance * np.array([np.cos(new_heading), np.sin(new_heading)]),
        lo,
        hi,
    )
    return candidate, new_heading


def _sample_start(rng, lo, hi, occupied, min_sep):
    for _ in range(80):
        candidate = rng.uniform(lo, hi, size=2)
        if all(np.linalg.norm(candidate - other) >= min_sep for other in occupied):
            return candidate
    return rng.uniform(lo, hi, size=2)


def _capture_radius(profile: VehicleProfile, spine_length: float) -> float:
    return max(8.0, min(spine_length * 0.08, max(profile.min_turn_radius * 0.3, 8.0)))


def _bearing(origin: np.ndarray, point: np.ndarray) -> float:
    delta = np.asarray(point, dtype=float) - np.asarray(origin, dtype=float)
    if np.hypot(delta[0], delta[1]) < 1e-9:
        return 0.0
    return float(np.arctan2(delta[1], delta[0]))


def _wrap(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def _time_inside(track: np.ndarray, time_vector: np.ndarray, spine_length: float) -> float:
    inside = (np.abs(track[0]) <= spine_length) & (np.abs(track[1]) <= spine_length)
    if time_vector.size < 2:
        return 0.0
    dt = float(time_vector[1] - time_vector[0])
    return float(np.count_nonzero(inside) * dt)


def _assert_limits(track: np.ndarray, plan: FlightPlan, profile: VehicleProfile, dt: float) -> None:
    speed = track[3]
    tolerance = 1e-4
    if np.any(speed < profile.speed_min - tolerance) or np.any(speed > profile.speed_max + tolerance):
        raise RuntimeError(
            f"Velocità fuori inviluppo per '{profile.name}': "
            f"[{speed.min():.3f}, {speed.max():.3f}] m/s, limiti "
            f"[{profile.speed_min}, {profile.speed_max}]."
        )
    if not profile.allow_reverse and np.any(speed < -tolerance):
        raise RuntimeError(f"Il profilo '{profile.name}' non ammette retromarcia.")
    if plan.long_vals.shape != plan.centr_vals.shape:
        raise RuntimeError("Comandi longitudinale e centripeta di lunghezze diverse.")
    if np.any(plan.long_vals < profile.long_accel_min - 1e-6) or np.any(plan.long_vals > profile.long_accel_max + 1e-6):
        raise RuntimeError(f"Accelerazione longitudinale fuori inviluppo per '{profile.name}'.")
    integrated = np.diff(speed) / dt
    if np.max(np.abs(integrated - plan.long_vals)) > 1e-6:
        raise RuntimeError(f"La velocità integrata non segue a_lon per '{profile.name}'.")
    for speed_k, a_cen in zip(speed[:-1], plan.centr_vals):
        if abs(a_cen) > _centr_limit(float(speed_k), profile) + 1e-6:
            raise RuntimeError(f"Accelerazione centripeta fuori inviluppo per '{profile.name}'.")
