"""Load a YAML scenario and build the objects the simulator already knows."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .data_structures import ExtendedObject
from .flight_plan import FlightPlan, VehicleProfile, sample_flight_plan


@dataclass
class ResolvedScenario:
    """Scenario after profile arenas and vehicle counts have been applied."""

    seed: int
    root: Path
    spine_length: float
    max_duration: float
    global_dt: float
    use_dropouts: bool
    use_los: bool
    use_oor: bool
    verbose: bool
    shape_library_path: Path
    tracks_csv: Path | None
    render_enabled: bool
    render: dict
    plans: list[FlightPlan]


def load_and_sample(config_path: Path) -> ResolvedScenario:
    """Read ``config_path``, sample one plan per vehicle, return a resolved scenario."""
    config_path = config_path.resolve()
    root = _find_root(config_path)
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    profiles = {
        name: _profile_from_mapping(name, body)
        for name, body in raw["profiles"].items()
    }
    names = _vehicle_names(raw["vehicles"], profiles)
    simulation = raw["simulation"]
    use_profile_arena = bool(simulation.get("use_profile_arena", True))
    if use_profile_arena:
        if len(set(names)) != 1:
            raise ValueError(
                "use_profile_arena: true richiede lo stesso profilo per ogni mezzo. "
                "Per mescolarli imposta use_profile_arena: false e scrivi spine_length, "
                "max_duration e global_dt sotto simulation."
            )
        spine_length = profiles[names[0]].spine_length
        max_duration = profiles[names[0]].max_duration
        global_dt = profiles[names[0]].global_dt
    else:
        spine_length = float(simulation["spine_length"])
        max_duration = float(simulation["max_duration"])
        global_dt = float(simulation["global_dt"])

    seed = int(raw["seed"])
    gap = float(raw["vehicles"].get("t_init_gap", 0.0))
    attempts = int(raw["vehicles"].get("max_attempts", 40))
    plans = []
    occupied: list[np.ndarray] = []
    for index, name in enumerate(names):
        t_init = round((index * gap) / global_dt) * global_dt
        plan = sample_flight_plan(
            profiles[name],
            np.random.default_rng(seed + 1000 * (index + 1)),
            spine_length=spine_length,
            max_duration=max_duration,
            dt=global_dt,
            t_init=t_init,
            occupied=occupied,
            max_attempts=attempts,
        )
        plans.append(plan)
        occupied.append(plan.initial_position.copy())

    export = raw.get("export") or {}
    tracks_value = export.get("tracks_csv", "output/tracks.csv")
    render_raw = raw.get("render") or {}
    return ResolvedScenario(
        seed=seed,
        root=root,
        spine_length=spine_length,
        max_duration=max_duration,
        global_dt=global_dt,
        use_dropouts=bool(simulation.get("use_dropouts", False)),
        use_los=bool(simulation.get("use_los", False)),
        use_oor=bool(simulation.get("use_oor", True)),
        verbose=bool(simulation.get("verbose", True)),
        shape_library_path=_resolve(root, raw.get("shape_library", "data/aircraft_shape_library.pkl")),
        tracks_csv=None if tracks_value in (None, "") else _resolve(root, tracks_value),
        render_enabled=bool(render_raw.get("enabled", False)),
        render=render_raw,
        plans=plans,
    )


def build_extended_objects(scenario: ResolvedScenario) -> list[ExtendedObject]:
    """Turn sampled plans into the data objects ``Simulator`` already integrates."""
    objects = []
    for identifier, plan in enumerate(scenario.plans):
        obj = ExtendedObject(
            identifier=identifier,
            shape_idx=plan.shape_idx,
            t_init=plan.t_init,
            initial_position=plan.initial_position,
            initial_heading=plan.initial_heading,
            initial_speed=plan.initial_speed,
            long_vals=plan.long_vals,
            long_times=plan.long_times,
            centr_vals=plan.centr_vals,
            centr_times=plan.centr_times,
            dt=plan.dt,
        )
        obj.profile_name = plan.profile_name
        objects.append(obj)
    return objects


def _vehicle_names(vehicles: dict, profiles: dict[str, VehicleProfile]) -> list[str]:
    listed = vehicles.get("profiles")
    single = vehicles.get("profile")
    if listed is not None and single is not None:
        raise ValueError("In vehicles usa 'profile' oppure 'profiles', non entrambi.")
    if listed is not None:
        names = [str(name) for name in listed]
        count = vehicles.get("count")
        if count is not None and int(count) != len(names):
            raise ValueError(
                f"vehicles.count è {count} ma vehicles.profiles ne elenca {len(names)}."
            )
    elif single is not None:
        count = int(vehicles.get("count", 1))
        if count < 1:
            raise ValueError("vehicles.count deve essere almeno 1.")
        names = [str(single)] * count
    else:
        raise ValueError("Manca vehicles.profile (o vehicles.profiles).")
    unknown = sorted({name for name in names if name not in profiles})
    if unknown:
        known = ", ".join(sorted(profiles))
        raise ValueError(f"Profili sconosciuti: {', '.join(unknown)}. Definiti: {known}.")
    return names


def _profile_from_mapping(name: str, body: dict) -> VehicleProfile:
    arena = body["arena"]
    allow_reverse = bool(body["allow_reverse"])
    speed_min = float(body["speed_min"])
    speed_max = float(body["speed_max"])
    if speed_max <= speed_min:
        raise ValueError(f"Profilo '{name}': speed_max deve essere maggiore di speed_min.")
    if not allow_reverse and speed_min < 0.0:
        raise ValueError(
            f"Profilo '{name}': allow_reverse è false, quindi speed_min non può essere negativo."
        )
    cruise = float(body["cruise_speed"])
    if not speed_min <= cruise <= speed_max:
        raise ValueError(f"Profilo '{name}': cruise_speed deve stare fra speed_min e speed_max.")
    long_min = float(body["long_accel_min"])
    long_max = float(body["long_accel_max"])
    if not long_min < 0.0 < long_max:
        raise ValueError(
            f"Profilo '{name}': long_accel_min deve essere negativa e long_accel_max positiva, "
            "altrimenti il mezzo non può tenere la velocità."
        )
    n_min = int(body["n_waypoints_min"])
    n_max = int(body["n_waypoints_max"])
    if n_min < 1 or n_max < n_min:
        raise ValueError(f"Profilo '{name}': n_waypoints_min/max non validi.")
    profile = VehicleProfile(
        name=name,
        shape_idx=int(body["shape_idx"]),
        speed_min=speed_min,
        speed_max=speed_max,
        cruise_speed=cruise,
        long_accel_min=long_min,
        long_accel_max=long_max,
        centr_accel_max=float(body["centr_accel_max"]),
        min_turn_radius=float(body["min_turn_radius"]),
        max_yaw_rate=float(body["max_yaw_rate"]),
        n_waypoints_min=n_min,
        n_waypoints_max=n_max,
        min_time_in_arena=float(body["min_time_in_arena"]),
        allow_reverse=allow_reverse,
        reduce_speed_in_turn=bool(body["reduce_speed_in_turn"]),
        spine_length=float(arena["spine_length"]),
        max_duration=float(arena["max_duration"]),
        global_dt=float(arena["global_dt"]),
    )
    if profile.centr_accel_max <= 0.0 or profile.min_turn_radius <= 0.0 or profile.max_yaw_rate <= 0.0:
        raise ValueError(f"Profilo '{name}': centr_accel_max, min_turn_radius e max_yaw_rate devono essere > 0.")
    if profile.spine_length <= 0.0 or profile.max_duration <= 0.0 or profile.global_dt <= 0.0:
        raise ValueError(f"Profilo '{name}': arena non valida.")
    if profile.min_time_in_arena <= 0.0:
        raise ValueError(f"Profilo '{name}': min_time_in_arena deve essere > 0.")
    return profile


def _find_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "src" / "tns_2d").is_dir():
            return candidate
    raise FileNotFoundError(f"Radice del repository non trovata a partire da {start}.")


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path
