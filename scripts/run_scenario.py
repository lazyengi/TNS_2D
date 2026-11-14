"""
Generate random, vehicle-limited trajectories from a YAML scenario.

The kinematic model is the simulator unicycle. Limits (no reverse, gentle
airspeed, wide turns, ...) are enforced on the commands, per profile.
"""

import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.tns_2d.simulation.scenario_config import build_extended_objects, load_and_sample
from src.tns_2d.simulation.simulator import Simulator


def main():
    parser = argparse.ArgumentParser(description="Traiettorie casuali vincolate, da YAML.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "config" / "scenario.yaml",
        help="File YAML di scenario. Default: config/scenario.yaml",
    )
    args = parser.parse_args()

    scenario = load_and_sample(args.config)
    with scenario.shape_library_path.open("rb") as handle:
        shape_library = pickle.load(handle)
    objects = build_extended_objects(scenario)
    for obj in objects:
        if not 0 <= obj.shape_idx < len(shape_library):
            raise ValueError(
                f"shape_idx {obj.shape_idx} fuori dalla libreria (0..{len(shape_library) - 1})."
            )

    print(
        f"Seed {scenario.seed}. Arena spine {scenario.spine_length:.1f} m, "
        f"durata {scenario.max_duration:.1f} s, dt {scenario.global_dt:.4f} s. "
        f"{len(objects)} mezzi."
    )
    simulator = Simulator(
        extended_objects=objects,
        sensors=[],
        shape_library=shape_library,
        global_dt=scenario.global_dt,
        spine_length=scenario.spine_length,
        max_duration=scenario.max_duration,
        use_dropouts=scenario.use_dropouts,
        use_los=scenario.use_los,
        use_oor=scenario.use_oor,
        verbose=scenario.verbose,
    )
    simulator.run()
    _print_summary(simulator)
    if scenario.tracks_csv is not None:
        _write_tracks(scenario.tracks_csv, simulator)
        print(f"Traiettorie scritte in {scenario.tracks_csv}")
    if scenario.render_enabled:
        _render(simulator, scenario)


def _print_summary(simulator: Simulator) -> None:
    for obj in simulator.extended_objects:
        track = obj.aligned_track
        finite = np.isfinite(track[0])
        if not np.any(finite):
            print(f"  id {obj.identifier} ({obj.profile_name}): nessun campione nel quadrato.")
            continue
        speed = track[3, finite]
        duration = float(np.count_nonzero(finite) * simulator.global_dt)
        print(
            f"  id {obj.identifier} ({obj.profile_name}): {duration:.1f} s nel quadrato, "
            f"velocità {speed.min():.2f}–{speed.max():.2f} m/s."
        )


def _write_tracks(path: Path, simulator: Simulator) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "profile", "t_s", "x_m", "y_m", "heading_rad", "speed_mps"])
        for obj in simulator.extended_objects:
            track = obj.aligned_track
            for index in range(track.shape[1]):
                writer.writerow([
                    obj.identifier,
                    obj.profile_name,
                    f"{simulator.time_grid[index]:.6f}",
                    _cell(track[0, index]),
                    _cell(track[1, index]),
                    _cell(track[2, index]),
                    _cell(track[3, index]),
                ])


def _cell(value: float) -> str:
    if not np.isfinite(value):
        return ""
    return f"{value:.6f}"


def _render(simulator: Simulator, scenario) -> None:
    from src.tns_2d.graphics.rendering_engine import render_animation

    render = scenario.render
    count = len(simulator.extended_objects)
    palette = [
        (1.0, 1.0, 1.0),
        (0.0, 1.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, 0.5, 0.0),
        (0.2, 1.0, 0.2),
        (0.3, 0.5, 1.0),
    ]
    mp4 = render.get("mp4_path", "output/scenario.mp4")
    mp4_path = Path(mp4)
    if not mp4_path.is_absolute():
        mp4_path = scenario.root / mp4_path
    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    figsize = render.get("figsize", [16, 9])
    render_animation(simulator, {
        "extended_object_colors": [palette[i % len(palette)] for i in range(count)],
        "sensor_colors": [],
        "extended_object_censor_levels": ["null"] * count,
        "sensor_censor_levels": [],
        "alpha": 1.0,
        "linewidth": 2.0,
        "min_linewidth": 0.3,
        "clutter_color_override": None,
        "MP4_PATH": mp4_path,
        "DURATION_S": scenario.max_duration,
        "SPINE_LENGTH": scenario.spine_length,
        "FIGSIZE": (float(figsize[0]), float(figsize[1])),
        "DPI": int(render.get("dpi", 80)),
        "FPS": int(render.get("fps", 30)),
        "ENCODER_CRF": 18,
        "ENCODER_PRESET": "veryfast",
        "ENCODER_PIX_FMT": "yuv420p",
    })


if __name__ == "__main__":
    main()
