#!/usr/bin/env python3
"""
Plot planner and controller trajectories on scenario 2D map (X-Y top view).

Works with ru_racer output (rrt_path.csv, nmpc_exec.csv, rrt_nodes.csv) or
synthetic demo trajectories. Shows both dodging and accident (drift-to-mitigate)
trajectories on EMRM Scene 1.

Usage:
  # From ru_racer results:
  python plot_trajectories_on_map.py --run-dir results/run_xxx/out_dry/run_xxx

  # From demo trajectories (no ru_racer needed):
  python plot_trajectories_on_map.py --demo --output figs/
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)


def load_csv(path: str, required_cols: list[str]) -> list[dict]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                rows.append({k: float(row.get(k, 0)) for k in required_cols})
            except (ValueError, TypeError):
                continue
    return rows


def world_to_plot(x: float, y: float, xmin: float, xmax: float, ymin: float, ymax: float) -> tuple[float, float]:
    """Map world (x,y) to plot coordinates. Y is flipped for top-down view."""
    return (x, y)  # Keep as-is for standard X-Y plot


def generate_demo_trajectories(
    start_x: float, start_y: float, goal_x: float, goal_y: float,
    truck_x: float = 60.0, truck_y: float = 0.0, truck_w: float = 4.0, truck_l: float = 8.0,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]], list[tuple[float, float]], list[tuple[float, float]]]:
    """
    Generate synthetic dodge and accident trajectories for EMRM Scene 1.
    Returns (dodge_planner, dodge_nmpc, accident_planner, accident_nmpc).
    Supports both horizontal (tangent) and vertical (T-junction) road layouts.
    """
    dx, dy = goal_x - start_x, goal_y - start_y
    n_pts = 80
    # Dodge: lateral evasion around truck - planner finds path to avoid
    # For vertical road (|dy| > |dx|): dodge in x. For horizontal: dodge in y.
    if abs(dy) > abs(dx):
        # Vertical path (T-junction): dodge laterally in x
        passage_left = truck_x - truck_w / 2 - 2.5
        dodge_planner, dodge_nmpc = [], []
        for i in range(n_pts + 1):
            t = i / n_pts
            y = start_y + t * dy
            x = start_x
            if truck_y - truck_l / 2 - 3 < y < truck_y + truck_l / 2 + 3:
                frac = (y - (truck_y - truck_l / 2 - 3)) / (truck_l + 6)
                frac = max(0, min(1, frac))
                x = start_x + (passage_left - start_x) * (0.5 - 0.5 * math.cos(math.pi * frac))
            dodge_planner.append((x, y))
            dodge_nmpc.append((x - 0.08, y - 0.3))
        dodge_nmpc[-1] = dodge_planner[-1]
    else:
        # Horizontal path (tangent road)
        passage_right = truck_y + truck_w / 2 + 2.5
        dodge_planner, dodge_nmpc = [], []
        for i in range(n_pts + 1):
            t = i / n_pts
            x = start_x + t * dx
            y = start_y
            if truck_x - truck_l / 2 - 5 < x < truck_x + truck_l / 2 + 5:
                frac = (x - (truck_x - truck_l / 2 - 5)) / (truck_l + 10)
                frac = max(0, min(1, frac))
                y = start_y + (passage_right - start_y) * (0.5 - 0.5 * math.cos(math.pi * frac))
            dodge_planner.append((x, y))
            dodge_nmpc.append((x - 0.3, y - 0.08))
        dodge_nmpc[-1] = dodge_planner[-1]

    # Accident (drift-to-mitigate): collision inevitable, lateral drift to reduce impact
    accident_planner, accident_nmpc = [], []
    if abs(dy) > abs(dx):
        impact_y = truck_y - 2.0
        for i in range(n_pts + 1):
            t = i / n_pts
            y = start_y + t * dy
            x = start_x
            if y > impact_y:
                frac = (y - impact_y) / 8.0
                frac = min(1.0, frac)
                x = start_x - 2.0 * frac
            accident_planner.append((x, y))
            accident_nmpc.append((x - 0.05, y - 0.2))
    else:
        impact_x = truck_x - 3.0
        for i in range(n_pts + 1):
            t = i / n_pts
            x = start_x + t * dx
            y = start_y + 2.5 * min(1.0, max(0, (x - impact_x) / 10.0)) if x > impact_x else start_y
            accident_planner.append((x, y))
            accident_nmpc.append((x - 0.2, y - 0.05))
    accident_nmpc[-1] = accident_planner[-1]

    return dodge_planner, dodge_nmpc, accident_planner, accident_nmpc


def write_demo_csvs(out_dir: str, scenario: "ScenarioDef") -> tuple[str, str, str, str]:
    """Write synthetic trajectories in ru_racer format. Returns paths to dodge/accident run dirs."""
    from framework.scenario_def import ScenarioDef
    from framework.map_region import build_map_region

    map_out = build_map_region(scenario.map_region, out_dir, scenario.id)
    # T-junction: truck at intersection center (~38, 27). Start (40,12), goal (40,48).
    # Tangent road: truck at (60, 0).
    truck_x = 38.0 if "tjunction" in scenario.id.lower() else 60.0
    truck_y = 27.0 if "tjunction" in scenario.id.lower() else 0.0

    s = scenario.initial_state
    g = scenario.goal_state
    start = (s.x, s.y)
    goal = (g.x, g.y)

    dodge_planner, dodge_nmpc, acc_planner, acc_nmpc = generate_demo_trajectories(
        start[0], start[1], goal[0], goal[1], truck_x=truck_x, truck_y=truck_y
    )

    def write_path_csv(path: str, pts: list, cols: list[str], ref_pts: list = None):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            ref = ref_pts or pts
            for i, (x, y) in enumerate(pts):
                row = {"x": x, "y": y}
                if "x_ref" in cols and i < len(ref):
                    row["x_ref"] = ref[i][0]
                    row["y_ref"] = ref[i][1]
                if "vx" in cols:
                    row["vx"] = 12.0
                    row["vy"] = 0.0
                w.writerow(row)

    dodge_dir = os.path.join(out_dir, "demo_dodge")
    acc_dir = os.path.join(out_dir, "demo_accident")
    os.makedirs(dodge_dir, exist_ok=True)
    os.makedirs(acc_dir, exist_ok=True)

    write_path_csv(os.path.join(dodge_dir, "rrt_path.csv"), dodge_planner, ["x", "y"])
    write_path_csv(os.path.join(dodge_dir, "nmpc_exec.csv"), dodge_nmpc, ["x", "y", "x_ref", "y_ref", "vx", "vy"], ref_pts=dodge_planner)
    write_path_csv(os.path.join(acc_dir, "rrt_path.csv"), acc_planner, ["x", "y"])
    write_path_csv(os.path.join(acc_dir, "nmpc_exec.csv"), acc_nmpc, ["x", "y", "x_ref", "y_ref", "vx", "vy"], ref_pts=acc_planner)

    return map_out.map_png_path or map_out.map_pgm_path, dodge_dir, acc_dir, (map_out.xmin, map_out.xmax, map_out.ymin, map_out.ymax)


def plot_trajectories(
    map_path: str,
    run_dirs: list[tuple[str, str]],  # (label, path)
    bounds: tuple[float, float, float, float],
    output_path: str,
    start_goal: tuple[float, float, float, float] | None = None,
    truck_rect: tuple[float, float, float, float] | None = None,
):
    """Plot planner and NMPC trajectories on map. run_dirs: [(label, dir), ...]."""
    import numpy as np
    import matplotlib.pyplot as plt

    xmin, xmax, ymin, ymax = bounds
    img = None
    if map_path and os.path.exists(map_path):
        try:
            from PIL import Image
            img = np.array(Image.open(map_path))
        except Exception:
            try:
                img = plt.imread(map_path)
            except Exception:
                pass

    fig, ax = plt.subplots(figsize=(8, 5))
    if img is not None:
        ax.imshow(img, extent=[xmin, xmax, ymin, ymax], origin="upper", cmap="gray", alpha=0.8)
    else:
        ax.set_facecolor("#e8e8e8")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("EMRM Scene 1: Planner and Controller Trajectories (X-Y Top View)")

    # Truck/obstacle overlay (optional; T-junction map has vehicles in image)
    if truck_rect:
        tx1, ty1, tx2, ty2 = truck_rect
        from matplotlib.patches import Rectangle
        ax.add_patch(Rectangle((tx1, ty1), tx2 - tx1, ty2 - ty1, facecolor="#8B4513", edgecolor="black", linewidth=1.5, alpha=0.8, label="Obstacle (truck)"))
        ax.text((tx1 + tx2) / 2, (ty1 + ty2) / 2, "TRUCK", ha="center", va="center", fontsize=9, color="white", fontweight="bold")

    colors = {"Dodge": "#2ecc71", "Accident (Drift)": "#e74c3c"}
    for label, run_dir in run_dirs:
        color = colors.get(label, "#3498db")
        rrt_path = os.path.join(run_dir, "rrt_path.csv")
        nmpc_path = os.path.join(run_dir, "nmpc_exec.csv")
        planner_pts = load_csv(rrt_path, ["x", "y"])
        nmpc_pts = load_csv(nmpc_path, ["x", "y"])
        if planner_pts:
            xs = [p["x"] for p in planner_pts]
            ys = [p["y"] for p in planner_pts]
            ax.plot(xs, ys, "-", color=color, linewidth=2.5, label=f"{label} (planner)")
        if nmpc_pts:
            xs = [p["x"] for p in nmpc_pts]
            ys = [p["y"] for p in nmpc_pts]
            ax.plot(xs, ys, "--", color=color, linewidth=1.5, alpha=0.8, label=f"{label} (NMPC)")

    if start_goal:
        sx, sy, gx, gy = start_goal
        ax.plot(sx, sy, "go", markersize=10, label="Start")
        ax.plot(gx, gy, "r*", markersize=12, label="Goal")

    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    ap = argparse.ArgumentParser(description="Plot planner/controller trajectories on scenario map")
    ap.add_argument("--run-dir", help="Single ru_racer run directory")
    ap.add_argument("--run-dirs", nargs="+", help="Multiple run dirs: label1 path1 label2 path2 ...")
    ap.add_argument("--demo", action="store_true", help="Generate and plot demo trajectories (no ru_racer)")
    ap.add_argument("--output", default="figs", help="Output path for figure")
    ap.add_argument("--scenario", default="scenarios/EMRM_Scene1_tjunction.yaml", help="Scenario YAML for demo")
    args = ap.parse_args()

    if args.demo:
        from framework.scenario_def import ScenarioDef, MapRegion, VehicleState
        scenario_path = os.path.join(THIS_DIR, args.scenario)
        try:
            scenario = ScenarioDef.from_yaml(scenario_path)
        except (ImportError, FileNotFoundError):
            # Fallback: EMRM Scene1 params (straight road, truck blocking)
            scenario = ScenarioDef(
                id="EMRM_Scene1_tjunction",
                name="EMRM Scene 1",
                map_region=MapRegion(road_shape="tangent", road_length_m=120, road_width_m=8, ppm=10),
                initial_state=VehicleState(x=0, y=0, psi=0, vx=15, vy=0, r=0),
                goal_state=VehicleState(x=100, y=0, psi=0, vx=10, vy=0, r=0),
            )
        out_dir = os.path.join(THIS_DIR, "results", "demo_trajectories")
        map_path, dodge_dir, acc_dir, bounds = write_demo_csvs(out_dir, scenario)
        s, g = scenario.initial_state, scenario.goal_state
        start_goal = (s.x, s.y, g.x, g.y)
        run_dirs = [("Dodge", dodge_dir), ("Accident (Drift)", acc_dir)]
        os.makedirs(args.output, exist_ok=True)
        out_path = os.path.join(args.output, "emrm_trajectories_map.png")
        plot_trajectories(map_path, run_dirs, bounds, out_path, start_goal=start_goal)
        return 0

    if args.run_dir:
        run_dir = args.run_dir
        if not os.path.isabs(run_dir):
            run_dir = os.path.join(THIS_DIR, run_dir)
        # Need map - look in parent
        parent = os.path.dirname(os.path.dirname(run_dir))
        map_path = os.path.join(parent, "road_map.png")
        if not os.path.exists(map_path):
            map_path = os.path.join(parent, "road_map.pgm")
        bounds = (-5, 125, -15, 15)  # Default for straight road
        run_dirs = [("Trajectory", run_dir)]
        out_path = os.path.join(args.output, "emrm_trajectories_map.png")
        os.makedirs(args.output, exist_ok=True)
        plot_trajectories(map_path, run_dirs, bounds, out_path)
        return 0

    if args.run_dirs and len(args.run_dirs) >= 4:
        run_dirs = [(args.run_dirs[i], args.run_dirs[i + 1]) for i in range(0, len(args.run_dirs), 2)]
        map_path = ""
        bounds = (-5, 125, -15, 15)
        out_path = os.path.join(args.output, "emrm_trajectories_map.png")
        os.makedirs(args.output, exist_ok=True)
        plot_trajectories(map_path, run_dirs, bounds, out_path)
        return 0

    print("Use --demo for synthetic trajectories, or --run-dir for ru_racer results")
    return 1


if __name__ == "__main__":
    sys.exit(main())
