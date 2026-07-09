"""
Map region builder: generates or loads map pieces for scenarios.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .scenario_def import MapRegion

# roadgen lives in repo root (parent of framework/)
THIS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
from roadgen import RoadSpec, centerline, rasterize_corridor, write_bg_png, write_pgm  # type: ignore


@dataclass
class MapOutput:
    """Output of map region build."""

    map_pgm_path: str
    map_png_path: Optional[str] = None
    ppm: int = 10
    origin_x_px: float = 0.0
    origin_y_px: float = 0.0
    xmin: float = 0.0
    xmax: float = 0.0
    ymin: float = 0.0
    ymax: float = 0.0
    centerline_pts: Optional[List[Tuple[float, float]]] = None


def build_map_region(
    region: MapRegion,
    output_dir: str,
    scenario_name: str = "scenario",
) -> MapOutput:
    """Build map from MapRegion definition."""
    if region.type == "file" and region.map_file:
        return _build_from_file(region, output_dir, scenario_name)
    return _build_generated(region, output_dir, scenario_name)


def _build_generated(
    region: MapRegion,
    output_dir: str,
    scenario_name: str,
) -> MapOutput:
    spec = RoadSpec(
        shape=region.road_shape,
        width_m=region.road_width_m,
        length_m=region.road_length_m,
        radius_m=region.road_radius_m,
        angle_deg=region.road_angle_deg,
        roundabout_radius_m=region.roundabout_radius_m,
        s_curve_radius_m=region.s_curve_radius_m,
        s_curve_angle_deg=region.s_curve_angle_deg,
    )
    ppm = max(5, region.ppm)
    cl = centerline(spec)
    data, W, H, xmin, xmax, ymin, ymax = rasterize_corridor(cl, spec.width_m, ppm)

    os.makedirs(output_dir, exist_ok=True)
    map_pgm = os.path.join(output_dir, "road_map.pgm")
    map_png = os.path.join(output_dir, "road_map.png")
    write_pgm(map_pgm, W, H, data)
    write_bg_png(map_png, W, H, data)

    origin_x_px = 1.0 - xmin * ppm
    origin_y_px = 1.0 + ymax * ppm

    return MapOutput(
        map_pgm_path=map_pgm,
        map_png_path=map_png,
        ppm=ppm,
        origin_x_px=origin_x_px,
        origin_y_px=origin_y_px,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        centerline_pts=cl,
    )


def _build_from_file(
    region: MapRegion,
    output_dir: str,
    scenario_name: str,
) -> MapOutput:
    map_path = region.map_file
    if not map_path:
        raise FileNotFoundError("map_file not specified")
    if not os.path.isabs(map_path):
        # Resolve relative to scenario-validation root (THIS_DIR = scenario-validation/)
        map_path = os.path.join(THIS_DIR, map_path)
    if not os.path.exists(map_path):
        raise FileNotFoundError(f"Map file not found: {map_path}")

    map_path = os.path.abspath(map_path)
    ppm = int(region.map_pixels_per_meter or 10)
    png_path = map_path

    # Support PNG: convert to PGM for ru_racer (expects PGM occupancy map)
    # Black = obstacle (0), white/light = free (255). Threshold ensures all black areas are obstacles.
    if map_path.lower().endswith(".png"):
        try:
            from PIL import Image
            import numpy as np
            img = Image.open(map_path).convert("L")
            arr = np.array(img)
            w, h = img.size
            # Threshold: pixel < 140 -> obstacle (0), else -> free (255)
            # Black (vehicles, curbs, blocks) = obstacle; white/gray (road, lane lines) = drivable
            obstacle_threshold = int(getattr(region, "obstacle_threshold", 140))  # Black -> obstacle
            occupancy = np.where(arr < obstacle_threshold, 0, 255).astype(np.uint8)
            pgm_path = map_path.replace(".png", ".pgm")
            if not os.path.exists(pgm_path) or os.path.getmtime(pgm_path) < os.path.getmtime(map_path):
                with open(pgm_path, "wb") as f:
                    f.write(f"P5\n{w} {h}\n255\n".encode())
                    f.write(occupancy.tobytes())
            map_path = pgm_path
        except ImportError:
            raise ImportError("PIL/Pillow and numpy required for PNG maps: pip install Pillow numpy")
    else:
        # Bounds from PGM header
        try:
            with open(map_path, "rb") as f:
                if f.readline().decode().strip() == "P5":
                    wh = f.readline().decode().strip().split()
                    w, h = int(wh[0]), int(wh[1])
                else:
                    w, h = 500, 500
        except Exception:
            w, h = 500, 500

    xmax = w / ppm
    ymax = h / ppm
    xmin, ymin = 0.0, 0.0

    return MapOutput(
        map_pgm_path=map_path,
        map_png_path=png_path if png_path != map_path else None,
        ppm=ppm,
        origin_x_px=float(region.map_origin_x_px or 0),
        origin_y_px=float(region.map_origin_y_px or 0),
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
    )
