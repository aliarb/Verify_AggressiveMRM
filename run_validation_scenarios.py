#!/usr/bin/env python3
"""
run_validation_scenarios.py
Generate 12 RRT T-junction validation figures.

  Groups (4 scenarios each):
    1–4   Passage width  — right-side barrier narrows the gap to the goal lane
    5–8   Road friction  — Pacejka D parameter scaled for dry / wet / snow / ice
    9–12  Ego speed      — initial longitudinal velocity 16 / 12 / 8 / 4 m/s

Usage:
    cd Simulations/scenario-validation
    python run_validation_scenarios.py
"""

from __future__ import annotations
import math, os, random, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import LineCollection
import matplotlib.cm as mcm

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import run_rrt_tjunction as rrt
import ru_racer_dynamics as rrd

# ── paths ──────────────────────────────────────────────────────────────────
MAP_PNG  = os.path.join(THIS_DIR, "maps", "tjunction_map.png")
MAP_PGM  = os.path.join(THIS_DIR, "maps", "tjunction_map.pgm")
FIGS_DIR = os.path.join(THIS_DIR, "..", "..", "figs")
PPM      = 10.0

# ── base dynamics constants (captured at import time) ──────────────────────
_B0_F = float(rrd.TIRE_F["B"]); _C0_F = float(rrd.TIRE_F["C"]); _D0_F = float(rrd.TIRE_F["D"])
_B0_R = float(rrd.TIRE_R["B"]); _C0_R = float(rrd.TIRE_R["C"]); _D0_R = float(rrd.TIRE_R["D"])
_LAT0 = float(rrt.MAX_LATERAL_ACCEL)

BASE_START = (41.8, 5.2, 1.57, 12.0)   # (x, y, psi, v)
GOAL       = (44.0, 48.0)

# Effective collision width of the vehicle (used in passage labels)
EFF_VEH_W = rrt.VEH_WIDTH * rrt.VEH_SCALE * rrt.VEH_WIDTH_SCALE   # ≈ 3.96 m

# ── passage-width barrier (right-side of the goal lane, in world metres) ───
#   The barrier runs x = [_BAR_X2 - obs_w, _BAR_X2], y = [_BAR_Y1, _BAR_Y2]
#   and is added on top of the existing truck obstacle.
#   Approximate truck right edge ≈ 40 m → current passage ≈ 6.5 m without barrier.
_BAR_X2 = 46.5   # right road edge (m)
_BAR_Y1 = 27.0   # top of junction choke zone (m)
_BAR_Y2 = 40.0   # bottom of junction choke zone (m)

# ── 12 scenario configs ────────────────────────────────────────────────────
#   Keys: id, group, title, v0, df (front Pacejka D), dr (rear Pacejka D),
#         obs_w (barrier width in m), outfile
_DEF = dict(v0=12.0, df=_D0_F, dr=_D0_R, obs_w=0.0)

SCENARIOS: list[dict] = [
    # ── 1-4: passage width ─────────────────────────────────────────────────
    {**_DEF, "id": 1, "group": "Passage Width",
     "title": (
         "Scenario 1 — Passage Width: Wide\n"
         f"≈2× vehicle width  |  gap ≈6.5 m  |  eff. vehicle width = {EFF_VEH_W:.1f} m"
     ),
     "obs_w": 0.0,
     "outfile": "rrt_scenario_01_passage_2x.png"},

    {**_DEF, "id": 2, "group": "Passage Width",
     "title": (
         "Scenario 2 — Passage Width: Moderate\n"
         "≈1.5× vehicle width  |  gap ≈5.0 m"
     ),
     "obs_w": 1.5,
     "outfile": "rrt_scenario_02_passage_1p5x.png"},

    {**_DEF, "id": 3, "group": "Passage Width",
     "title": (
         "Scenario 3 — Passage Width: Tight\n"
         "≈1.1× vehicle width  |  gap ≈4.5 m"
     ),
     "obs_w": 2.0,
     "outfile": "rrt_scenario_03_passage_1p1x.png"},

    {**_DEF, "id": 4, "group": "Passage Width",
     "title": (
         "Scenario 4 — Passage Width: Blocked\n"
         "<1× vehicle width  |  gap ≈3.0 m  →  fail-safe only"
     ),
     "obs_w": 3.5,
     "outfile": "rrt_scenario_04_passage_blocked.png"},

    # ── 5-8: road friction ─────────────────────────────────────────────────
    {**_DEF, "id": 5, "group": "Friction",
     "title": "Scenario 5 — Friction: High (dry road)  μ≈1.0",
     "df": _D0_F,
     "dr": _D0_R,
     "outfile": "rrt_scenario_05_friction_high.png"},

    {**_DEF, "id": 6, "group": "Friction",
     "title": "Scenario 6 — Friction: Medium (wet road)  μ≈0.65",
     "df": _D0_F * 0.65,
     "dr": _D0_R * 0.65,
     "outfile": "rrt_scenario_06_friction_med.png"},

    {**_DEF, "id": 7, "group": "Friction",
     "title": "Scenario 7 — Friction: Low (snow / slush)  μ≈0.35",
     "df": _D0_F * 0.35,
     "dr": _D0_R * 0.35,
     "outfile": "rrt_scenario_07_friction_low.png"},

    {**_DEF, "id": 8, "group": "Friction",
     "title": "Scenario 8 — Friction: Very Low (ice)  μ≈0.15",
     "df": _D0_F * 0.15,
     "dr": _D0_R * 0.15,
     "outfile": "rrt_scenario_08_friction_verylow.png"},

    # ── 9-12: ego speed ────────────────────────────────────────────────────
    {**_DEF, "id": 9, "group": "Ego Speed",
     "title": "Scenario 9 — Ego Speed: Very High  16 m/s (≈58 km/h)",
     "v0": 16.0,
     "outfile": "rrt_scenario_09_speed_veryhigh.png"},

    {**_DEF, "id": 10, "group": "Ego Speed",
     "title": "Scenario 10 — Ego Speed: High  12 m/s (≈43 km/h)  [baseline]",
     "v0": 12.0,
     "outfile": "rrt_scenario_10_speed_high.png"},

    {**_DEF, "id": 11, "group": "Ego Speed",
     "title": "Scenario 11 — Ego Speed: Medium  8 m/s (≈29 km/h)",
     "v0": 8.0,
     "outfile": "rrt_scenario_11_speed_med.png"},

    {**_DEF, "id": 12, "group": "Ego Speed",
     "title": "Scenario 12 — Ego Speed: Low  4 m/s (≈14 km/h)",
     "v0": 4.0,
     "outfile": "rrt_scenario_12_speed_low.png"},
]


# ── helpers ────────────────────────────────────────────────────────────────

def _ensure_pgm() -> None:
    """Regenerate MAP_PGM from MAP_PNG if PNG is newer or PGM is missing."""
    if not os.path.exists(MAP_PNG):
        return
    try:
        from PIL import Image
        img = Image.open(MAP_PNG).convert("L")
        arr = np.array(img)
        h, w = arr.shape
        data = np.where(arr < 50, 0, 255).astype(np.uint8)
        with open(MAP_PGM, "wb") as f:
            f.write(f"P5\n{w} {h}\n255\n".encode())
            f.write(data.tobytes())
    except ImportError:
        pass  # PIL not available; rely on existing PGM


def _add_obstacle_rect(
    grid: np.ndarray, ymax: float, ppm: float,
    wx1: float, wx2: float, wy1: float, wy2: float,
) -> np.ndarray:
    """Return a copy of *grid* with rectangle [wx1,wx2]×[wy1,wy2] set to obstacle (0)."""
    g = grid.copy()
    h, w = g.shape
    c1 = max(0, int(math.floor(wx1 * ppm)))
    c2 = min(w, int(math.ceil(wx2 * ppm)))
    # image row 0 = world y=ymax (origin upper)
    r1 = max(0, int(math.floor((ymax - wy2) * ppm)))
    r2 = min(h, int(math.ceil((ymax - wy1) * ppm)))
    g[r1:r2, c1:c2] = 0
    return g


def _smooth_path(path: list, start_psi: float):
    """B-spline smooth of path [(x,y,psi)]. Returns (xy_list, psi_list)."""
    if not path or len(path) < 3:
        return [(p[0], p[1]) for p in path], [p[2] for p in path]
    try:
        from scipy.interpolate import splprep, splev
        arr = np.array([(p[0], p[1]) for p in path])
        tck, _ = splprep([arr[:, 0], arr[:, 1]], s=2.0, k=min(3, len(path) - 1))
        u_s = np.linspace(0, 1, max(50, len(path) * 2))
        xs, ys = splev(u_s, tck)
        pts = list(zip(xs, ys))
        psis: list[float] = []
        for i in range(len(pts)):
            if i + 1 < len(pts):
                psis.append(math.atan2(pts[i + 1][1] - pts[i][1],
                                       pts[i + 1][0] - pts[i][0]))
            else:
                psis.append(psis[-1] if psis else start_psi)
        return pts, psis
    except Exception:
        return [(p[0], p[1]) for p in path], [p[2] for p in path]


def _node_spd(n: tuple) -> float:
    return math.sqrt(n[3] * n[3] + n[4] * n[4])


# ── core run ───────────────────────────────────────────────────────────────

def run_scenario(
    cfg: dict,
    base_grid: np.ndarray,
    xmin: float, xmax: float, ymin: float, ymax: float,
) -> dict:
    """
    Patch module globals, modify grid, run RRT, restore globals.
    Returns a result dict passed straight to plot_scenario().
    """
    # patch tire friction
    rrd.TIRE_F = {"B": _B0_F, "C": _C0_F, "D": cfg["df"]}
    rrd.TIRE_R = {"B": _B0_R, "C": _C0_R, "D": cfg["dr"]}

    # passage barrier
    grid = base_grid.copy()
    obs_w = cfg["obs_w"]
    barrier: tuple | None = None
    if obs_w > 0.0:
        bx1 = _BAR_X2 - obs_w
        bx2 = _BAR_X2
        barrier = (bx1, bx2, _BAR_Y1, _BAR_Y2)
        grid = _add_obstacle_rect(grid, ymax, PPM, bx1, bx2, _BAR_Y1, _BAR_Y2)

    start = (BASE_START[0], BASE_START[1], BASE_START[2], float(cfg["v0"]))
    random.seed(42)
    tree_g, tree_o, path, n_drained, n_stop = rrt.rrt_double_tree(
        start, GOAL, grid, xmin, xmax, ymin, ymax, PPM,
        max_nodes=6000, dt=0.03, goal_sample_rate=0.55,
    )

    # restore globals
    rrd.TIRE_F = {"B": _B0_F, "C": _C0_F, "D": _D0_F}
    rrd.TIRE_R = {"B": _B0_R, "C": _C0_R, "D": _D0_R}
    rrt.MAX_LATERAL_ACCEL = _LAT0

    return dict(
        cfg=cfg, start=start, grid=grid,
        xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
        tree_g=tree_g, tree_o=tree_o,
        path=path, n_stop=n_stop,
        barrier=barrier,
    )


# ── plotting ───────────────────────────────────────────────────────────────

def plot_scenario(res: dict) -> None:
    cfg     = res["cfg"]
    start   = res["start"]
    xmin    = res["xmin"]; xmax = res["xmax"]
    ymin    = res["ymin"]; ymax = res["ymax"]
    ng      = res["tree_g"]["nodes"]
    eg      = res["tree_g"]["edges"]
    no      = res["tree_o"]["nodes"]
    eo      = res["tree_o"]["edges"]
    path    = res["path"]
    n_stop  = res["n_stop"]
    barrier = res["barrier"]

    path_smooth, path_smooth_psi = _smooth_path(path, start[2])

    # speed colormap over all nodes
    all_spds = [_node_spd(n) for n in ng] + [_node_spd(n) for n in no]
    v_min = min(all_spds) if all_spds else 0.0
    v_max = max(all_spds) if all_spds else 1.0
    if v_max <= v_min:
        v_max = v_min + 1e-6
    cmap = plt.cm.viridis

    def _nv(v: float) -> float:
        return (v - v_min) / (v_max - v_min)

    def _pk(x: float, y: float) -> tuple:
        return (round(x, 3), round(y, 3))

    spd_g = {_pk(n[0], n[1]): _node_spd(n) for n in ng}
    spd_o = {_pk(n[0], n[1]): _node_spd(n) for n in no}

    def _spd_at(p: tuple, green: bool = True) -> float:
        d = spd_g if green else spd_o
        k = _pk(p[0], p[1])
        if k in d:
            return d[k]
        best, bd = v_min, float("inf")
        for kk, vv in d.items():
            dd = (p[0] - kk[0]) ** 2 + (p[1] - kk[1]) ** 2
            if dd < bd:
                bd, best = dd, vv
        return best

    # ── figure ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 10))

    if os.path.exists(MAP_PNG):
        bg = np.array(plt.imread(MAP_PNG))
        ax.imshow(bg, extent=[xmin, xmax, ymin, ymax], origin="upper", alpha=0.9)
    else:
        ax.imshow(res["grid"], extent=[xmin, xmax, ymin, ymax],
                  origin="upper", cmap="gray", alpha=0.8)

    # draw barrier as filled black polygon (world coordinates)
    if barrier is not None:
        bx1, bx2, by1, by2 = barrier
        bar_corners = [(bx1, by1), (bx2, by1), (bx2, by2), (bx1, by2)]
        ax.add_patch(Polygon(
            bar_corners, closed=True,
            facecolor="black", edgecolor="#333333",
            linewidth=1.2, alpha=0.90, zorder=4,
        ))
        # annotate barrier width
        ax.annotate(
            f"barrier {bx2 - bx1:.1f} m",
            xy=((bx1 + bx2) / 2, (by1 + by2) / 2),
            fontsize=7, ha="center", va="center", color="white", zorder=5,
        )

    # branch-start positions of orange tree (to skip in green tree)
    o_starts = {(p1[0], p1[1]) for p1, _ in eo}

    def _is_o_start(p: tuple) -> bool:
        for ox, oy in o_starts:
            if abs(p[0] - ox) < 0.05 and abs(p[1] - oy) < 0.05:
                return True
        return False

    # green tree edges
    segs_g: list = []
    cols_g: list = []
    for (p1, p2) in eg:
        if _is_o_start(p2):
            continue
        v = _spd_at(p2, True)
        segs_g.append([[p1[0], p1[1]], [p2[0], p2[1]]])
        cols_g.append(cmap(_nv(v)))
    if segs_g:
        ax.add_collection(LineCollection(segs_g, colors=cols_g, linewidths=0.6, alpha=0.85))
    if ng:
        gx = [n[0] for n in ng]; gy = [n[1] for n in ng]
        gc = [cmap(_nv(_node_spd(n))) for n in ng]
        ax.scatter(gx, gy, c=gc, s=10, alpha=0.9, edgecolors="none")

    # orange tree edges
    segs_o: list = []
    cols_o: list = []
    for (p1, p2) in eo:
        v = _spd_at(p2, False)
        segs_o.append([[p1[0], p1[1]], [p2[0], p2[1]]])
        cols_o.append(cmap(_nv(v)))
    if segs_o:
        ax.add_collection(LineCollection(segs_o, colors=cols_o, linewidths=0.6, alpha=0.85))
    if no:
        ox_ = [n[0] for n in no]; oy_ = [n[1] for n in no]
        oc  = [cmap(_nv(_node_spd(n))) for n in no]
        ax.scatter(ox_, oy_, c=oc, s=14, alpha=0.9, edgecolors="none")

    # min-cost collision marker on orange tree
    if len(no) > n_stop:
        coll_nodes = no[n_stop:]
        costs = [rrt._collision_cost(n, xmin, xmax, ymin, ymax) for n in coll_nodes]
        min_c = min(costs)
        for n, c in zip(coll_nodes, costs):
            if c == min_c:
                x, y, psi, vx, vy, omega = n[0], n[1], n[2], n[3], n[4], n[5]
                corners = rrt._vehicle_rect_corners(x, y, psi)
                ax.add_patch(Polygon(corners, facecolor="orange", edgecolor="darkorange",
                                     alpha=0.35, linewidth=1.5))
                ax.plot(x, y, "o", color="orange", markersize=5,
                        label="Min-cost collision (prefer object)")
                v_spd = math.sqrt(vx * vx + vy * vy)
                if v_spd >= 0.1:
                    dx_, dy_ = vx * 0.5, vy * 0.5
                    ax.annotate("", xy=(x + dx_, y + dy_), xytext=(x, y),
                                arrowprops=dict(arrowstyle="->", color="darkred",
                                                lw=1.5, alpha=0.9))
                break

    # legend proxies
    ax.plot([], [], "-", color="green",   linewidth=2, label="Normal (goal)")
    ax.plot([], [], "-", color="orange",  linewidth=2, label="Fail-safe (stop + collision)")
    ax.plot([], [], "-", color="darkred", linewidth=2, label="Velocity (min-cost)")

    # best path raw
    p_spds: list[float] = []
    if path and len(path) > 1:
        p_spds = [_spd_at((p[0], p[1]), True) for p in path]
        segs_p = [[[path[i][0], path[i][1]], [path[i + 1][0], path[i + 1][1]]]
                  for i in range(len(path) - 1)]
        lc_p = LineCollection(segs_p,
                              colors=[cmap(_nv(p_spds[i + 1])) for i in range(len(path) - 1)],
                              linewidths=1.2, alpha=0.7)
        ax.add_collection(lc_p)

    # smoothed best path + vehicle boxes
    if path_smooth and len(path_smooth) > 1:
        if p_spds:
            p_spds2 = [
                p_spds[min(int(round(i / max(1, len(path_smooth) - 1) * (len(p_spds) - 1))),
                           len(p_spds) - 1)]
                for i in range(len(path_smooth))
            ]
        else:
            p_spds2 = [v_min] * len(path_smooth)

        segs_s = [[[path_smooth[i][0], path_smooth[i][1]],
                   [path_smooth[i + 1][0], path_smooth[i + 1][1]]]
                  for i in range(len(path_smooth) - 1)]
        lc_s = LineCollection(segs_s,
                              colors=[cmap(_nv(p_spds2[i + 1])) for i in range(len(path_smooth) - 1)],
                              linewidths=3, alpha=0.9)
        ax.add_collection(lc_s)

        # vehicle footprint boxes along path
        step = max(1, len(path_smooth) // 12)
        for i in range(0, len(path_smooth), step):
            px_, py_ = path_smooth[i]
            pp_ = path_smooth_psi[i] if i < len(path_smooth_psi) else start[2]
            corners_ = rrt._vehicle_rect_corners(px_, py_, pp_)
            ax.add_patch(Polygon(corners_, facecolor="lime", edgecolor="darkgreen",
                                 alpha=0.25, linewidth=0.8))
    else:
        print(f"  [!] No goal path found — only fail-safe tree shown.")

    # start / goal markers
    psi0 = start[2]
    ax.plot(start[0], start[1], "go", markersize=12, label="Start")
    ax.annotate("", xy=(start[0] + 2 * math.cos(psi0), start[1] + 2 * math.sin(psi0)),
                xytext=(start[0], start[1]),
                arrowprops=dict(arrowstyle="->", color="green", lw=2))
    ax.annotate("Initial state",
                xy=(start[0], start[1] - 3), fontsize=9, ha="center",
                color="darkgreen", weight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="darkgreen", alpha=0.9))
    ax.plot(GOAL[0], GOAL[1], "r*", markersize=14, label="Goal")

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)   # y=0 at top (image convention)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(cfg["title"], fontsize=10, pad=8)
    ax.legend(loc="upper left", fontsize=8)

    sm = mcm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=v_min, vmax=v_max))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
    cbar.set_label("Speed (m/s)")

    plt.tight_layout()
    out = os.path.join(FIGS_DIR, cfg["outfile"])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved → {out}")


# ── main ───────────────────────────────────────────────────────────────────

def main() -> int:
    _ensure_pgm()
    if not os.path.exists(MAP_PGM):
        print("ERROR: map PGM not found — run from Simulations/scenario-validation/")
        return 1

    base_grid, xmin, xmax, ymin, ymax = rrt.load_occupancy(MAP_PGM, PPM)
    print(f"Map loaded: {xmax:.1f} × {ymax:.1f} m  |  grid {base_grid.shape}")
    print(f"Effective vehicle width: {EFF_VEH_W:.2f} m")
    print(f"Running {len(SCENARIOS)} scenarios...\n")

    for cfg in SCENARIOS:
        print(f"[{cfg['id']:02d}/12]  {cfg['group']}: {cfg['title'].splitlines()[0]}")
        res = run_scenario(cfg, base_grid, xmin, xmax, ymin, ymax)
        ng_n  = len(res["tree_g"]["nodes"])
        no_n  = len(res["tree_o"]["nodes"])
        n_stp = res["n_stop"]
        n_pth = len(res["path"])
        print(f"        green={ng_n} nodes | orange={no_n} "
              f"({n_stp} stop, {no_n - n_stp} collision) | path={n_pth} nodes")
        plot_scenario(res)
        print()

    print(f"All {len(SCENARIOS)} figures written to:  {os.path.abspath(FIGS_DIR)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
