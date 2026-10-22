#!/usr/bin/env python3
"""
render_fig8.py  -  publication-scale composite of the 12 canonical RRT
scenarios (revised Fig. 8).

Re-runs the scenarios of run_validation_scenarios.py with their original
settings (6,000 nodes, dt = 0.03 s, seed 42), caches each result in
results/fig8_cache/, and draws a 3 x 4 composite with one fixed speed colour
scale, one shared colour bar, and one complete legend.

  ../figs/rrt_validation_12scenarios_rev.png
"""
from __future__ import annotations
import math, os, pickle, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon, Patch
from matplotlib.lines import Line2D

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
import run_validation_scenarios as rvs
import run_rrt_tjunction as rrt

CACHE = os.path.join(THIS_DIR, "results", "fig8_cache")
OUT = os.path.join(THIS_DIR, "..", "figs", "rrt_validation_12scenarios_rev.png")
VMAX = 16.0

LABELS = {
    1: "Wide passage (gap 6.5 m)", 2: "Moderate passage (5.0 m)",
    3: "Tight passage (4.5 m)", 4: "Blocked passage (3.0 m)",
    5: r"Dry, $\mu\approx1.0$", 6: r"Wet, $\mu\approx0.65$",
    7: r"Snow, $\mu\approx0.35$", 8: r"Ice, $\mu\approx0.15$",
    9: "16 m/s (58 km/h)", 10: "12 m/s (43 km/h)",
    11: "8 m/s (29 km/h)", 12: "4 m/s (14 km/h)",
}
ROWS = ["Passage width", "Road friction", "Ego speed"]


def run_all():
    os.makedirs(CACHE, exist_ok=True)
    rvs._ensure_pgm()
    base, xmin, xmax, ymin, ymax = rrt.load_occupancy(rvs.MAP_PGM, rvs.PPM)
    for cfg in rvs.SCENARIOS:
        p = os.path.join(CACHE, f"s{cfg['id']:02d}.pkl")
        if os.path.exists(p):
            continue
        print(f"running scenario {cfg['id']}", flush=True)
        res = rvs.run_scenario(cfg, base, xmin, xmax, ymin, ymax)
        res.pop("grid", None)
        pickle.dump(res, open(p, "wb"))


def spd(n):
    return math.hypot(n[3], n[4])


def panel(ax, res, cmap, norm, bg):
    xmin, xmax, ymin, ymax = res["xmin"], res["xmax"], res["ymin"], res["ymax"]
    ax.imshow(bg, extent=[xmin, xmax, ymin, ymax], origin="upper", alpha=0.85)
    if res["barrier"] is not None:
        bx1, bx2, by1, by2 = res["barrier"]
        ax.add_patch(Polygon([(bx1, by1), (bx2, by1), (bx2, by2), (bx1, by2)],
                             facecolor="black", edgecolor="black", zorder=4))
    ng, eg = res["tree_g"]["nodes"], res["tree_g"]["edges"]
    no, eo = res["tree_o"]["nodes"], res["tree_o"]["edges"]
    sg = {(round(n[0], 3), round(n[1], 3)): spd(n) for n in ng}
    so = {(round(n[0], 3), round(n[1], 3)): spd(n) for n in no}
    seg = [[p1, p2] for p1, p2 in eg]
    col = [cmap(norm(sg.get((round(p2[0], 3), round(p2[1], 3)), 0))) for _, p2 in eg]
    ax.add_collection(LineCollection(seg, colors=col, linewidths=0.9, alpha=0.9, zorder=5))
    seg = [[p1, p2] for p1, p2 in eo]
    col = [cmap(norm(so.get((round(p2[0], 3), round(p2[1], 3)), 0))) for _, p2 in eo]
    ax.add_collection(LineCollection(seg, colors=col, linewidths=0.9, alpha=0.9,
                                     linestyles="dashed", zorder=5))
    path = res["path"]
    if path and len(path) > 1:
        ax.plot([p[0] for p in path], [p[1] for p in path], color="#00c853", lw=3.0, zorder=7)
        step = max(1, len(path) // 8)
        for i in range(0, len(path), step):
            ax.add_patch(Polygon(rrt._vehicle_rect_corners(*path[i]), facecolor="#69f0ae",
                                 edgecolor="#1b5e20", alpha=0.25, lw=0.6, zorder=6))
    n_stop = res["n_stop"]
    if len(no) > n_stop:
        coll = no[n_stop:]
        costs = [rrt._collision_cost(n, xmin, xmax, ymin, ymax) for n in coll]
        n = coll[int(np.argmin(costs))]
        ax.add_patch(Polygon(rrt._vehicle_rect_corners(n[0], n[1], n[2]), facecolor="orange",
                             edgecolor="#e65100", alpha=0.5, lw=1.2, zorder=8))
    s = res["start"]
    ax.plot(s[0], s[1], "o", color="#1b5e20", ms=8, zorder=9)
    ax.plot(*rvs.GOAL, "*", color="red", ms=14, zorder=9)
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymax, ymin)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")


def compose():
    bg = np.array(plt.imread(rvs.MAP_PNG))
    cmap, norm = plt.cm.viridis, plt.Normalize(0, VMAX)
    fig, axs = plt.subplots(3, 4, figsize=(16, 13.6))
    fig.subplots_adjust(left=0.04, right=0.90, top=0.97, bottom=0.08, wspace=0.04, hspace=0.12)
    for cfg in rvs.SCENARIOS:
        i = cfg["id"] - 1
        ax = axs[i // 4, i % 4]
        res = pickle.load(open(os.path.join(CACHE, f"s{cfg['id']:02d}.pkl"), "rb"))
        panel(ax, res, cmap, norm, bg)
        tag = "avoidance path found" if res["path"] else "no avoidance path"
        ax.set_title(f"({chr(97 + i)}) {LABELS[cfg['id']]}\n{tag}", fontsize=13)
    for r, name in enumerate(ROWS):
        axs[r, 0].set_ylabel(name, fontsize=15)
    cax = fig.add_axes([0.915, 0.25, 0.015, 0.5])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label("Node speed (m/s)", fontsize=14); cb.ax.tick_params(labelsize=12)
    h = [Line2D([], [], color="gray", lw=1.5, label="Goal-seeking (avoidance) tree, colour = speed"),
         Line2D([], [], color="gray", lw=1.5, ls="--", label="Fail-safe tree: braking and drift-to-impact branches"),
         Line2D([], [], color="#00c853", lw=3, label="Selected avoidance path (FSM $S_1\\to S_2\\to S_3$)"),
         Patch(facecolor="orange", edgecolor="#e65100", alpha=0.5, label="Minimum-cost impact footprint (FSM $S_2\\to S_4\\to S_5$)"),
         Line2D([], [], marker="o", color="#1b5e20", ls="", ms=8, label="Ego start"),
         Line2D([], [], marker="*", color="red", ls="", ms=12, label="Goal")]
    fig.legend(handles=h, loc="lower center", ncol=3, fontsize=12.5, frameon=True, bbox_to_anchor=(0.47, 0.0))
    fig.savefig(OUT, dpi=200, facecolor="white"); plt.close(fig)
    print("->", OUT)


if __name__ == "__main__":
    run_all()
    compose()
