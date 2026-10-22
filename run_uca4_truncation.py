#!/usr/bin/env python3
"""
run_uca4_truncation.py  -  test of STPA UCA4 (EMRM stopped too soon).

For every ego speed of the fine grid, the cell midway (in TTC) between the
empirical avoidance boundary and the stop boundary is re-planned with the
same RRT settings as run_coverage_fine.py (seed 42).  The avoidance path is
then terminated prematurely after a fraction f of its nodes, after which the
vehicle brakes in a straight line (same braking branch as the RRT fail-safe
tree).  A collision of the vehicle footprint during this braking segment
means the premature termination re-creates the hazard.

  results/uca4_truncation.csv
"""
from __future__ import annotations
import csv, math, os, random, sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
import run_rrt_tjunction as rrt

RES = os.path.join(THIS_DIR, "results")
MAP_PGM = os.path.join(THIS_DIR, "maps", "tjunction_map.pgm")
PPM, A_BRAKE = 10.0, 6.0
TRUCK_Y, START_X, START_PSI, GOAL = 27.0, 41.8, 1.57, (44.0, 48.0)
FRACS = [0.25, 0.50, 0.75]


def pick_cells():
    rows = list(csv.DictReader(open(os.path.join(RES, "coverage_fine.csv"))))
    cells = []
    for vk in range(30, 73):
        ys = sorted(float(r["TTC_s"]) for r in rows
                    if int(r["v_kmh"]) == vk and r["has_path"] == "True"
                    and float(r["TTC_s"]) < vk / 3.6 / (2 * A_BRAKE))
        if ys:
            cells.append((vk, ys[len(ys) // 2]))
    return cells


def main():
    grid, xmin, xmax, ymin, ymax = rrt.load_occupancy(MAP_PGM, PPM)
    out = open(os.path.join(RES, "uca4_truncation.csv"), "w", newline="")
    w = csv.writer(out)
    w.writerow(["v_kmh", "TTC_s", "path_nodes", "frac", "collision", "impact_v"])
    for vk, ttc in pick_cells():
        v = vk / 3.6
        random.seed(42)
        tg, to, path, _, _ = rrt.rrt_double_tree(
            (START_X, TRUCK_Y - v * ttc, START_PSI, v), GOAL, grid, xmin, xmax, ymin, ymax, PPM,
            max_nodes=600, dt=0.04, goal_sample_rate=0.55)
        if not path:
            continue
        full = {(round(n[0], 3), round(n[1], 3)): n for n in tg["nodes"]}
        for f in FRACS:
            k = max(1, int(f * (len(path) - 1)))
            n = full.get((round(path[k][0], 3), round(path[k][1], 3)))
            if n is None:
                continue
            from ru_racer_dynamics import ru_racer_propagate
            s = tuple(n)
            hit, iv = False, 0.0
            for _ in range(400):
                s = ru_racer_propagate(*s, 0.0, -0.35, 0.02)
                if rrt.check_rect_collision(s[0], s[1], s[2], grid, ymax, PPM):
                    hit, iv = True, math.hypot(s[3], s[4])
                    break
                if math.hypot(s[3], s[4]) < 0.3:
                    break
            w.writerow([vk, f"{ttc:.2f}", len(path), f, hit, f"{iv:.2f}"])
            out.flush()
        print(vk, ttc, "done", flush=True)
    out.close()


if __name__ == "__main__":
    main()
