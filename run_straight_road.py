#!/usr/bin/env python3
"""
run_straight_road.py  -  second scenario: straight road, stalled vehicle in the ego lane.

The map has the size, scale, and road corridor of the T-junction map, but the
road is straight and the only obstacle is a stalled vehicle that blocks the ego
lane while the adjacent lane is free.  The rear edge of the stalled vehicle is
placed so that first footprint contact on the ego centreline occurs at the same
position as at the T-junction; straight-line braking therefore has the same
outcome on both maps, and differences in EMRM performance come from the escape
geometry alone.

Steps
  1. build maps/straight_road_map.pgm (+ .png) and verify the contact position
  2. simulated braking on every fine-grid cell (results/straight_brake.csv)
  3. RRT (seed 42, same settings as the main campaign) on every cell where
     braking collides (results/straight_rrt.csv); cells where braking stops are
     avoided by handing over to braking (transition tau_4) and need no planning.
"""
from __future__ import annotations
import csv, math, os, random, sys, time
from multiprocessing import Pool, cpu_count

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
import run_rrt_tjunction as rrt                     # noqa: E402
import ru_racer_dynamics as rrd                     # noqa: E402

RES = os.path.join(THIS_DIR, "results")
MAP_T = os.path.join(THIS_DIR, "maps", "tjunction_map.pgm")
MAP_S = os.path.join(THIS_DIR, "maps", "straight_road_map.pgm")
PPM, MAX_NODES, DT_RRT, SEED = 10.0, 600, 0.04, 42
START_X, START_PSI, REF_Y, MIN_Y = 41.8, 1.57, 27.0, 2.5
ROAD_X = (25.3, 46.9)
LANE_SPLIT = 40.0            # left edge of the stalled vehicle (ego footprint spans 39.8-43.8)
STALL_LEN = 10.0
GOAL = (36.5, 51.0)          # adjacent lane, beside and beyond the stalled vehicle
BOXES = {
    "junction": ((32.0, 39.0), (28.0, 52.0)),
    "left": ((25.5, 38.0), (34.0, 52.0)),
    "right": (((LANE_SPLIT, ROAD_X[1])), (30.0, 46.0)),
}
SLIP_BRAKE, DT_BRAKE = -0.35, 0.01
PASS_Y = 47.5               # beyond the stalled vehicle (front edge at y = 46.25 m)

SPEEDS = list(range(30, 73))
TTCS = [round(0.25 + 0.06 * i, 2) for i in range(46)]


def contact_y(grid, ymax):
    y = 10.0
    while y < 50.0:
        if rrt.check_rect_collision(START_X, y, START_PSI, grid, ymax, PPM):
            return y
        y += 0.01
    return None


def build_map():
    gT, _, _, _, ymaxT = rrt.load_occupancy(MAP_T, PPM)
    yc = contact_y(gT, ymaxT)
    h, w = gT.shape
    g = np.zeros((h, w), dtype=np.uint8)
    c0, c1 = int(round(ROAD_X[0] * PPM)), int(round(ROAD_X[1] * PPM)) + 1
    g[:, c0:c1] = 1
    half_len = rrt.VEH_LENGTH * rrt.VEH_SCALE * rrt.VEH_LENGTH_SCALE / 2
    y_rear = yc + half_len
    # place the rear edge so the centreline contact matches the T-junction to within 0.01 m
    for _ in range(40):
        y0, y1 = y_rear, y_rear + STALL_LEN
        gs = g.copy()
        r0, r1 = int(round((ymaxT - y1) * PPM)), int(round((ymaxT - y0) * PPM))
        gs[r0:r1 + 1, int(round(LANE_SPLIT * PPM)):c1] = 0
        ys = contact_y(gs, ymaxT)
        if abs(ys - yc) < 0.015:
            break
        y_rear += (yc - ys)
    with open(MAP_S, "wb") as f:
        f.write(b"P5\n%d %d\n255\n" % (w, h))
        f.write((gs * 255).astype(np.uint8).tobytes())
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        img = np.where(gs[..., None] == 1, [0.80, 0.80, 0.80], [0.30, 0.45, 0.30])
        rr0, rr1 = int(round((ymaxT - (y_rear + STALL_LEN)) * PPM)), int(round((ymaxT - y_rear) * PPM))
        img[rr0:rr1 + 1, int(round(LANE_SPLIT * PPM)):c1] = [0.15, 0.15, 0.15]
        plt.imsave(MAP_S.replace(".pgm", ".png"), img)
    except Exception:
        pass
    print(f"T-junction contact y = {yc:.2f}; straight-road contact y = {ys:.2f}; "
          f"stalled vehicle y = {y_rear:.2f}-{y_rear + STALL_LEN:.2f}", flush=True)
    return yc, ys


_G = None


def _init():
    global _G
    _G = rrt.load_occupancy(MAP_S, PPM)


def brake(job):
    vk, ttc = job
    g, _, _, _, ymax = _G
    v = vk / 3.6
    s = (START_X, REF_Y - v * ttc, START_PSI, v, 0.0, 0.0)
    t = 0.0
    while t < 20.0:
        s = rrd.ru_racer_propagate(*s, 0.0, SLIP_BRAKE, DT_BRAKE)
        t += DT_BRAKE
        sp = math.hypot(s[3], s[4])
        if rrt.check_rect_collision(s[0], s[1], s[2], g, ymax, PPM):
            return vk, ttc, sp
        if sp < 0.05:
            return vk, ttc, 0.0
    return vk, ttc, 0.0


def plan(job):
    vk, ttc = job
    grid, xmin, xmax, ymin, ymax = _G
    v = vk / 3.6
    start = (START_X, REF_Y - v * ttc, START_PSI, v)
    random.seed(SEED)
    t0 = time.time()
    tg, to, path, _, n_stop = rrt.rrt_double_tree(
        start, GOAL, grid, xmin, xmax, ymin, ymax, PPM,
        max_nodes=MAX_NODES, dt=DT_RRT, goal_sample_rate=0.55, sample_boxes=BOXES)
    no = to["nodes"]
    n_coll = max(0, len(no) - n_stop)
    mvc = min(math.hypot(n[3], n[4]) for n in no[n_stop:]) if n_coll else float("nan")
    passed = any(n[1] >= PASS_Y and n[0] < LANE_SPLIT for n in tg["nodes"])
    return vk, ttc, len(path) > 0, passed, n_stop, n_coll, mvc, time.time() - t0


def cells():
    return [(vk, t) for vk in SPEEDS for t in TTCS if REF_Y - vk / 3.6 * t >= MIN_Y]


def main():
    build_map()
    workers = max(1, cpu_count() - 1)
    out_b = os.path.join(RES, "straight_brake.csv")
    with Pool(workers, initializer=_init) as p:
        rows = p.map(brake, cells(), chunksize=16)
    with open(out_b, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["v_kmh", "TTC_s", "impact_v"])
        for vk, t, iv in sorted(rows):
            w.writerow([vk, f"{t:.2f}", f"{iv:.3f}"])
    coll = [(vk, t) for vk, t, iv in rows if iv > 0]
    print(f"braking: {len(rows)} cells, {len(rows) - len(coll)} stop, {len(coll)} collide", flush=True)

    out_r = os.path.join(RES, "straight_rrt.csv")
    done = set()
    if os.path.exists(out_r):
        for r in csv.DictReader(open(out_r)):
            done.add((int(r["v_kmh"]), round(float(r["TTC_s"]), 2)))
    todo = sorted(c for c in coll if (c[0], round(c[1], 2)) not in done)
    if len(sys.argv) > 1 and sys.argv[1] == "pilot":
        todo = todo[:: max(1, len(todo) // (2 * workers))][: 2 * workers]
    print(f"RRT: {len(coll)} braking-collision cells, {len(done)} done, {len(todo)} to run", flush=True)
    new = not os.path.exists(out_r)
    t0 = time.time()
    with open(out_r, "a", newline="") as f, Pool(workers, initializer=_init) as p:
        w = csv.writer(f)
        if new:
            w.writerow(["v_kmh", "TTC_s", "has_path", "passed", "n_stop", "n_coll", "min_v_coll", "t_run"])
        for i, (vk, t, hp, ps, ns, nc, mvc, tr) in enumerate(p.imap_unordered(plan, todo), 1):
            w.writerow([vk, f"{t:.2f}", hp, ps, ns, nc, f"{mvc:.3f}", f"{tr:.1f}"])
            f.flush()
            if i % 10 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"  {i}/{len(todo)}  ETA {(len(todo) - i) * el / i / 60:.0f} min", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
