#!/usr/bin/env python3
"""
run_seed_campaign.py  -  repeated-seed RRT campaign for the revision.

Campaign A (seed replication): every geometrically feasible cell of the fine
  Speed x TTC grid (43 x 46, mu = 1.0, 1,165 cells) is re-planned with
  N_SEEDS independent random seeds, so that the avoidance outcome of each cell
  becomes a Bernoulli sample and Wilson intervals can be reported.

Campaign B (boundary refinement): at five ego speeds the TTC axis is refined to
  0.02 s steps around the avoidance and stop boundaries, with N_SEEDS_B seeds
  per point, to quantify discretisation error of the 0.06 s grid.

Planner settings are identical to run_coverage_fine.py (600 nodes, dt = 0.04 s,
goal bias 0.55). Output is incremental and resumable.

  results/seed_campaign_A.csv
  results/seed_campaign_B.csv
"""
from __future__ import annotations
import csv, math, os, random, sys, time
from multiprocessing import Pool

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
import run_rrt_tjunction as rrt

MAP_PGM = os.path.join(THIS_DIR, "maps", "tjunction_map.pgm")
RES = os.path.join(THIS_DIR, "results")
PPM, MAX_NODES, DT = 10.0, 600, 0.04
TRUCK_Y, MIN_Y, START_X, START_PSI, GOAL = 27.0, 2.5, 41.8, 1.57, (44.0, 48.0)

SPEEDS = list(range(30, 73))
TTCS = [round(0.25 + 0.06 * i, 2) for i in range(46)]
N_SEEDS = 3
SEEDS_A = [1000 + k for k in range(N_SEEDS)]

SPEEDS_B = [30, 40, 50, 60, 72]
TTCS_B = [round(0.20 + 0.02 * i, 2) for i in range(51)]   # 0.20 .. 1.20 s
N_SEEDS_B = 5
SEEDS_B = [2000 + k for k in range(N_SEEDS_B)]

HEADER = ["v_kmh", "TTC_s", "seed", "has_path", "n_stop", "n_coll", "min_v_coll"]

_G = None


def _init():
    global _G
    _G = rrt.load_occupancy(MAP_PGM, PPM)


def run_one(job):
    vk, ttc, seed = job
    grid, xmin, xmax, ymin, ymax = _G
    v = vk / 3.6
    start = (START_X, TRUCK_Y - v * ttc, START_PSI, v)
    random.seed(seed)
    tg, to, path, _, n_stop = rrt.rrt_double_tree(
        start, GOAL, grid, xmin, xmax, ymin, ymax, PPM,
        max_nodes=MAX_NODES, dt=DT, goal_sample_rate=0.55)
    no = to["nodes"]
    n_coll = max(0, len(no) - n_stop)
    mvc = ""
    if n_coll:
        mvc = f"{min(math.hypot(n[3], n[4]) for n in no[n_stop:]):.3f}"
    return [vk, f"{ttc:.2f}", seed, str(len(path) > 0), n_stop, n_coll, mvc]


def campaign(name, jobs, workers):
    path = os.path.join(RES, name)
    done = set()
    if os.path.exists(path):
        for r in csv.DictReader(open(path)):
            done.add((int(r["v_kmh"]), float(r["TTC_s"]), int(r["seed"])))
    todo = [j for j in jobs if (j[0], j[1], j[2]) not in done]
    print(f"{name}: {len(jobs)} jobs, {len(done)} done, {len(todo)} to run", flush=True)
    new = not os.path.exists(path) or not done
    f = open(path, "a", newline="")
    w = csv.writer(f)
    if new:
        w.writerow(HEADER)
    t0 = time.time()
    with Pool(workers, initializer=_init) as pool:
        for i, row in enumerate(pool.imap_unordered(run_one, todo, chunksize=4), 1):
            w.writerow(row); f.flush()
            if i % 200 == 0:
                el = time.time() - t0
                print(f"  {name} {i}/{len(todo)}  ETA {(len(todo)-i)*el/i/60:.0f} min", flush=True)
    f.close()
    print(f"{name}: complete", flush=True)


def main():
    workers = max(1, (os.cpu_count() or 2) - 1)
    jobs_a = [(vk, t, s) for vk in SPEEDS for t in TTCS
              if TRUCK_Y - vk / 3.6 * t >= MIN_Y for s in SEEDS_A]
    jobs_b = [(vk, t, s) for vk in SPEEDS_B for t in TTCS_B
              if TRUCK_Y - vk / 3.6 * t >= MIN_Y for s in SEEDS_B]
    campaign("seed_campaign_B.csv", jobs_b, workers)
    campaign("seed_campaign_A.csv", jobs_a, workers)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
