#!/usr/bin/env python3
"""
run_sim_brake.py  -  emergency-braking baseline simulated with the RRT plant.

Straight-line braking (zero steering, longitudinal slip -0.35, i.e. the same
braking input as the fail-safe branches of the RRT) is propagated with
ru_racer_propagate and checked with the same inflated-footprint collision test
as the planner, until contact or standstill.  This makes the braking baseline
use exactly the vehicle model, start pose, and collision geometry of the EMRM
runs, so the two are compared on identical terms.

Output: results/sim_brake.csv with columns
    dataset, v_kmh, TTC_s, mu, impact_v, t_end
where impact_v = 0 means braking stopped without contact.
"""
from __future__ import annotations
import csv, math, os, sys
from multiprocessing import Pool, cpu_count

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
import run_rrt_tjunction as rrt                     # noqa: E402
import ru_racer_dynamics as rrd                     # noqa: E402

RES = os.path.join(THIS_DIR, "results")
OUT = os.path.join(RES, "sim_brake.csv")
SLIP_BRAKE, DT = -0.35, 0.01
BASE_F, BASE_R = rrd.TIRE_F["D"], rrd.TIRE_R["D"]
_G = None


def _init():
    global _G
    _G = rrt.load_occupancy(os.path.join(THIS_DIR, "maps", "tjunction_map.pgm"), 10.0)


def brake(job):
    ds, vk, ttc, mu = job
    rrd.TIRE_F["D"], rrd.TIRE_R["D"] = BASE_F * mu, BASE_R * mu
    g, _, _, _, ymax = _G
    v = vk / 3.6
    s = (41.8, 27.0 - v * ttc, 1.57, v, 0.0, 0.0)
    t = 0.0
    while t < 20.0:
        s = rrd.ru_racer_propagate(*s, 0.0, SLIP_BRAKE, DT)
        t += DT
        sp = math.hypot(s[3], s[4])
        if rrt.check_rect_collision(s[0], s[1], s[2], g, ymax, 10.0):
            return ds, vk, ttc, mu, sp, t
        if sp < 0.05:
            return ds, vk, ttc, mu, 0.0, t
    return ds, vk, ttc, mu, 0.0, t


def jobs():
    out = []
    for r in csv.DictReader(open(os.path.join(RES, "coverage_fine.csv"))):
        out.append(("fine", int(r["v_kmh"]), float(r["TTC_s"]), 1.0))
    for name in ("friction_speed", "friction_ttc"):
        for r in csv.DictReader(open(os.path.join(RES, f"{name}.csv"))):
            out.append((name, int(r["v_kmh"]), float(r["TTC_s"]), float(r["mu"])))
    # boundary-refinement grid of run_seed_campaign.py (campaign B)
    for vk in (30, 40, 50, 60, 72):
        for i in range(51):
            out.append(("refine", vk, round(0.20 + 0.02 * i, 2), 1.0))
    return out


def main():
    todo = jobs()
    done = set()
    if os.path.exists(OUT):
        for r in csv.DictReader(open(OUT)):
            done.add((r["dataset"], int(r["v_kmh"]), round(float(r["TTC_s"]), 2), round(float(r["mu"]), 2)))
    todo = [j for j in todo if (j[0], j[1], round(j[2], 2), round(j[3], 2)) not in done]
    new = not os.path.exists(OUT)
    with open(OUT, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["dataset", "v_kmh", "TTC_s", "mu", "impact_v", "t_end"])
        with Pool(max(1, cpu_count() - 1), initializer=_init) as p:
            for i, (ds, vk, ttc, mu, iv, t) in enumerate(p.imap_unordered(brake, todo, chunksize=8), 1):
                w.writerow([ds, vk, f"{ttc:.2f}", f"{mu:.2f}", f"{iv:.3f}", f"{t:.2f}"])
                if i % 100 == 0:
                    f.flush()
                    print(f"{i}/{len(todo)}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
