#!/usr/bin/env python3
"""
revision_tables.py  -  reproduces the tables added in the revision from the
released result files, without re-running any simulation.

  Table 10   EMRM and baselines on the 1,165 fine-grid cells (mu = 1); the EMRM
             for seed 42 and as the mean (range) over seeds 42 and 1000-1002
  Table 11   CAR by speed band and TTC band (seed 42)
  Sec. 5.3.8 worst-case points of the HARA-only and STPA-only test plans
  Table 13   straight road with a stalled vehicle versus the T-junction

Inputs (results/): coverage_fine.csv (planner, seed 42), seed_campaign_A.csv
(planner, seeds 1000-1002), sim_brake.csv (simulated braking), straight_brake.csv
and straight_rrt.csv (straight road).

A cell is avoided by braking if simulated braking stops without contact, and by
the EMRM if braking stops or the planner finds a collision-free path; otherwise
the impact speed is that of simulated braking.

Usage:  python revision_tables.py
"""
from __future__ import annotations
import csv, os, statistics

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(THIS_DIR, "results")

A_REF = 6.0        # m/s^2, constant-deceleration reference
CONTACT = 4.6      # m, first contact of the safety envelope beyond the reference line
V_MIT = 7.0        # m/s, mitigated-impact threshold of the outcome classes
V_HIGH = 10.0      # m/s, high-severity impact threshold
SEEDS = ["42", "1000", "1001", "1002"]
SPEED_BANDS = [(30, 40), (41, 56), (57, 72)]
TTC_BANDS = [(0.25, 0.67), (0.73, 1.27), (1.33, 2.89)]
N_HAZARDS, N_UCAS = 3, 4


def key(r):
    return int(float(r["v_kmh"])), round(float(r["TTC_s"]), 2)


def rows(name):
    return csv.DictReader(open(os.path.join(RES, name)))


def load():
    cells = {}
    for r in rows("coverage_fine.csv"):
        vk, _ = key(r)
        cells[key(r)] = dict(v=vk / 3.6, d=float(r["D_m"]), path={"42": r["has_path"] == "True"})
    for r in rows("sim_brake.csv"):
        if r["dataset"] == "fine":
            cells[key(r)]["brake"] = float(r["impact_v"])
    for r in rows("seed_campaign_A.csv"):
        cells[key(r)]["path"][r["seed"]] = r["has_path"] == "True"
    assert all("brake" in c and set(c["path"]) == set(SEEDS) for c in cells.values())
    return cells


def emrm(c, seed):
    return 0.0 if c["brake"] == 0.0 or c["path"][seed] else c["brake"]


def metrics(cells, impact):
    n = len(cells)
    imp = {k: impact(c) for k, c in cells.items()}
    hit = [v for v in imp.values() if v > 0]
    return dict(
        car=100 * (n - len(hit)) / n,
        colliding=len(hit),
        mean_all=3.6 * sum(imp.values()) / n,
        mean_coll=3.6 * sum(hit) / max(1, len(hit)),
        energy=100 * sum(v * v for v in imp.values()) / sum(c["v"] ** 2 for c in cells.values()),
        high=100 * sum(v >= V_HIGH for v in imp.values()) / n,
    )


def table10(cells):
    cols = {
        "No action": metrics(cells, lambda c: c["v"]),
        "Brake 6 m/s^2": metrics(cells, lambda c: max(0.0, c["v"] ** 2 - 2 * A_REF * (c["d"] + CONTACT)) ** 0.5),
        "Brake (sim.)": metrics(cells, lambda c: c["brake"]),
        "EMRM, seed 42": metrics(cells, lambda c: emrm(c, "42")),
    }
    per_seed = {s: metrics(cells, lambda c, s=s: emrm(c, s)) for s in SEEDS}
    labels = [("car", "CAR (%)", "{:.1f}"), ("colliding", "Colliding cells", "{:.0f}"),
              ("mean_all", "Mean impact, all (km/h)", "{:.1f}"),
              ("mean_coll", "Mean impact, colliding (km/h)", "{:.1f}"),
              ("energy", "Residual energy (%)", "{:.1f}"), ("high", "Impact >= 10 m/s (% cells)", "{:.1f}")]
    print(f"Table 10: EMRM and baselines on the {len(cells)} fine-grid cells (mu = 1)")
    print(f"  {'Metric':31s}" + "".join(f"{c:>15s}" for c in cols) + f"{'EMRM, four seeds':>24s}")
    for k, name, fmt in labels:
        vals = [m[k] for m in per_seed.values()]
        four = f"{fmt.format(statistics.mean(vals))} ({fmt.format(min(vals))}-{fmt.format(max(vals))})"
        print(f"  {name:31s}" + "".join(f"{fmt.format(m[k]):>15s}" for m in cols.values()) + f"{four:>24s}")
    brake = cols["Brake (sim.)"]["car"]
    print("  Per seed: " + "; ".join(
        f"seed {s} CAR {m['car']:.1f}%, gain {m['car'] - brake:.1f} points" for s, m in per_seed.items()))
    gains = [m["car"] - brake for m in per_seed.values()]
    print(f"  Gain over simulated braking: mean {statistics.mean(gains):.1f} points "
          f"(range {min(gains):.1f}-{max(gains):.1f})\n")


def in_band(x, band):
    return band[0] - 1e-9 <= x <= band[1] + 1e-9


def table11(cells):
    print("Table 11: CAR (%) by parameter band (mu = 1; EMRM for seed 42)")
    print(f"  {'Band':22s}{'Levels':>8s}{'Cells':>8s}{'Braking':>9s}{'EMRM':>8s}")
    groups = [(f"Speed {a}-{b} km/h", lambda k, a=a, b=b: in_band(k[0], (a, b)), 0) for a, b in SPEED_BANDS]
    groups += [(f"TTC {a:.2f}-{b:.2f} s", lambda k, a=a, b=b: in_band(k[1], (a, b)), 1) for a, b in TTC_BANDS]
    groups += [("All cells", lambda k: True, None)]
    for name, fn, axis in groups:
        sub = {k: c for k, c in cells.items() if fn(k)}
        levels = str(len({k[axis] for k in sub})) if axis is not None else "-"
        b, e = metrics(sub, lambda c: c["brake"])["car"], metrics(sub, lambda c: emrm(c, "42"))["car"]
        print(f"  {name:22s}{levels:>8s}{len(sub):8d}{b:9.1f}{e:8.1f}")
    print()


def outcome_class(c):
    if c["brake"] == 0.0:
        return "braking avoids"
    if c["path"]["42"]:
        return "only steering avoids"
    return "no path, impact < 7 m/s" if c["brake"] < V_MIT else "no path, impact >= 7 m/s"


def worst_case_points(cells):
    print("Section 5.3.8: worst-case points of the HARA-only and STPA-only test plans (seed 42)")
    points = []
    for sb in SPEED_BANDS:
        for tb in TTC_BANDS:
            band = [k for k in cells if in_band(k[0], sb) and in_band(k[1], tb)]
            ttc = min(k[1] for k in band)
            point = (max(k[0] for k in band if k[1] == ttc), ttc)
            points.append(point)
            print(f"  speed {sb[0]}-{sb[1]} km/h, TTC {tb[0]:.2f}-{tb[1]:.2f} s: "
                  f"{point[0]} km/h, TTC {point[1]:.2f} s -> {outcome_class(cells[point])}")
    steering_only = [k for k, c in cells.items() if outcome_class(c) == "only steering avoids"]
    print(f"  HARA-only: {N_HAZARDS} x {len(points)} = {N_HAZARDS * len(points)} tests; "
          f"STPA-only: {N_UCAS} x {len(points)} = {N_UCAS * len(points)} tests; "
          f"{len(set(points))} distinct cells of {len(cells)}")
    print(f"  Points in the {len(steering_only)} cells where only steering avoids the truck: "
          f"{len(set(points) & set(steering_only))}\n")


def table13(cells):
    sb = {key(r): float(r["impact_v"]) for r in rows("straight_brake.csv")}
    sr = {key(r): r for r in rows("straight_rrt.csv")}
    same = sb.keys() == cells.keys() and all((sb[k] == 0.0) == (cells[k]["brake"] == 0.0) for k in sb)
    print("Table 13: CAR (%) on the straight road and at the T-junction (mu = 1, seed 42)")
    print(f"  Braking outcome identical in both geometries: {same}; "
          f"planner run in the {len(sr)} cells in which braking collides")
    print(f"  {'Band':16s}{'Cells':>7s}{'Braking':>9s}{'EMRM T-junction':>17s}{'EMRM straight':>15s}{'Branch past':>13s}")
    for name, (a, b) in [(f"{a}-{b} km/h", (a, b)) for a, b in SPEED_BANDS] + [("All", (30, 72))]:
        ks = [k for k in sb if a <= k[0] <= b]
        n = len(ks)
        brake = sum(sb[k] == 0.0 for k in ks)
        goal = sum(sr[k]["has_path"] == "True" for k in ks if k in sr)
        past = sum(sr[k]["passed"] == "True" for k in ks if k in sr)
        tj = sum(emrm(cells[k], "42") == 0.0 for k in ks)
        print(f"  {name:16s}{n:7d}{100 * brake / n:9.1f}{100 * tj / n:17.1f}{100 * (brake + goal) / n:15.1f}{past:13d}")
    past_all = sum(r["passed"] == "True" for r in sr.values())
    brake_all = sum(v == 0.0 for v in sb.values())
    print(f"  Counting the {past_all} branch-past cells as avoided gives a CAR of "
          f"{100 * (brake_all + past_all) / len(sb):.1f}%")


def main():
    cells = load()
    table10(cells)
    table11(cells)
    worst_case_points(cells)
    table13(cells)


if __name__ == "__main__":
    main()
