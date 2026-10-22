#!/usr/bin/env python3
"""
audit_revision.py  -  single-source-of-truth audit of the released RRT data.

Recomputes every count, percentage, and metric reported in the revised
manuscript directly from results/coverage_fine.csv (fine Speed x TTC grid,
mu = 1.0) and results/friction_*.csv, without re-running any simulation.

Braking model (identical to run_rrt_tjunction.MAX_DECEL and to the
classifier in run_coverage_fine.py): constant deceleration A_BRAKE = 6 m/s^2,
so straight-line braking stops before the truck iff TTC >= v / (2 A_BRAKE).

Usage:  python audit_revision.py  [--figs OUTDIR]
"""
from __future__ import annotations
import argparse, csv, json, math, os, sys
from collections import Counter

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(THIS_DIR, "results")

A_BRAKE = 6.0                    # m/s^2, straight-line emergency braking
G_ACC = 9.81
TRUCK_Y, MIN_Y = 27.0, 2.5
MAX_D = TRUCK_Y - MIN_Y          # 24.5 m available run-up distance
SPEEDS = list(range(30, 73))     # 43 levels, km/h
TTCS = [round(0.25 + 0.06 * i, 2) for i in range(46)]   # 46 levels, s

# Illustrative impact-speed severity proxy used by run_coverage_fine.v2loss
def v2loss(v):
    if v == 0:      return 0
    if v < 5.0:     return 2
    if v < 10.0:    return 4
    if v < 15.0:    return 6
    return 7


def load_fine():
    rows = []
    with open(os.path.join(RES, "coverage_fine.csv")) as f:
        for r in csv.DictReader(f):
            vk = int(r["v_kmh"]); v = vk / 3.6; ttc = float(r["TTC_s"])
            d = float(r["D_m"])
            brake_v = math.sqrt(max(0.0, v * v - 2 * A_BRAKE * d))
            rows.append(dict(
                vk=vk, v=v, ttc=ttc, d=d,
                path=r["has_path"] == "True",
                n_stop=int(r["n_stop"]),
                min_vc=float(r["min_v_coll"]) if r["min_v_coll"] else float("nan"),
                outcome=r["outcome"], brake_v=brake_v,
                stoppable=brake_v == 0.0,
            ))
    return rows


def impact(r, s):
    """Impact speed (m/s) for strategy s, as in run_coverage_fine.plot_loss_comparison."""
    if s == "estop":
        return r["brake_v"]
    if s in ("dodge", "all"):
        return 0.0 if r["path"] else r["brake_v"]
    # drift: avoid if a path exists, else the minimum-speed collision node of the
    # orange (fail-safe) tree when the cell is mitigatable, else straight braking
    if r["path"] or r["stoppable"]:
        return 0.0
    if r["outcome"] == "MITIGATABLE" and not math.isnan(r["min_vc"]):
        return min(r["min_vc"], r["brake_v"])
    return r["brake_v"]


def metrics(rows, s):
    n = len(rows)
    imp = [impact(r, s) for r in rows]
    loss0 = [v2loss(r["v"]) for r in rows]
    loss1 = [v2loss(v) for v in imp]
    ke0 = sum(r["v"] ** 2 for r in rows)
    return dict(
        n=n,
        car=100 * sum(v == 0 for v in imp) / n,
        mean_imp_kmh=3.6 * sum(imp) / n,
        mean_imp_coll_kmh=(3.6 * sum(v for v in imp if v > 0) / max(1, sum(v > 0 for v in imp))),
        ke_frac=100 * sum(v * v for v in imp) / ke0,
        mit=100 * sum(a - b > 0 for a, b in zip(loss0, loss1)) / n,
        irrev0=100 * sum(a >= 6 for a in loss0) / n,
        irrev1=100 * sum(b >= 6 for b in loss1) / n,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    out = {}

    rows = load_fine()
    n_grid = len(SPEEDS) * len(TTCS)
    n_geom = sum(1 for vk in SPEEDS for t in TTCS if TRUCK_Y - vk / 3.6 * t >= MIN_Y)
    out["grid"] = dict(full=n_grid, geometric_feasible=n_geom,
                       excluded=n_grid - n_geom, simulated=len(rows))

    cls = Counter(r["outcome"] for r in rows)
    out["classes"] = {k: (v, round(100 * v / len(rows), 1)) for k, v in cls.items()}

    # Where do the colour classes lie relative to the stop boundary?
    side = Counter((r["outcome"], r["stoppable"]) for r in rows)
    out["class_vs_stop_boundary"] = {f"{k[0]}|stoppable={k[1]}": v for k, v in side.items()}

    # Analytical vs RRT confusion: analytical "stoppable" vs RRT "avoidance path found"
    conf = Counter((r["stoppable"], r["path"]) for r in rows)
    out["confusion_stop_vs_path"] = {f"stoppable={a}|path={b}": v for (a, b), v in conf.items()}

    # Empirical avoidance boundary: smallest TTC with a path, per speed
    emp = {}
    nonmono = []
    for vk in SPEEDS:
        sub = sorted((r for r in rows if r["vk"] == vk), key=lambda r: r["ttc"])
        ok = [r["ttc"] for r in sub if r["path"]]
        emp[vk] = min(ok) if ok else None
        if ok:
            # cells above the first success with no path = non-monotone (planner) disagreements
            nonmono += [(vk, r["ttc"]) for r in sub if r["ttc"] > min(ok) and not r["path"]]
    out["empirical_avoid_ttc_min"] = {k: emp[k] for k in (30, 40, 50, 60, 72)}
    out["empirical_avoid_ttc_range"] = (min(v for v in emp.values() if v), max(v for v in emp.values() if v))
    out["empirical_avoid_D_min"] = {k: round(emp[k] * k / 3.6, 2) for k in (30, 40, 50, 60, 72) if emp[k]}
    out["nonmonotone_cells"] = len(nonmono)
    out["nonmonotone_examples"] = nonmono[:8]

    # Representative disagreement cells (stoppable but no path; not stoppable but path)
    ex_a = [(r["vk"], r["ttc"], r["outcome"]) for r in rows if r["stoppable"] and not r["path"]][:6]
    ex_b = [(r["vk"], r["ttc"], r["outcome"]) for r in rows if (not r["stoppable"]) and r["path"]][:3]
    out["examples_stoppable_nopath"] = ex_a
    out["examples_notstoppable_path"] = ex_b

    # What the original 4-panel analytic map assumed (stop boundary with mu*g)
    above_mug = sum(1 for r in rows if r["ttc"] >= r["v"] / (2 * G_ACC))
    out["frac_feasible_above_v_over_2g"] = round(100 * above_mug / len(rows), 1)
    out["frac_feasible_above_v_over_12"] = round(100 * sum(r["stoppable"] for r in rows) / len(rows), 1)

    # Strategy metrics on the full feasible set
    out["metrics"] = {s: {k: round(v, 2) for k, v in metrics(rows, s).items()}
                      for s in ("estop", "dodge", "drift", "all")}

    # Bands
    bands = {
        "speed_low_30_40": lambda r: 30 <= r["vk"] <= 40,
        "speed_med_41_56": lambda r: 41 <= r["vk"] <= 56,
        "speed_high_57_72": lambda r: 57 <= r["vk"] <= 72,
        "ttc_low_le_0.70": lambda r: r["ttc"] <= 0.70,
        "ttc_med_0.73_1.27": lambda r: 0.70 < r["ttc"] <= 1.30,
        "ttc_high_ge_1.33": lambda r: r["ttc"] > 1.30,
    }
    bt = {}
    for name, fn in bands.items():
        sub = [r for r in rows if fn(r)]
        levels = sorted({r["ttc"] for r in sub}) if "ttc" in name else sorted({r["vk"] for r in sub})
        bt[name] = dict(
            cells=len(sub), levels=len(levels), lv_min=levels[0], lv_max=levels[-1],
            **{f"car_{s}": round(metrics(sub, s)["car"]) for s in ("estop", "dodge", "drift")},
            **{f"mit_{s}": round(metrics(sub, s)["mit"]) for s in ("estop", "dodge", "drift")},
        )
    out["bands"] = bt

    # Friction sweeps (stored outcomes and has_path)
    for name in ("friction_speed", "friction_ttc"):
        p = os.path.join(RES, f"{name}.csv")
        fr = list(csv.DictReader(open(p)))
        per_mu = {}
        for mu in sorted({r["mu"] for r in fr}):
            sub = [r for r in fr if r["mu"] == mu]
            per_mu[mu] = dict(n=len(sub), path=sum(r["has_path"] == "True" for r in sub))
        out[name] = dict(rows=len(fr), per_mu=per_mu,
                         override_G_without_path=sum(r["has_path"] == "False" and r["outcome"] in ("FULLY_AVOIDABLE", "AGGRESSIVE_AVOID") for r in fr))

    # Perception latency / TTC over-estimation: the manoeuvre starts when the true
    # TTC is TTC - delta, so the outcome of cell (v, TTC) becomes that of (v, TTC - delta).
    look = {(r["vk"], r["ttc"]): r for r in rows}
    lat = {}
    for k in (0, 1, 2, 3, 5):
        delta = round(0.06 * k, 2)
        ok_e = ok_b = 0
        for r in rows:
            t2 = round(r["ttc"] - delta, 2)
            s = look.get((r["vk"], t2))
            if s is None:
                continue
            ok_b += s["stoppable"]
            ok_e += s["stoppable"] or s["path"]
        lat[delta] = dict(car_brake=round(100 * ok_b / len(rows), 1), car_emrm=round(100 * ok_e / len(rows), 1))
    out["latency_shift"] = lat

    # Abstract-cell (3 speed bands x 3 TTC bands) coverage of alternative test sets
    def band(vk, t):
        sb = 0 if vk <= 40 else (1 if vk <= 56 else 2)
        tb = 0 if t <= 0.70 else (1 if t <= 1.30 else 2)
        return sb, tb
    feas_bands = {band(r["vk"], r["ttc"]) for r in rows}
    manual_pts = [(round(v * 3.6, 1), round(21.8 / v, 2)) for v in (16.0, 12.0, 8.0, 4.0)]
    manual_in = [(vk, t) for vk, t in manual_pts if 30 <= vk <= 72 and 0.25 <= t <= 2.95]
    coarse = [(int(float(r["v_kmh"])), float(r["TTC_s"])) for r in
              csv.DictReader(open(os.path.join(RES, "coverage_results.csv"))) if r["v_kmh"].strip()]
    coarse_feas = [(vk, t) for vk, t in coarse if TRUCK_Y - vk / 3.6 * t >= MIN_Y]
    out["abstract_cells"] = dict(
        feasible_bands=len(feas_bands),
        manual_points_in_domain=manual_in, manual_bands=len({band(*p) for p in manual_in}),
        coarse_points=len(coarse), coarse_feasible=len(coarse_feas),
        coarse_bands=len({band(*p) for p in coarse_feas}),
    )

    files = {f: sum(1 for _ in open(os.path.join(RES, f))) - 1 for f in
             ("coverage_fine.csv", "coverage_results.csv", "friction_speed.csv", "friction_ttc.csv",
              "emrm_validation/emrm_validation_results.csv")}
    out["released_rows"] = dict(files, total=sum(files.values()))

    txt = json.dumps(out, indent=1, default=str)
    print(txt)
    if args.json:
        open(args.json, "w").write(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
