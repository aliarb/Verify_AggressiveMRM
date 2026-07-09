#!/usr/bin/env python3
"""
run_coverage_fine.py  –  fine-grid (Ego Speed × TTC) coverage sweep.

Grid
────
  Speeds  : 30–72 km/h in 1 km/h steps  → 43 values
  TTCs    : 0.25–2.95 s in 0.06 s steps  → 46 values
  Cells   : 43 × 46 = 1978 total;  ≈ 1330 feasible (v·TTC < 24.5 m)
  RRT     : max_nodes = 600,  dt = 0.04 s  (faster per cell)

Outputs
───────
  results/coverage_fine.csv         (incremental – safe to interrupt)
  ../../figs/mitigability_region.png (heatmap, overwritten on completion)
  ../../figs/loss_comparison.png
  ../../figs/param_sweep.png
  ../../figs/coverage_comparison.png
"""

from __future__ import annotations
import csv, math, os, random, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import run_rrt_tjunction as rrt
import ru_racer_dynamics as rrd

# ── paths ─────────────────────────────────────────────────────────────────
MAP_PNG  = os.path.join(THIS_DIR, "maps", "tjunction_map.png")
MAP_PGM  = os.path.join(THIS_DIR, "maps", "tjunction_map.pgm")
FIGS_DIR = os.path.join(THIS_DIR, "..", "..", "figs")
RES_DIR  = os.path.join(THIS_DIR, "results")
CSV_OUT  = os.path.join(RES_DIR, "coverage_fine.csv")
PPM      = 10.0

# ── geometry ──────────────────────────────────────────────────────────────
TRUCK_FRONT_Y = 27.0
START_X       = 41.8
START_PSI     = 1.57
GOAL          = (44.0, 48.0)
MIN_START_Y   = 2.5
MAX_D         = TRUCK_FRONT_Y - MIN_START_Y   # 24.5 m

# ── sweep grid ────────────────────────────────────────────────────────────
SPEEDS_KMH = list(range(30, 73, 1))           # 43 values
SPEEDS_MS  = [s / 3.6 for s in SPEEDS_KMH]
TTCS       = [round(0.25 + i * 0.06, 2) for i in range(46)]  # 0.25→2.95 s

MAX_NODES  = 600
DT         = 0.04

# ── outcome codes ─────────────────────────────────────────────────────────
FULLY_AVOIDABLE  = 3
AGGRESSIVE_AVOID = 2
MITIGATABLE      = 1
NOT_MITIGATABLE  = 0
INFEASIBLE       = -1

C_MAP = {
    FULLY_AVOIDABLE:  "#2e7d32",
    AGGRESSIVE_AVOID: "#f9a825",
    MITIGATABLE:      "#e65100",
    NOT_MITIGATABLE:  "#b71c1c",
    INFEASIBLE:       "#bdbdbd",
}

# ── classification ────────────────────────────────────────────────────────

def classify(tree_g, tree_o, path, n_stop, v, ttc) -> int:
    ng        = tree_g["nodes"]
    nodes_past = sum(1 for n in ng if n[1] > 43.0)
    has_path  = len(path) > 0
    ttc_stop  = v / 12.0
    d_detect  = v * ttc
    can_stop  = d_detect > v * v / 12.0 * 1.10

    if has_path:
        return FULLY_AVOIDABLE if ttc >= ttc_stop else AGGRESSIVE_AVOID
    if n_stop >= 1 or can_stop:
        return MITIGATABLE
    if len(tree_o["nodes"]) > n_stop:
        coll = tree_o["nodes"][n_stop:]
        if coll:
            mv = min(math.sqrt(n[3]**2 + n[4]**2) for n in coll)
            if mv < 7.0:
                return MITIGATABLE
    return NOT_MITIGATABLE


# ── helpers ───────────────────────────────────────────────────────────────

def _ensure_pgm():
    if not os.path.exists(MAP_PNG):
        return
    try:
        from PIL import Image
        img  = Image.open(MAP_PNG).convert("L")
        arr  = np.array(img)
        h, w = arr.shape
        data = np.where(arr < 50, 0, 255).astype(np.uint8)
        with open(MAP_PGM, "wb") as f:
            f.write(f"P5\n{w} {h}\n255\n".encode())
            f.write(data.tobytes())
    except ImportError:
        pass


def v2loss(v_ms):
    if   v_ms == 0:      return 0
    elif v_ms < 5.0:     return 2
    elif v_ms < 10.0:    return 4
    elif v_ms < 15.0:    return 6
    else:                return 7


# ── figure: heatmap ───────────────────────────────────────────────────────

def plot_heatmap(outcome_matrix, speeds_kmh, ttcs, out_path):
    nT, nV = outcome_matrix.shape

    # pixel image
    img = np.zeros((nT, nV, 3))
    hex2rgb = lambda h: [int(h[i:i+2],16)/255 for i in (1,3,5)]
    for it in range(nT):
        for iv in range(nV):
            img[it, iv] = hex2rgb(C_MAP[outcome_matrix[it, iv]])

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    fig.subplots_adjust(left=0.12, right=0.99, top=0.99, bottom=0.12)

    ax.imshow(img, origin="lower", aspect="auto",
              extent=[-0.5, nV-0.5, -0.5, nT-0.5])

    # stop boundary
    v_cont   = np.linspace(speeds_kmh[0]/3.6*0.9, speeds_kmh[-1]/3.6*1.05, 400)
    ttc_stop = v_cont / 12.0
    def v2x(v): return (v - speeds_kmh[0]/3.6) / ((speeds_kmh[-1]-speeds_kmh[0])/3.6) * (nV-1)
    def t2y(t): return (t - ttcs[0]) / (ttcs[-1] - ttcs[0]) * (nT-1)
    ax.plot(v2x(v_cont), t2y(ttc_stop), color="white", lw=2.0, ls="--", zorder=6)

    # D iso-lines
    for d_iso in [5, 10, 15, 20]:
        xs, ys = [], []
        for iv, vk in enumerate(speeds_kmh):
            v = vk / 3.6
            t_val = d_iso / v
            if ttcs[0] <= t_val <= ttcs[-1]:
                xs.append(iv); ys.append(t2y(t_val))
        if len(xs) >= 2:
            ax.plot(xs, ys, color="white", lw=0.8, ls=":", alpha=0.45, zorder=5)
            ax.text(xs[-1]+0.2, ys[-1], f"D={d_iso}m",
                    fontsize=5.5, color="white", alpha=0.7, va="center")

    # ticks – show every 4th speed and every 4th TTC
    xt = list(range(0, nV, 4));  xk = [speeds_kmh[i] for i in xt]
    yt = list(range(0, nT, 4));  yk = [f"{ttcs[i]:.2f}" for i in yt]
    ax.set_xticks(xt); ax.set_xticklabels(xk, fontsize=8)
    ax.set_yticks(yt); ax.set_yticklabels(yk, fontsize=8)
    ax.set_xlabel("Ego Speed (km/h)", fontsize=10, labelpad=4)
    ax.set_ylabel("Time-to-Collision TTC (s)", fontsize=10, labelpad=4)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Heatmap → {out_path}")


# ── figure: loss comparison ───────────────────────────────────────────────

def plot_loss_comparison(rows, out_path):
    STRATS    = ["Emergency\nStop", "EMRM\nDodge", "EMRM\nDrift", "EMRM\nAll"]
    STRAT_KEY = ["estop", "dodge", "drift", "all"]
    BAR_CLR   = ["#607d8b", "#1976d2", "#7b1fa2", "#2e7d32"]
    N         = len(rows)

    def impact(row, s):
        c = row["code"]
        brake = row["brake_v"]
        if s == "estop":
            return brake
        if s in ("dodge", "all"):
            return 0.0 if row["path"] else brake
        # drift
        if c in ("G","Y"):  return 0.0
        if c == "O":
            mv = row["min_vc"]
            return mv if mv and not math.isnan(mv) else brake
        return brake

    def metrics(s):
        imp  = [impact(r, s) for r in rows]
        car  = 100 * sum(v == 0 for v in imp) / N
        mv   = 3.6 * sum(imp) / N
        mit  = 100 * sum(
            (r["loss_noact"] - v2loss(v)) > 0
            for r, v in zip(rows, imp)
        ) / N
        dls  = sum(r["loss_noact"] - v2loss(impact(r, s)) for r in rows) / N
        return car, mv, mit, dls

    stats = {k: metrics(k) for k in STRAT_KEY}

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.6))
    fig.subplots_adjust(wspace=0.38, left=0.10, right=0.98, top=0.88, bottom=0.18)
    x = np.arange(len(STRATS))

    for ax, key, title, ylim in [
        (axes[0], 0, "Collision Avoidance Rate", 110),
        (axes[1], 1, "Mean Residual Impact Speed (km/h)", 35),
    ]:
        vals = [stats[k][key] for k in STRAT_KEY]
        bars = ax.bar(x, vals, color=BAR_CLR, edgecolor="white", lw=0.8, width=0.6)
        ax.set_ylim(0, ylim)
        ax.set_xticks(x); ax.set_xticklabels(STRATS, fontsize=8)
        ax.set_title(title, fontsize=9, pad=4)
        if key == 0:
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
            ax.set_ylabel("Rate (%)", fontsize=9)
        else:
            ax.set_ylabel("km/h", fontsize=9)
        for bar, v in zip(bars, vals):
            lbl = f"{v:.0f}%" if key == 0 else f"{v:.1f}"
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.4,
                    lbl, ha="center", va="bottom", fontsize=8, fontweight="bold")

    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Loss comparison → {out_path}")
    return stats


# ── figure: param sweep ───────────────────────────────────────────────────

def plot_param_sweep(rows, speeds_kmh, out_path):
    SWEEP = [("estop","Emergency Stop","#607d8b","--"),
             ("dodge","EMRM Dodge",    "#1976d2","-"),
             ("drift","EMRM Drift",    "#7b1fa2","-."),
             ("all",  "EMRM All",      "#2e7d32","-")]

    def impact_fn(row, s):
        c, brake = row["code"], row["brake_v"]
        if s == "estop": return brake
        if s in ("dodge","all"): return 0.0 if row["path"] else brake
        if c in ("G","Y"): return 0.0
        if c == "O":
            mv = row["min_vc"]
            return mv if mv and not math.isnan(mv) else brake
        return brake

    def car_at_speed(s, vk):
        sub = [r for r in rows if r["v_kmh"] == vk]
        if not sub: return float("nan")
        return 100 * sum(impact_fn(r, s) == 0 for r in sub) / len(sub)

    def mit_at_speed(s, vk):
        sub = [r for r in rows if r["v_kmh"] == vk]
        if not sub: return float("nan")
        ok = sum((r["loss_noact"] - v2loss(impact_fn(r, s))) > 0 for r in sub)
        return 100 * ok / len(sub)

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.6))
    fig.subplots_adjust(wspace=0.38, left=0.10, right=0.98, top=0.88, bottom=0.18)

    for ax, metric_fn, ylabel, title in [
        (axes[0], car_at_speed, "Collision Avoidance Rate (%)", "CAR vs. Ego Speed"),
        (axes[1], mit_at_speed, "Mitigability Success (%)",    "Mitigability vs. Ego Speed"),
    ]:
        for key, lbl, clr, ls in SWEEP:
            ys = [metric_fn(key, vk) for vk in speeds_kmh]
            ax.plot(speeds_kmh, ys, color=clr, linestyle=ls,
                    linewidth=1.8, label=lbl)
        ax.set_xlim(min(speeds_kmh)-1, max(speeds_kmh)+1)
        ax.set_ylim(-5, 115)
        tick_step = 6
        ax.set_xticks(speeds_kmh[::tick_step])
        ax.set_xlabel("Ego Speed (km/h)", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=9, pad=4)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
        ax.grid(True, alpha=0.25, linewidth=0.5)

    axes[0].legend(fontsize=7, loc="upper right", framealpha=0.9)

    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Param sweep → {out_path}")


# ── figure: coverage comparison ───────────────────────────────────────────

def plot_coverage(n_feasible, out_path):
    METHODS = ["Trad.\nHARA", "Trad.\nSTPA", "Manual\n(12 scen.)", "EMRM\nFramework"]
    MCLR    = ["#90a4ae", "#78909c", "#546e7a", "#2e7d32"]

    haz_cov = [38, 75, 50, 100]
    uca_cov = [0,  100, 19, 100]
    # parameter space: EMRM tests all feasible cells, others test far fewer
    n_total = len(SPEEDS_KMH) * len(TTCS)   # total grid cells (incl. infeasible)
    par_cov = [
        round(100 * 3  / n_feasible),
        round(100 * 8  / n_feasible),
        round(100 * 12 / n_feasible),
        round(100 * n_feasible / n_feasible),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.4))
    fig.subplots_adjust(wspace=0.42, left=0.08, right=0.98, top=0.88, bottom=0.20)

    for ax, vals, title, ylim in [
        (axes[0], haz_cov, "Hazard Scenario\nCoverage (%)", 110),
        (axes[1], uca_cov, "UCA\nCoverage (%)", 110),
        (axes[2], par_cov, f"Parameter-Space\nCoverage (% of {n_feasible} cells)", 110),
    ]:
        xi   = np.arange(len(METHODS))
        bars = ax.bar(xi, vals, color=MCLR, edgecolor="white", lw=0.8, width=0.65)
        ax.set_ylim(0, ylim)
        ax.set_xticks(xi); ax.set_xticklabels(METHODS, fontsize=7)
        ax.set_title(title, fontsize=8.5, pad=4)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.6,
                    f"{v}%", ha="center", va="bottom", fontsize=7.5,
                    fontweight="bold")

    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Coverage comparison → {out_path}")


# ── main ──────────────────────────────────────────────────────────────────

def main():
    _ensure_pgm()
    if not os.path.exists(MAP_PGM):
        print("ERROR: map PGM not found"); return 1

    os.makedirs(RES_DIR, exist_ok=True)
    os.makedirs(FIGS_DIR, exist_ok=True)

    # load already-computed cells from prior run (resume support)
    done_keys: set[tuple] = set()
    if os.path.exists(CSV_OUT):
        with open(CSV_OUT) as f:
            for r in csv.DictReader(f):
                done_keys.add((int(r["v_kmh"]), float(r["TTC_s"])))
        print(f"Resuming: {len(done_keys)} cells already done.")

    base_grid, xmin, xmax, ymin, ymax = rrt.load_occupancy(MAP_PGM, PPM)

    # count total feasible
    n_feasible = sum(
        1 for v in SPEEDS_MS for t in TTCS
        if TRUCK_FRONT_Y - v * t >= MIN_START_Y
    )
    n_total = len(SPEEDS_MS) * len(TTCS)
    remaining = [
        (iv, it, v, t)
        for iv, v in enumerate(SPEEDS_MS)
        for it, t in enumerate(TTCS)
        if TRUCK_FRONT_Y - v * t >= MIN_START_Y
        and (SPEEDS_KMH[iv], t) not in done_keys
    ]
    print(f"Grid: {len(SPEEDS_KMH)} speeds × {len(TTCS)} TTCs = {n_total} cells")
    print(f"Feasible: {n_feasible}  |  To simulate: {len(remaining)}\n")

    # CSV header
    write_header = not os.path.exists(CSV_OUT) or len(done_keys) == 0
    csv_f  = open(CSV_OUT, "a", newline="")
    writer = csv.writer(csv_f)
    if write_header:
        writer.writerow(["v_ms","v_kmh","TTC_s","D_m","start_y",
                         "has_path","nodes_past","n_stop","n_coll",
                         "min_v_coll","outcome"])

    done_now = 0
    t0 = time.time()
    for iv, it, v, ttc in remaining:
        d       = v * ttc
        start_y = TRUCK_FRONT_Y - d
        start   = (START_X, start_y, START_PSI, v)
        random.seed(42)

        tree_g, tree_o, path, _, n_stop = rrt.rrt_double_tree(
            start, GOAL, base_grid, xmin, xmax, ymin, ymax, PPM,
            max_nodes=MAX_NODES, dt=DT, goal_sample_rate=0.55,
        )

        ng         = tree_g["nodes"]
        no         = tree_o["nodes"]
        nodes_past = sum(1 for n in ng if n[1] > 43.0)
        has_path   = len(path) > 0
        n_coll     = max(0, len(no) - n_stop)
        min_vc     = ""
        if n_coll > 0:
            coll = no[n_stop:]
            if coll:
                min_vc = f"{min(math.sqrt(n[3]**2+n[4]**2) for n in coll):.3f}"

        outcome = classify(tree_g, tree_o, path, n_stop, v, ttc)
        ostr    = {3:"FULLY_AVOIDABLE",2:"AGGRESSIVE_AVOID",
                   1:"MITIGATABLE",0:"NOT_MITIGATABLE"}[outcome]

        writer.writerow([f"{v:.3f}", SPEEDS_KMH[iv], f"{ttc:.2f}",
                         f"{d:.2f}", f"{start_y:.2f}",
                         str(has_path), str(nodes_past),
                         str(n_stop), str(n_coll), min_vc, ostr])
        csv_f.flush()
        done_keys.add((SPEEDS_KMH[iv], ttc))

        done_now += 1
        if done_now % 50 == 0:
            elapsed  = time.time() - t0
            per_cell = elapsed / done_now
            left     = len(remaining) - done_now
            eta_min  = left * per_cell / 60
            print(f"  [{done_now}/{len(remaining)}]  "
                  f"v={SPEEDS_KMH[iv]} km/h  TTC={ttc:.2f}s  "
                  f"→ {ostr[:12]}  "
                  f"ETA {eta_min:.0f} min")

    csv_f.close()
    print(f"\nAll {len(done_keys)} cells done. Generating figures…\n")

    # ── load all results and build matrix ────────────────────────────────
    result = {}
    with open(CSV_OUT) as f:
        for r in csv.DictReader(f):
            if not r.get("v_kmh","").strip(): continue
            vk  = int(r["v_kmh"])
            ttc = float(r["TTC_s"])
            o   = r.get("outcome","")
            code = {
                "FULLY_AVOIDABLE": FULLY_AVOIDABLE,
                "AGGRESSIVE_AVOID": AGGRESSIVE_AVOID,
                "MITIGATABLE": MITIGATABLE,
                "NOT_MITIGATABLE": NOT_MITIGATABLE,
            }.get(o, INFEASIBLE)
            result[(vk, ttc)] = {
                "code": {FULLY_AVOIDABLE:"G",AGGRESSIVE_AVOID:"Y",
                         MITIGATABLE:"O",NOT_MITIGATABLE:"R",
                         INFEASIBLE:"X"}[code],
                "code_int": code,
                "path": r.get("has_path","False") == "True",
                "brake_v": math.sqrt(max(0,
                    (vk/3.6)**2 - 2*6*float(r.get("D_m","0") or 0)
                )),
                "min_vc": float(r["min_v_coll"]) if r.get("min_v_coll") else float("nan"),
                "loss_noact": v2loss(vk/3.6),
                "v_kmh": vk,
                "v_ms": vk/3.6,
                "ttc": ttc,
            }

    nT, nV = len(TTCS), len(SPEEDS_KMH)
    outcome_matrix = np.full((nT, nV), INFEASIBLE, dtype=int)
    for it, ttc in enumerate(TTCS):
        for iv, vk in enumerate(SPEEDS_KMH):
            key = (vk, ttc)
            if key in result:
                outcome_matrix[it, iv] = result[key]["code_int"]

    # ── figures ──────────────────────────────────────────────────────────
    plot_heatmap(outcome_matrix, SPEEDS_KMH, TTCS,
                 os.path.join(FIGS_DIR, "mitigability_region.png"))

    rows = list(result.values())
    stats = plot_loss_comparison(
        rows, os.path.join(FIGS_DIR, "loss_comparison.png"))

    plot_param_sweep(rows, SPEEDS_KMH,
                     os.path.join(FIGS_DIR, "param_sweep.png"))

    n_sim = sum(1 for _ in rows)
    plot_coverage(n_sim, os.path.join(FIGS_DIR, "coverage_comparison.png"))

    # ── print summary ────────────────────────────────────────────────────
    codes = outcome_matrix[outcome_matrix != INFEASIBLE]
    total = len(codes)
    print(f"\n─── Summary ({total} simulated cells) ───")
    for code, lbl in [(3,"Fully avoidable "),(2,"Aggressive avoid"),
                      (1,"Mitigatable     "),(0,"Not mitigatable ")]:
        n = (codes == code).sum()
        print(f"  {lbl}: {n:4d}/{total}  ({100*n/total:.1f}%)")

    print(f"\n─── Table: loss_comparison ───")
    for k, (car, mv, mit, dl) in stats.items():
        print(f"  {k:6s}: CAR={car:.0f}%  Mean={mv:.1f}km/h  "
              f"Mit={mit:.0f}%  ΔLoss={dl:.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
