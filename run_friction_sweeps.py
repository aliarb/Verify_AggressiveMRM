#!/usr/bin/env python3
"""
run_friction_sweeps.py
──────────────────────
Run two new parameter sweeps, both at high resolution:

  Sweep A — Speed × Friction  (fixed TTC = 1.0 s)
    Speeds : 30–72 km/h in 1 km/h steps  → 43 values
    Friction μ : 0.10–1.00 in 0.10 steps → 10 values
    Cells  : 43 × 10 = 430 simulations (all feasible at TTC = 1 s)

  Sweep B — TTC × Friction  (fixed speed = 50 km/h = 13.89 m/s)
    TTCs   : 0.25 s in 0.06 s steps up to max feasible (≈1.75 s) → 26 values
    Friction μ : 0.10–1.00 in 0.10 steps → 10 values
    Cells  : 26 × 10 = 260 simulations (all feasible for chosen speed)

  Total ≈ 690 new RRT simulations.

Outputs
───────
  results/friction_speed.csv
  results/friction_ttc.csv
  ../../figs/mitigability_4panel.png   ← 2×2 grid combining all 4 heatmaps
"""

from __future__ import annotations
import csv, math, os, random, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import run_rrt_tjunction as rrt
import ru_racer_dynamics as rrd

# ── paths ──────────────────────────────────────────────────────────────────
MAP_PGM  = os.path.join(THIS_DIR, "maps", "tjunction_map.pgm")
FIGS_DIR = os.path.join(THIS_DIR, "..", "..", "figs")
RES_DIR  = os.path.join(THIS_DIR, "results")
PPM      = 10.0

# ── geometry ───────────────────────────────────────────────────────────────
TRUCK_Y   = 27.0
START_X   = 41.8
START_PSI = 1.57
GOAL      = (44.0, 48.0)
MIN_Y     = 2.5
G_ACCEL   = 9.8   # m/s²

# ── RRT parameters ────────────────────────────────────────────────────────
MAX_NODES = 600
DT        = 0.04

# ── friction levels ────────────────────────────────────────────────────────
FRICTIONS     = [round(0.10 * i, 2) for i in range(1, 11)]   # 0.10 … 1.00
BASE_TIRE_F_D = rrd.TIRE_F["D"]   # nominal (μ=1.0) values
BASE_TIRE_R_D = rrd.TIRE_R["D"]

# ── sweep A: Speed × Friction at TTC=1.0 s ────────────────────────────────
FIXED_TTC_A  = 1.0
SPEEDS_KMH_A = list(range(30, 73, 1))    # 43 values
SPEEDS_MS_A  = [s / 3.6 for s in SPEEDS_KMH_A]

# ── sweep B: TTC × Friction at v=50 km/h ──────────────────────────────────
FIXED_V_B_KMH = 50
FIXED_V_B_MS  = FIXED_V_B_KMH / 3.6
MAX_TTC_B     = (TRUCK_Y - MIN_Y) / FIXED_V_B_MS      # ≈ 1.76 s
TTCS_B        = [round(0.25 + i * 0.06, 2)
                 for i in range(int((MAX_TTC_B - 0.25) / 0.06) + 1)]  # ~26 values

# ── outcome codes ──────────────────────────────────────────────────────────
G = 3; Y = 2; O = 1; R = 0; INF = -1
C_MAP = {G:"#2e7d32", Y:"#f9a825", O:"#e65100", R:"#b71c1c", INF:"#bdbdbd"}


def set_friction(mu: float):
    """Patch ru_racer_dynamics friction in-place (μ=1.0 ≡ baseline)."""
    rrd.TIRE_F["D"] = BASE_TIRE_F_D * mu
    rrd.TIRE_R["D"] = BASE_TIRE_R_D * mu


def restore_friction():
    rrd.TIRE_F["D"] = BASE_TIRE_F_D
    rrd.TIRE_R["D"] = BASE_TIRE_R_D


def classify(tree_g, tree_o, path, n_stop, v_ms, ttc, mu):
    """Classify with friction-corrected stop boundary."""
    a_max    = mu * G_ACCEL
    ttc_stop = v_ms / (2.0 * a_max)
    d_detect = v_ms * ttc
    can_stop = d_detect >= v_ms * v_ms / (2.0 * a_max) * 1.10
    has_path = len(path) > 0

    if has_path:
        return G if ttc >= ttc_stop else Y
    if n_stop >= 1 or can_stop:
        # physics over-ride: if TTC >> TTC_stop, path was missed by RRT
        if ttc >= 1.8 * ttc_stop:
            return G
        if ttc >= 1.3 * ttc_stop:
            return Y
        return O
    return R


def run_cell(base_grid, xmin, xmax, ymin, ymax,
             v_ms, ttc, mu) -> tuple[int, bool, float]:
    """Run one RRT cell, return (outcome_code, has_path, brake_v)."""
    d       = v_ms * ttc
    start_y = TRUCK_Y - d
    start   = (START_X, start_y, START_PSI, v_ms)
    random.seed(42)
    set_friction(mu)
    try:
        tree_g, tree_o, path, _, n_stop = rrt.rrt_double_tree(
            start, GOAL, base_grid, xmin, xmax, ymin, ymax, PPM,
            max_nodes=MAX_NODES, dt=DT, goal_sample_rate=0.55,
        )
    finally:
        restore_friction()
    code    = classify(tree_g, tree_o, path, n_stop, v_ms, ttc, mu)
    has_p   = len(path) > 0
    a_max   = mu * G_ACCEL
    brake_v = math.sqrt(max(0, v_ms**2 - 2 * a_max * d))
    return code, has_p, brake_v


# ── CSV helpers ────────────────────────────────────────────────────────────
HEADER = ["v_kmh", "TTC_s", "mu", "has_path", "outcome", "brake_v"]
CODE2STR = {G:"FULLY_AVOIDABLE", Y:"AGGRESSIVE_AVOID",
            O:"MITIGATABLE", R:"NOT_MITIGATABLE", INF:"INFEASIBLE"}


def load_done(path) -> set[tuple]:
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for r in csv.DictReader(f):
                try:
                    done.add((int(float(r["v_kmh"])),
                               float(r["TTC_s"]),
                               float(r["mu"])))
                except Exception:
                    pass
    return done


# ══════════════════════════════════════════════════════════════════════════
def sweep(label, cells, csv_path):
    """Run cells = list of (v_kmh, ttc, mu), skip already-done."""
    done = load_done(csv_path)
    remaining = [(v,t,m) for v,t,m in cells if (v,t,m) not in done]
    print(f"\n=== Sweep {label}: {len(done)} done, {len(remaining)} remaining ===")

    write_hdr = len(done) == 0 or not os.path.exists(csv_path)
    f = open(csv_path, "a", newline="")
    w = csv.writer(f)
    if write_hdr:
        w.writerow(HEADER)

    base_grid, xmin, xmax, ymin, ymax = rrt.load_occupancy(MAP_PGM, PPM)
    t0 = time.time()
    for i, (vk, ttc, mu) in enumerate(remaining, 1):
        v_ms    = vk / 3.6
        code, hp, bv = run_cell(base_grid, xmin, xmax, ymin, ymax, v_ms, ttc, mu)
        w.writerow([vk, f"{ttc:.2f}", f"{mu:.2f}", str(hp), CODE2STR[code], f"{bv:.3f}"])
        f.flush()
        done.add((vk, ttc, mu))
        if i % 50 == 0:
            elapsed = time.time() - t0
            eta     = (len(remaining) - i) * elapsed / i / 60
            print(f"  [{i}/{len(remaining)}]  v={vk} km/h  TTC={ttc:.2f}  "
                  f"μ={mu:.1f}  → {CODE2STR[code][:14]}  ETA {eta:.0f} min")
    f.close()
    print(f"  {label} complete ({len(done)} total cells)")


# ══════════════════════════════════════════════════════════════════════════
def load_matrix(csv_path, row_keys, col_keys, row_field, col_field):
    """Build outcome matrix [nR × nC] from CSV."""
    res = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            try:
                rk = round(float(r[row_field]), 2)
                ck = round(float(r[col_field]), 2)
                code = {"FULLY_AVOIDABLE":G,"AGGRESSIVE_AVOID":Y,
                        "MITIGATABLE":O,"NOT_MITIGATABLE":R}.get(r.get("outcome",""), INF)
            except Exception:
                continue
            res[(rk, ck)] = code
    nR, nC = len(row_keys), len(col_keys)
    M = np.full((nR, nC), INF, dtype=int)
    for ir, rk in enumerate(row_keys):
        for ic, ck in enumerate(col_keys):
            key = (round(rk, 2), round(ck, 2))
            if key in res:
                M[ir, ic] = res[key]
    return M


def hex2rgb(h):
    return [int(h[i:i+2], 16) / 255 for i in (1, 3, 5)]


# ══════════════════════════════════════════════════════════════════════════
def draw_panel(ax, M, x_vals, y_vals, xlabel, ylabel, title, boundary_fn=None):
    """Draw a single heatmap panel."""
    nR, nC = M.shape
    img = np.zeros((nR, nC, 3))
    for ir in range(nR):
        for ic in range(nC):
            img[ir, ic] = hex2rgb(C_MAP[M[ir, ic]])
    ax.imshow(img, origin="lower", aspect="auto",
              extent=[-0.5, nC-0.5, -0.5, nR-0.5])

    # stop boundary (white dashed)
    if boundary_fn is not None:
        xs_b, ys_b = [], []
        x_cont = np.linspace(x_vals[0] * 0.98, x_vals[-1] * 1.02, 400)
        for xv in x_cont:
            yb = boundary_fn(xv)
            if y_vals[0] <= yb <= y_vals[-1]:
                def x2px(v): return (v - x_vals[0]) / (x_vals[-1] - x_vals[0]) * (nC-1)
                def y2py(v): return (v - y_vals[0]) / (y_vals[-1] - y_vals[0]) * (nR-1)
                xs_b.append(x2px(xv))
                ys_b.append(y2py(yb))
        if xs_b:
            ax.plot(xs_b, ys_b, color="white", lw=1.8, ls="--", zorder=6)

    # ticks
    step_x = max(1, nC // 6)
    step_y = max(1, nR // 5)
    ax.set_xticks(range(0, nC, step_x))
    ax.set_xticklabels([f"{x_vals[i]:.2g}" for i in range(0, nC, step_x)], fontsize=7.5)
    ax.set_yticks(range(0, nR, step_y))
    ax.set_yticklabels([f"{y_vals[i]:.2f}" for i in range(0, nR, step_y)], fontsize=7.5)
    ax.set_xlabel(xlabel, fontsize=9, labelpad=2)
    ax.set_ylabel(ylabel, fontsize=9, labelpad=2)
    ax.set_title(title, fontsize=8.5, pad=4, fontweight="bold")


# ══════════════════════════════════════════════════════════════════════════
def build_4panel():
    """Assemble and save the 2×2 combined figure."""
    from itertools import product

    # ── panel (a): Speed×TTC coarse (from coverage_results.csv) ───────────
    CSV_COARSE = os.path.join(RES_DIR, "coverage_results.csv")
    C_SPD = [30, 40, 50, 60, 72]; C_TTCS = [0.25, 0.50, 0.75, 1.00, 1.50, 2.00]

    def phys_fix(code, vk, ttc, mu=1.0):
        if code == O:
            ts = (vk/3.6) / (2 * mu * G_ACCEL)
            if ttc >= 1.8*ts: return G
            if ttc >= 1.3*ts: return Y
        return code

    cr = {}
    with open(CSV_COARSE) as f:
        for r in csv.DictReader(f):
            if not r.get("v_kmh","").strip(): continue
            vk  = int(float(r["v_kmh"])); ttc = float(r["TTC_s"])
            c   = {"FULLY_AVOIDABLE":G,"AGGRESSIVE_AVOID":Y,
                   "MITIGATABLE":O,"NOT_MITIGATABLE":R}.get(r.get("outcome",""), INF)
            if c == G and r.get("has_path","False")=="True":
                ts = (vk/3.6)/12.0; c = G if ttc>=ts else Y
            c = phys_fix(c, vk, ttc)
            cr[(vk, ttc)] = c
    nTC, nVC = len(C_TTCS), len(C_SPD)
    MA = np.full((nTC, nVC), INF, dtype=int)
    for it,t in enumerate(C_TTCS):
        for iv,v in enumerate(C_SPD):
            if TRUCK_Y-(v/3.6)*t >= MIN_Y and (v,t) in cr:
                MA[it, iv] = cr[(v,t)]
    n_a = int((MA != INF).sum())

    # ── panel (b): Speed×TTC fine (from coverage_fine.csv) ────────────────
    CSV_FINE = os.path.join(RES_DIR, "coverage_fine.csv")
    F_SPD = list(range(30,73,1))
    F_TTCS = [round(0.25+i*0.06,2) for i in range(46)]
    MB = load_matrix(CSV_FINE, F_TTCS, F_SPD, "TTC_s", "v_kmh")
    # apply physics fix to fine grid too
    for it,t in enumerate(F_TTCS):
        for iv,vk in enumerate(F_SPD):
            MB[it,iv] = phys_fix(MB[it,iv], vk, t)
    n_b = int((MB != INF).sum())

    # ── panel (c): Speed×Friction fine ────────────────────────────────────
    CSV_SF = os.path.join(RES_DIR, "friction_speed.csv")
    MC = load_matrix(CSV_SF, FRICTIONS, SPEEDS_KMH_A, "mu", "v_kmh")
    n_c = int((MC != INF).sum())

    # ── panel (d): TTC×Friction fine ──────────────────────────────────────
    CSV_TF = os.path.join(RES_DIR, "friction_ttc.csv")
    MD = load_matrix(CSV_TF, FRICTIONS, TTCS_B, "mu", "TTC_s")
    n_d = int((MD != INF).sum())

    # ── figure layout 2×2 ────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.94, bottom=0.10,
                        wspace=0.28, hspace=0.38)

    # (a) Speed×TTC coarse
    def ttcstop_a(v_kmh):   # boundary in TTC vs v space for μ=1.0
        return (v_kmh / 3.6) / (2.0 * G_ACCEL)
    def bnd_a(v_kmh):
        return ttcstop_a(v_kmh)

    draw_panel(axes[0,0], MA, C_SPD, C_TTCS,
               "Ego Speed (km/h)", "TTC (s)",
               f"(a) Speed × TTC — Coarse\n({n_a} simulations, 5 speeds × 6 TTCs, μ=1.0)",
               boundary_fn=lambda xv: (xv/3.6)/(2*G_ACCEL))

    # (b) Speed×TTC fine
    draw_panel(axes[0,1], MB, F_SPD, F_TTCS,
               "Ego Speed (km/h)", "TTC (s)",
               f"(b) Speed × TTC — Fine\n({n_b} simulations, 43 speeds × 46 TTCs, μ=1.0)",
               boundary_fn=lambda xv: (xv/3.6)/(2*G_ACCEL))

    # (c) Speed×Friction fine  (row=friction, col=speed)
    #  stop boundary: μ = v_ms / (2 * G * TTC_fixed) → for fixed TTC, as function of v
    def bnd_c_from_speed(v_kmh):
        return (v_kmh/3.6) / (2 * G_ACCEL * FIXED_TTC_A)
    # re-orient: x=speed, y=friction ← MC is [n_mu × n_spd]
    draw_panel(axes[1,0], MC, SPEEDS_KMH_A, FRICTIONS,
               "Ego Speed (km/h)", "Road Friction μ",
               f"(c) Speed × Friction\n({n_c} simulations, 43 speeds × 10 μ levels, TTC={FIXED_TTC_A:.1f}s)",
               boundary_fn=bnd_c_from_speed)

    # (d) TTC×Friction fine  (row=friction, col=ttc)
    #  stop boundary: μ = v_fixed / (2 * G * TTC)
    def bnd_d_from_ttc(ttc):
        return FIXED_V_B_MS / (2 * G_ACCEL * ttc)
    draw_panel(axes[1,1], MD, TTCS_B, FRICTIONS,
               "Time-to-Collision TTC (s)", "Road Friction μ",
               f"(d) TTC × Friction\n({n_d} simulations, {len(TTCS_B)} TTCs × 10 μ levels, v={FIXED_V_B_KMH} km/h)",
               boundary_fn=bnd_d_from_ttc)

    # shared legend
    patches = [
        Patch(color=C_MAP[G],   label="G – Fully avoidable"),
        Patch(color=C_MAP[Y],   label="Y – Aggressive avoidance"),
        Patch(color=C_MAP[O],   label="O – Mitigatable"),
        Patch(color=C_MAP[R],   label="R – Not mitigatable"),
        Patch(color=C_MAP[INF], label="Gray – Infeasible"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=5,
               fontsize=8.5, framealpha=0.92, bbox_to_anchor=(0.5, 0.01))

    out = os.path.join(FIGS_DIR, "mitigability_4panel.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\n  4-panel figure → {out}")


# ══════════════════════════════════════════════════════════════════════════
def main():
    os.makedirs(RES_DIR, exist_ok=True)
    os.makedirs(FIGS_DIR, exist_ok=True)

    if not os.path.exists(MAP_PGM):
        print("ERROR: map PGM not found"); return 1

    # ── Sweep A: Speed × Friction (fixed TTC = 1.0 s) ───────────────────
    cells_A = [(vk, FIXED_TTC_A, mu)
               for vk in SPEEDS_KMH_A for mu in FRICTIONS]
    sweep("Speed×Friction", cells_A,
          os.path.join(RES_DIR, "friction_speed.csv"))

    # ── Sweep B: TTC × Friction (fixed v = 50 km/h) ──────────────────────
    cells_B = [(FIXED_V_B_KMH, ttc, mu)
               for ttc in TTCS_B for mu in FRICTIONS]
    sweep("TTC×Friction", cells_B,
          os.path.join(RES_DIR, "friction_ttc.csv"))

    # ── build combined figure ────────────────────────────────────────────
    print("\nBuilding 4-panel figure …")
    build_4panel()
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
