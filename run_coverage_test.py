#!/usr/bin/env python3
"""
run_coverage_test.py  –  (Ego Speed × TTC) mitigability coverage sweep.

Setup
─────
  • High-friction baseline (Pacejka unchanged).
  • Detection distance  D = v_ego × TTC  (fixed per cell; start position
    moves closer to the truck as TTC decreases or speed increases).
  • start_y = TRUCK_FRONT_Y − D   (all scenarios detect the hazard at the
    same *relative* distance to the truck; only speed+TTC vary).
  • 5 speed levels × 6 TTC levels = up to 30 cells; infeasible cells
    (start outside map) are skipped.

Outcomes (4 regions)
────────────────────
  GREEN   – Fully avoidable:     RRT finds clear path to goal
  YELLOW  – Aggressive avoidance: marginal path found (tight maneuvering)
  ORANGE  – Mitigatable:         fail-safe stop OR low-impact drift collision
  RED     – Not mitigatable:     unavoidable high-speed collision

Outputs
───────
  ../../figs/mitigability_region.png
  results/coverage_results.csv
  results/coverage_table_fragment.tex   (paste into tab:mitigation_by_param)
"""

from __future__ import annotations
import csv, math, os, random, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import run_rrt_tjunction as rrt
import ru_racer_dynamics as rrd

# ── paths ──────────────────────────────────────────────────────────────────
MAP_PNG  = os.path.join(THIS_DIR, "maps", "tjunction_map.png")
MAP_PGM  = os.path.join(THIS_DIR, "maps", "tjunction_map.pgm")
FIGS_DIR = os.path.join(THIS_DIR, "..", "..", "figs")
RES_DIR  = os.path.join(THIS_DIR, "results")
PPM      = 10.0

# ── scenario geometry ─────────────────────────────────────────────────────
# Distance measured to the TOP of the truck blocking zone (world y ≈ 27 m).
TRUCK_FRONT_Y = 27.0
START_X       = 41.8
START_PSI     = 1.57   # heading south (+y)
GOAL          = (44.0, 48.0)
MIN_START_Y   = 2.5    # map boundary guard

# ── sweep grid ────────────────────────────────────────────────────────────
# Speeds in m/s  (≈ 30 / 40 / 50 / 60 / 72 km/h)
SPEEDS = [8.3, 11.1, 13.9, 16.7, 20.0]
SPEED_LABELS = ["30 km/h\n(8.3 m/s)",
                "40 km/h\n(11.1 m/s)",
                "50 km/h\n(13.9 m/s)",
                "60 km/h\n(16.7 m/s)",
                "72 km/h\n(20.0 m/s)"]
SPEED_KMH   = [30, 40, 50, 60, 72]

# TTC values in seconds
TTCS = [0.25, 0.50, 0.75, 1.00, 1.50, 2.00]

# ── outcome codes ─────────────────────────────────────────────────────────
FULLY_AVOIDABLE  = 3   # green
AGGRESSIVE_AVOID = 2   # yellow
MITIGATABLE      = 1   # orange
NOT_MITIGATABLE  = 0   # red
INFEASIBLE       = -1  # gray

OUTCOME_LABEL = {
    FULLY_AVOIDABLE:  "Fully\nAvoidable",
    AGGRESSIVE_AVOID: "Aggressive\nAvoidance",
    MITIGATABLE:      "Mitigatable\n(drift/stop)",
    NOT_MITIGATABLE:  "Not\nMitigatable",
    INFEASIBLE:       "—",
}
OUTCOME_COLOR = {
    FULLY_AVOIDABLE:  "#2e7d32",   # dark green
    AGGRESSIVE_AVOID: "#f9a825",   # amber
    MITIGATABLE:      "#e65100",   # deep orange
    NOT_MITIGATABLE:  "#b71c1c",   # dark red
    INFEASIBLE:       "#bdbdbd",   # gray
}
OUTCOME_SHORT = {3: "G", 2: "Y", 1: "O", 0: "R", -1: "—"}


# ── classification ────────────────────────────────────────────────────────

def _classify(tree_g, tree_o, path, n_stop, v0, ttc) -> int:
    """
    Map RRT output to one of the 4 outcome categories.

    Thresholds
    ──────────
    nodes_past  : green-tree nodes with y > 43 m (past the truck blocking zone)
    has_path    : RRT reached the goal
    n_stop      : orange-tree emergency-stop successes
    min_v_coll  : minimum impact speed among orange collision nodes (m/s)
    can_stop    : analytic check  D = v*TTC  >  v²/(2·6)·1.1  (10% margin)
    """
    ng = tree_g["nodes"]
    no = tree_o["nodes"]

    nodes_past = sum(1 for n in ng if n[1] > 43.0)
    has_path   = len(path) > 0

    # analytic stop check (decel = 6 m/s², 10 % safety margin)
    d_detect  = v0 * ttc
    d_brake   = v0 * v0 / (2.0 * 6.0) * 1.10
    can_stop  = d_detect > d_brake

    # ── classify ──
    if has_path and nodes_past >= 12:
        return FULLY_AVOIDABLE
    if has_path or nodes_past >= 3:
        return AGGRESSIVE_AVOID

    # no clean path → check fail-safe
    if n_stop >= 2 or can_stop:
        return MITIGATABLE

    if len(no) > n_stop:
        coll_nodes = no[n_stop:]
        if coll_nodes:
            min_v = min(math.sqrt(n[3] ** 2 + n[4] ** 2) for n in coll_nodes)
            if min_v < 7.0:          # below ≈ 25 km/h impact → drift mitigation
                return MITIGATABLE

    return NOT_MITIGATABLE


# ── one RRT cell ──────────────────────────────────────────────────────────

def _run_cell(grid, xmin, xmax, ymin, ymax, v0, ttc):
    d = v0 * ttc
    start_y = TRUCK_FRONT_Y - d
    if start_y < MIN_START_Y:
        return None, None, [], 0, start_y

    start = (START_X, start_y, START_PSI, v0)
    random.seed(42)
    tree_g, tree_o, path, _, n_stop = rrt.rrt_double_tree(
        start, GOAL, grid, xmin, xmax, ymin, ymax, PPM,
        max_nodes=3000, dt=0.03, goal_sample_rate=0.55,
    )
    return tree_g, tree_o, path, n_stop, start_y


# ── helpers ───────────────────────────────────────────────────────────────

def _ensure_pgm():
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
        pass


def _success_rate(outcomes: list[int], target_codes: set[int]) -> float:
    """Fraction of feasible cells that are in target_codes."""
    feasible = [o for o in outcomes if o != INFEASIBLE]
    if not feasible:
        return float("nan")
    return 100.0 * sum(1 for o in feasible if o in target_codes) / len(feasible)


# ── plotting ──────────────────────────────────────────────────────────────

def _plot_heatmap(outcome_matrix, extra_data, speeds, ttcs,
                  speed_kmh, out_path):
    """
    outcome_matrix  : shape (n_ttc, n_speed), int codes
    extra_data      : dict[(i_ttc, i_spd)] → {'nodes_past', 'n_stop', 'path_len'}
    speeds          : list of m/s values
    ttcs            : list of TTC values
    speed_kmh       : list of km/h labels
    """
    nT, nV = len(ttcs), len(speeds)

    # ── custom colormap ───────────────────────────────────────────────────
    colors = [
        OUTCOME_COLOR[NOT_MITIGATABLE],   # 0
        OUTCOME_COLOR[MITIGATABLE],        # 1
        OUTCOME_COLOR[AGGRESSIVE_AVOID],   # 2
        OUTCOME_COLOR[FULLY_AVOIDABLE],    # 3
    ]
    cmap4 = ListedColormap(colors)
    norm4 = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], ncolors=4)

    # ── figure ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    fig.subplots_adjust(left=0.14, right=0.80, top=0.88, bottom=0.18)

    # Draw cells
    for i_t, ttc in enumerate(ttcs):
        for i_v, v in enumerate(speeds):
            code = outcome_matrix[i_t, i_v]
            color = OUTCOME_COLOR.get(code, OUTCOME_COLOR[INFEASIBLE])
            rect = plt.Rectangle(
                (i_v - 0.5, i_t - 0.5), 1.0, 1.0,
                facecolor=color, edgecolor="white", linewidth=1.0,
            )
            ax.add_patch(rect)

            if code == INFEASIBLE:
                ax.text(i_v, i_t, "—", ha="center", va="center",
                        fontsize=9, color="#555555")
            else:
                info = extra_data.get((i_t, i_v), {})
                plen  = info.get("path_len", 0)
                past  = info.get("nodes_past", 0)
                nstop = info.get("n_stop", 0)
                lbl   = OUTCOME_SHORT[code]
                ax.text(i_v, i_t + 0.22, lbl,
                        ha="center", va="center", fontsize=10,
                        fontweight="bold", color="white")
                detail = f"path={plen}" if plen > 0 else f"stop={nstop}"
                ax.text(i_v, i_t - 0.22, detail,
                        ha="center", va="center", fontsize=6.5,
                        color="white", alpha=0.85)

    # ── analytic stop boundary (TTC = v / 12, with 10% margin) ───────────
    v_cont = np.linspace(min(speeds) * 0.9, max(speeds) * 1.1, 200)
    # TTC_stop = v / (2*6) * 1.1 = v/10.909
    ttc_stop = v_cont / (2.0 * 6.0) * 1.10
    # convert to axis coords: x = (v - speeds[0])/(speeds[-1]-speeds[0])*(nV-1)
    def v2x(v):
        return (v - speeds[0]) / (speeds[-1] - speeds[0]) * (nV - 1)
    def t2y(t):
        return (t - ttcs[0]) / (ttcs[-1] - ttcs[0]) * (nT - 1)

    ax.plot(v2x(v_cont), t2y(ttc_stop),
            color="white", linewidth=2.0, linestyle="--",
            label="Emergency stop boundary\n$TTC = v/(2a_{\\mathrm{max}})$",
            zorder=5)

    # ── axes ──────────────────────────────────────────────────────────────
    ax.set_xlim(-0.5, nV - 0.5)
    ax.set_ylim(-0.5, nT - 0.5)
    ax.set_xticks(range(nV))
    ax.set_xticklabels([f"{kmh}" for kmh in speed_kmh], fontsize=9)
    ax.set_yticks(range(nT))
    ax.set_yticklabels([f"{t:.2f}" for t in ttcs], fontsize=9)
    ax.set_xlabel("Ego Speed (km/h)", fontsize=10, labelpad=6)
    ax.set_ylabel("Time-to-Collision TTC (s)", fontsize=10, labelpad=6)
    ax.set_title(
        "Mitigability Region — Ego Speed × TTC\n"
        "(High friction, dry road, $\\mu\\approx1.0$, fixed detection distance)",
        fontsize=10, pad=8,
    )

    # ── detection-distance iso-lines ──────────────────────────────────────
    for d_iso in [5, 10, 15, 20]:
        # TTC = D / v  →  i_t such that ttcs[i_t] == D/v
        xs, ys = [], []
        for i_v, v in enumerate(speeds):
            t_iso = d_iso / v
            if ttcs[0] <= t_iso <= ttcs[-1]:
                xs.append(i_v)
                ys.append(t2y(t_iso))
        if len(xs) >= 2:
            ax.plot(xs, ys, color="white", linewidth=0.8,
                    linestyle=":", alpha=0.6, zorder=4)
            ax.text(xs[-1] + 0.08, ys[-1], f"D={d_iso}m",
                    fontsize=6.5, color="white", alpha=0.8, va="center")

    # ── legend ────────────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(facecolor=OUTCOME_COLOR[FULLY_AVOIDABLE],
                       edgecolor="white", label="Fully avoidable (Dodge)"),
        mpatches.Patch(facecolor=OUTCOME_COLOR[AGGRESSIVE_AVOID],
                       edgecolor="white", label="Aggressive avoidance"),
        mpatches.Patch(facecolor=OUTCOME_COLOR[MITIGATABLE],
                       edgecolor="white", label="Mitigatable (stop/drift)"),
        mpatches.Patch(facecolor=OUTCOME_COLOR[NOT_MITIGATABLE],
                       edgecolor="white", label="Not mitigatable"),
        Line2D([0], [0], color="white", linewidth=2, linestyle="--",
               label="Stop boundary"),
    ]
    ax.legend(handles=legend_patches, loc="upper left",
              bbox_to_anchor=(1.02, 1.0), fontsize=8,
              framealpha=0.95, borderpad=0.6)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved heatmap → {out_path}")


# ── LaTeX table fragment ──────────────────────────────────────────────────

def _build_latex_table(outcome_matrix, speeds, ttcs, speed_kmh) -> str:
    """
    Compute success rates by parameter group for tab:mitigation_by_param.

    Dodge success         = FULLY_AVOIDABLE (green)
    Drift-to-avoid        = FULLY_AVOIDABLE + AGGRESSIVE_AVOID (green+yellow)
    Drift-to-accident     = any mitigated  (green+yellow+orange)
    Emergency stop        = analytically possible (TTC > v/12*1.1)
    """
    def sr(outcomes, codes):
        return _success_rate(outcomes, codes)

    # group speeds into Low/Med/High
    # Low ≤ 40 km/h, Med 41-60, High > 60
    def speed_group(i):
        k = speed_kmh[i]
        if k <= 40:   return "Low"
        if k <= 60:   return "Med"
        return "High"

    # group TTCs into Low/Med/High
    # Low ≤ 0.5, Med 0.51-1.0, High > 1.0
    def ttc_group(j):
        t = ttcs[j]
        if t <= 0.50:  return "Low"
        if t <= 1.00:  return "Med"
        return "High"

    nT, nV = outcome_matrix.shape

    groups = {"Low": [], "Med": [], "High": []}
    for i in range(nV):
        grp = speed_group(i)
        row_outcomes = [outcome_matrix[j, i] for j in range(nT)]
        groups[grp].extend(row_outcomes)

    def fmt(o_list, codes):
        r = _success_rate(o_list, codes)
        return "N/A" if math.isnan(r) else f"{r:.0f}"

    DODGE  = {FULLY_AVOIDABLE}
    D2VOID = {FULLY_AVOIDABLE, AGGRESSIVE_AVOID}
    D2ACC  = {FULLY_AVOIDABLE, AGGRESSIVE_AVOID, MITIGATABLE}

    # emergency stop: analytic
    def estop_rate(indices_v, indices_t):
        ok = total = 0
        for iv in indices_v:
            v = speeds[iv]
            for it in indices_t:
                if outcome_matrix[it, iv] == INFEASIBLE:
                    continue
                total += 1
                ttc = ttcs[it]
                if v * ttc > v * v / 12.0 * 1.1:
                    ok += 1
        return (100 * ok // total) if total else float("nan")

    # per-speed group rows
    spd_rows = []
    for grp in ["Low", "Med", "High"]:
        oo = groups[grp]
        ivs = [i for i in range(nV) if speed_group(i) == grp]
        its = list(range(nT))
        es  = estop_rate(ivs, its)
        es_s = "N/A" if math.isnan(es) else f"{es:.0f}"
        n = len([o for o in oo if o != INFEASIBLE])
        spd_rows.append((grp,
                         fmt(oo, DODGE),
                         fmt(oo, D2VOID),
                         fmt(oo, D2ACC),
                         es_s, n))

    # per-TTC group rows
    ttc_groups = {"Low": [], "Med": [], "High": []}
    for j in range(nT):
        grp = ttc_group(j)
        col_outcomes = [outcome_matrix[j, i] for i in range(nV)]
        ttc_groups[grp].extend(col_outcomes)

    ttc_rows = []
    for grp in ["Low", "Med", "High"]:
        oo = ttc_groups[grp]
        its = [j for j in range(nT) if ttc_group(j) == grp]
        ivs = list(range(nV))
        es  = estop_rate(ivs, its)
        es_s = "N/A" if math.isnan(es) else f"{es:.0f}"
        n = len([o for o in oo if o != INFEASIBLE])
        ttc_rows.append((grp,
                         fmt(oo, DODGE),
                         fmt(oo, D2VOID),
                         fmt(oo, D2ACC),
                         es_s, n))

    def row_str(label, r):
        grp, d, da, dat, es, n = r
        return (f"\\textit{{{label}: {grp}}} & {d} & {da} & {dat} & {es} "
                f"& {n} \\\\ \\hline")

    lines = []
    lines.append(r"\hline")
    lines.append(r"\textbf{Parameter Level} & \textbf{Dodge} & "
                 r"\textbf{Drift to Avoid} & \textbf{Drift to Accident} & "
                 r"\textbf{Emergency Stop} & \textbf{No.\ Tests} \\ \hline")
    for r in spd_rows:
        lines.append(row_str("Ego Speed", r))
    for r in ttc_rows:
        lines.append(row_str("TTC", r))
    # Friction row – high friction only (single level)
    all_o = outcome_matrix.flatten().tolist()
    f_d   = fmt(all_o, DODGE)
    f_da  = fmt(all_o, D2VOID)
    f_dat = fmt(all_o, D2ACC)
    n_feas = len([o for o in all_o if o != INFEASIBLE])
    lines.append(
        f"\\textit{{Friction: High (dry)}} & {f_d} & {f_da} & {f_dat} "
        f"& 100 & {n_feas} \\\\ \\hline"
    )
    return "\n".join(lines)


# ── main ──────────────────────────────────────────────────────────────────

def main() -> int:
    _ensure_pgm()
    if not os.path.exists(MAP_PGM):
        print("ERROR: map PGM not found.")
        return 1

    base_grid, xmin, xmax, ymin, ymax = rrt.load_occupancy(MAP_PGM, PPM)
    os.makedirs(RES_DIR, exist_ok=True)
    os.makedirs(FIGS_DIR, exist_ok=True)

    nT, nV = len(TTCS), len(SPEEDS)
    outcome_matrix = np.full((nT, nV), INFEASIBLE, dtype=int)
    extra_data: dict = {}

    n_feasible = sum(
        1 for v in SPEEDS for ttc in TTCS
        if TRUCK_FRONT_Y - v * ttc >= MIN_START_Y
    )
    done = 0

    csv_rows = [["v_ms", "v_kmh", "TTC_s", "D_m", "start_y",
                 "has_path", "nodes_past", "n_stop", "n_coll",
                 "min_v_coll", "outcome"]]

    print(f"Coverage sweep: {nV} speeds × {nT} TTCs = {n_feasible} feasible cells\n")

    for i_t, ttc in enumerate(TTCS):
        for i_v, v in enumerate(SPEEDS):
            d = v * ttc
            start_y = TRUCK_FRONT_Y - d
            if start_y < MIN_START_Y:
                print(f"  SKIP  v={v:.1f} m/s  TTC={ttc:.2f}s  "
                      f"(D={d:.1f}m, start_y={start_y:.1f} < {MIN_START_Y})")
                csv_rows.append([f"{v:.1f}", "", f"{ttc:.2f}", f"{d:.1f}",
                                  f"{start_y:.1f}", "", "", "", "", "", "INFEASIBLE"])
                continue

            done += 1
            print(f"[{done:02d}/{n_feasible}] "
                  f"v={v:.1f} m/s ({SPEED_KMH[i_v]} km/h)  "
                  f"TTC={ttc:.2f}s  D={d:.1f}m  start_y={start_y:.1f}")

            tree_g, tree_o, path, n_stop, _ = _run_cell(
                base_grid, xmin, xmax, ymin, ymax, v, ttc
            )

            ng = tree_g["nodes"]
            no = tree_o["nodes"]
            nodes_past = sum(1 for n in ng if n[1] > 43.0)
            has_path   = len(path) > 0
            n_coll     = len(no) - n_stop
            min_v_coll = float("nan")
            if n_coll > 0:
                coll = no[n_stop:]
                if coll:
                    min_v_coll = min(math.sqrt(n[3] ** 2 + n[4] ** 2)
                                     for n in coll)

            outcome = _classify(tree_g, tree_o, path, n_stop, v, ttc)
            outcome_matrix[i_t, i_v] = outcome
            extra_data[(i_t, i_v)] = {
                "path_len":   len(path),
                "nodes_past": nodes_past,
                "n_stop":     n_stop,
            }

            outcome_str = {3:"FULLY_AVOIDABLE", 2:"AGGRESSIVE_AVOID",
                           1:"MITIGATABLE", 0:"NOT_MITIGATABLE"}[outcome]
            print(f"       → {outcome_str}  "
                  f"path={len(path)}  past={nodes_past}  "
                  f"stop={n_stop}/{len(no)}  "
                  f"min_v_coll={min_v_coll:.1f}\n" if not math.isnan(min_v_coll)
                  else f"       → {outcome_str}  "
                  f"path={len(path)}  past={nodes_past}  "
                  f"stop={n_stop}/{len(no)}\n")

            min_vc_s = f"{min_v_coll:.2f}" if not math.isnan(min_v_coll) else ""
            csv_rows.append([
                f"{v:.2f}", str(SPEED_KMH[i_v]), f"{ttc:.2f}",
                f"{d:.1f}", f"{start_y:.1f}",
                str(has_path), str(nodes_past),
                str(n_stop), str(n_coll), min_vc_s, outcome_str,
            ])

    # ── save CSV ──────────────────────────────────────────────────────────
    csv_path = os.path.join(RES_DIR, "coverage_results.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows(csv_rows)
    print(f"CSV → {csv_path}")

    # ── heatmap ───────────────────────────────────────────────────────────
    hmap_path = os.path.join(FIGS_DIR, "mitigability_region.png")
    _plot_heatmap(outcome_matrix, extra_data, SPEEDS, TTCS,
                  SPEED_KMH, hmap_path)

    # ── LaTeX table fragment ──────────────────────────────────────────────
    tex = _build_latex_table(outcome_matrix, SPEEDS, TTCS, SPEED_KMH)
    tex_path = os.path.join(RES_DIR, "coverage_table_fragment.tex")
    with open(tex_path, "w") as f:
        f.write(tex + "\n")
    print(f"LaTeX fragment → {tex_path}")

    # ── print table to stdout ─────────────────────────────────────────────
    print("\n─── Table fragment (tab:mitigation_by_param) ───")
    print(tex)

    # ── summary ───────────────────────────────────────────────────────────
    codes = outcome_matrix[outcome_matrix != INFEASIBLE]
    total = len(codes)
    print(f"\n─── Summary ({total} feasible cells) ───")
    for code, label in [(3, "Fully avoidable "), (2, "Aggressive avoid"),
                        (1, "Mitigatable     "), (0, "Not mitigatable ")]:
        n = (codes == code).sum()
        print(f"  {label}: {n}/{total}  ({100*n/total:.0f}%)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
