#!/usr/bin/env python3
"""
plot_revision_figs.py  -  figures for the revised manuscript, drawn only from
simulated outcomes (no analytical region overlays are used to colour cells).

  ../figs/mitigability_rrt.png   outcome maps from the RRT campaigns
  ../figs/param_sweep_rrt.png    CAR vs ego speed, and boundary refinement
                                 with multi-seed Wilson intervals

Emergency braking is the simulated baseline of run_sim_brake.py
(results/sim_brake.csv): straight-line braking with the RRT vehicle model and
collision checker.  Cell classes:
  G  braking stops without contact
  Y  braking collides, the RRT goal tree finds a collision-free path
  O  no avoidance; simulated braking impact speed < V_MIT
  R  no avoidance; impact speed >= V_MIT
  X  excluded (start position outside the map)
"""
from __future__ import annotations
import csv, math, os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(THIS_DIR, "results")
FIGS = os.path.join(THIS_DIR, "..", "figs")

V_MIT = 7.0
SPEEDS = list(range(30, 73))
TTCS = [round(0.25 + 0.06 * i, 2) for i in range(46)]
MUS = [round(0.1 * i, 2) for i in range(1, 11)]
TTCS_B = [round(0.25 + 0.06 * i, 2) for i in range(26)]

G, Y, O, R, X = 4, 3, 2, 1, 0
COL = {G: "#2e7d32", Y: "#f9a825", O: "#e65100", R: "#b71c1c", X: "#d9d9d9"}
LAB = {G: "Avoidable by emergency braking",
       Y: "Avoidable only by EMRM steering",
       O: r"No path within budget; impact speed $<7$ m/s",
       R: r"No path within budget; impact speed $\geq 7$ m/s",
       X: "Excluded (start outside map)"}


def load_brake():
    out = {}
    for r in csv.DictReader(open(os.path.join(RES, "sim_brake.csv"))):
        out[(r["dataset"], int(r["v_kmh"]), round(float(r["TTC_s"]), 2), round(float(r["mu"]), 2))] = float(r["impact_v"])
    return out


SB = load_brake()


def code(has_path, iv):
    if iv == 0.0:
        return G
    if has_path:
        return Y
    return O if iv < V_MIT else R


def load(name, csvname, rowkeys, colkeys, rowf, colf):
    M = np.full((len(rowkeys), len(colkeys)), X)
    for r in csv.DictReader(open(os.path.join(RES, csvname))):
        vk, t = int(float(r["v_kmh"])), round(float(r["TTC_s"]), 2)
        mu = round(float(r.get("mu", 1.0)), 2)
        rv, cv = round(float(r[rowf]), 2), round(float(r[colf]), 2)
        M[rowkeys.index(rv), colkeys.index(cv)] = code(r["has_path"] == "True", SB[(name, vk, t, mu)])
    return M


def step(ax, M, classes, **kw):
    px, py = [], []
    for j in range(M.shape[1]):
        rows = [i for i in range(M.shape[0]) if M[i, j] in classes]
        if rows:
            px += [j - 0.5, j + 0.5]; py += [min(rows) - 0.5] * 2
    ax.plot(px, py, **kw)


def draw(ax, M, xv, yv, xl, yl, title, boundaries=False):
    img = np.array([[matplotlib.colors.to_rgb(COL[c]) for c in row] for row in M])
    ax.imshow(img, origin="lower", aspect="auto",
              extent=[-0.5, len(xv) - 0.5, -0.5, len(yv) - 0.5], interpolation="nearest")
    if boundaries:
        step(ax, M, (G,), color="white", lw=2.2, ls="--")
        step(ax, M, (G, Y), color="black", lw=1.6, ls=":")
    nx = max(1, len(xv) // 6); ny = max(1, len(yv) // 6)
    ax.set_xticks(range(0, len(xv), nx)); ax.set_xticklabels([f"{xv[i]:g}" for i in range(0, len(xv), nx)], fontsize=11)
    ax.set_yticks(range(0, len(yv), ny)); ax.set_yticklabels([f"{yv[i]:.2f}" for i in range(0, len(yv), ny)], fontsize=11)
    ax.set_xlabel(xl, fontsize=12); ax.set_ylabel(yl, fontsize=12)
    ax.set_title(title, fontsize=12)


def fig_maps():
    MA = load("fine", "coverage_fine.csv", TTCS, SPEEDS, "TTC_s", "v_kmh")
    MB = load("friction_speed", "friction_speed.csv", MUS, SPEEDS, "mu", "v_kmh")
    MC = load("friction_ttc", "friction_ttc.csv", MUS, TTCS_B, "mu", "TTC_s")
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.6), gridspec_kw=dict(width_ratios=[1.25, 1, 1]))
    fig.subplots_adjust(left=0.05, right=0.99, top=0.91, bottom=0.25, wspace=0.22)
    draw(axs[0], MA, SPEEDS, TTCS, "Ego speed (km/h)", "TTC (s)",
         r"(a) Speed $\times$ TTC, $\mu=1.0$ (1,165 cells)", boundaries=True)
    draw(axs[1], MB, SPEEDS, MUS, "Ego speed (km/h)", r"Road friction $\mu$",
         r"(b) Speed $\times\,\mu$, TTC $=1.0$ s (430 cells)")
    draw(axs[2], MC, TTCS_B, MUS, "TTC (s)", r"Road friction $\mu$",
         r"(c) TTC $\times\,\mu$, $v=50$ km/h (260 cells)")
    handles = [Patch(color=COL[c], label=LAB[c]) for c in (G, Y, O, R, X)]
    handles += [Line2D([], [], color="gray", lw=2.2, ls="--", label="Simulated braking boundary"),
                Line2D([], [], color="black", lw=1.6, ls=":", label="Simulated avoidance boundary (braking or RRT path)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=11, frameon=True, bbox_to_anchor=(0.5, 0.0))
    out = os.path.join(FIGS, "mitigability_rrt.png")
    fig.savefig(out, dpi=220, facecolor="white"); plt.close(fig)
    for name, M in (("a", MA), ("b", MB), ("c", MC)):
        print(f"map({name}) counts:", {LAB[c]: int((M == c).sum()) for c in (G, Y, O, R)})
    print("->", out)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def fig_param_sweep():
    rows = list(csv.DictReader(open(os.path.join(RES, "coverage_fine.csv"))))
    per = defaultdict(lambda: [0, 0, 0])
    for r in rows:
        vk, t = int(r["v_kmh"]), round(float(r["TTC_s"]), 2)
        st = SB[("fine", vk, t, 1.0)] == 0.0
        p = per[vk]; p[0] += 1; p[1] += st; p[2] += (st or r["has_path"] == "True")
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.2))
    xs = SPEEDS
    ax.plot(xs, [100 * per[v][1] / per[v][0] for v in xs], "--", color="#607d8b", lw=2, label="Emergency braking (simulated)")
    ax.plot(xs, [100 * per[v][2] / per[v][0] for v in xs], "-", color="#2e7d32", lw=2, label="EMRM, planner seed 42")
    seedA = os.path.join(RES, "seed_campaign_A.csv")
    covered = {int(r["v_kmh"]) for r in csv.DictReader(open(seedA))} if os.path.exists(seedA) else set()
    if covered >= set(SPEEDS):
        sp = defaultdict(lambda: [0, 0]); seeds = set()
        for r in csv.DictReader(open(seedA)):
            vk, t = int(r["v_kmh"]), round(float(r["TTC_s"]), 2)
            seeds.add(r["seed"])
            ok = SB[("fine", vk, t, 1.0)] == 0.0 or r["has_path"] == "True"
            sp[vk][0] += 1; sp[vk][1] += ok
        vs = [v for v in xs if sp[v][0]]
        ci = [wilson(sp[v][1], sp[v][0]) for v in vs]
        ax.plot(vs, [100 * c[0] for c in ci], "-", color="#1565c0", lw=1.6, label=f"EMRM, {len(seeds)} further seeds")
        ax.fill_between(vs, [100 * c[1] for c in ci], [100 * c[2] for c in ci], color="#1565c0", alpha=0.2,
                        label="95% Wilson interval")
    ax.set_xlabel("Ego speed (km/h)", fontsize=11); ax.set_ylabel("Collision avoidance rate (%)", fontsize=11)
    ax.set_ylim(-3, 103); ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="lower left")
    ax.set_title("(a) Collision avoidance rate versus ego speed", fontsize=11)

    seedB = os.path.join(RES, "seed_campaign_B.csv")
    vks = [30, 40, 50, 60, 72]
    if os.path.exists(seedB):
        cnt = defaultdict(lambda: [0, 0]); seeds = set()
        for r in csv.DictReader(open(seedB)):
            vk, t = int(r["v_kmh"]), round(float(r["TTC_s"]), 2)
            seeds.add(r["seed"])
            cnt[(vk, t)][0] += 1; cnt[(vk, t)][1] += r["has_path"] == "True"
        ts = sorted({t for (_, t) in cnt})
        P = np.array([[cnt[(vk, t)][1] / max(1, cnt[(vk, t)][0]) for t in ts] for vk in vks])
        im = bx.imshow(P, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1,
                       extent=[ts[0] - 0.01, ts[-1] + 0.01, -0.5, len(vks) - 0.5], interpolation="nearest")
        for i, vk in enumerate(vks):
            stop = [t for t in ts if SB.get(("refine", vk, t, 1.0), 1) == 0.0]
            if stop:
                bx.plot([min(stop) - 0.01] * 2, [i - 0.5, i + 0.5], color="white", lw=2.5, ls="--")
        cb = fig.colorbar(im, ax=bx, pad=0.02); cb.set_label("Fraction of seeds with a path", fontsize=10)
        bx.set_yticks(range(len(vks))); bx.set_yticklabels([str(v) for v in vks])
        bx.set_title(f"(b) RRT path-finding over {len(seeds)} seeds (TTC step 0.02 s)", fontsize=11)
        bx.legend(handles=[Line2D([], [], color="white", ls="--", lw=2.5, label="Simulated braking boundary")],
                  fontsize=9, loc="upper left", facecolor="#555555", labelcolor="white")
    bx.set_xlabel("TTC (s)", fontsize=11); bx.set_ylabel("Ego speed (km/h)", fontsize=11)
    out = os.path.join(FIGS, "param_sweep_rrt.png")
    fig.tight_layout(); fig.savefig(out, dpi=220, facecolor="white"); plt.close(fig)
    print("->", out)


if __name__ == "__main__":
    os.makedirs(FIGS, exist_ok=True)
    fig_maps()
    fig_param_sweep()
