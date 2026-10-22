#!/usr/bin/env python3
"""
Run RRT on T-junction map and plot all feasible paths (tree + best path).

Uses occupancy map: 0 = obstacle, 255 = free.
Dynamical RRT: RU-Racer EoM from https://github.com/ru-racer/ru-racer (Bike.m).
Pacejka tire model, slip-based forces, full 6-DOF bicycle dynamics.
"""

from __future__ import annotations

import math
import os
import random
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

# Vehicle params (from fullsize_common.cfg)
WHEELBASE = 2.80  # lf + lr
VEH_LENGTH = 2.80
VEH_WIDTH = 1.80
VEH_MASS = 1500.0  # kg
VEH_INERTIA = 2500.0  # kg·m² (yaw inertia Izz)
VEH_SCALE = 2.0  # Vehicle-to-road scale: match base map (adjust if vehicle looks wrong)
VEH_LENGTH_SCALE = 1.68  # 1.2 * 1.4: rectangle length scale
VEH_WIDTH_SCALE = 1.1  # Rectangle width scale
MAX_STEER_RAD = 0.5536  # ~30 deg (delta_max from config)
MAX_CURVATURE_STEER = math.tan(MAX_STEER_RAD) / WHEELBASE  # ~0.206 rad/m
MAX_LATERAL_ACCEL = 2 * 9.81  # ~7.85 m/s^2 (dry road, mu*g)
MAX_STEER_RATE_RAD_S = 0.5  # max steering angle rate, rad/s (limits rotation at high speed)
MAX_ACCEL = 5.0  # m/s² (longitudinal acceleration limit)
MAX_DECEL = -6.0  # m/s² (braking limit)
STEER_TIME_CONST = 0.05  # s (first-order steering dynamics τ, fast response)


def load_occupancy(map_path: str, ppm: float) -> tuple:
    """Load PGM occupancy. Returns (grid, xmin, xmax, ymin, ymax). 0=obstacle, 1=free."""
    import numpy as np
    with open(map_path, "rb") as f:
        assert f.readline().strip() == b"P5"
        w, h = map(int, f.readline().decode().split())
        f.readline()  # maxval
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape((h, w))
    # 0 = obstacle, 255 = free -> 0 = obstacle, 1 = free
    grid = (data > 0).astype(np.uint8)
    xmax, ymax = w / ppm, h / ppm
    return grid, 0.0, xmax, 0.0, ymax


def world_to_grid(x: float, y: float, xmin: float, ymin: float, ppm: float) -> tuple[int, int]:
    """World (x,y) to grid (col, row). Image: row 0 = top (y=0)."""
    col = int(round((x - xmin) * ppm))
    row = int(round((ymax_from_ppm(ppm) - y) * ppm)) if 'ymax_from_ppm' in dir() else int(round((ymin + (grid_h := 0) / ppm - y) * ppm))
    return col, row


def check_collision(
    x1: float, y1: float, x2: float, y2: float,
    grid, xmin: float, ymax: float, ppm: float, step: float = 0.5, margin: int = 0
) -> bool:
    """True if segment (x1,y1)-(x2,y2) collides with obstacle. World: y=0 bottom, y=ymax top."""
    import numpy as np
    h, w = grid.shape
    dist = math.hypot(x2 - x1, y2 - y1)
    n = max(2, int(dist / step))
    for i in range(n + 1):
        t = i / n
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        col = int(round(x * ppm))
        row = int(round((ymax - y) * ppm))  # world y=0 -> bottom of image
        col = max(0, min(w - 1, col))
        row = max(0, min(h - 1, row))  # clamp to image bounds
        r0 = max(0, row - margin)
        r1 = min(h, row + margin + 1)
        c0 = max(0, col - margin)
        c1 = min(w, col + margin + 1)
        if np.any(grid[r0:r1, c0:c1] == 0):
            return True
    return False


def _normalize_angle(a: float) -> float:
    """Wrap angle to [-pi, pi]."""
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def max_curvature_at_speed(v: float) -> float:
    """Speed-dependent max curvature: limited by lateral acceleration a_y = v^2*k."""
    k_steer = MAX_CURVATURE_STEER
    k_ay = MAX_LATERAL_ACCEL / max(0.1, v * v)
    return min(k_steer, k_ay)


def _brake_to_stop_extend(
    x: float, y: float, psi: float, vx: float, vy: float, dotpsi: float,
    dt: float, max_steps: int = 40, stop_v: float = 3.0, sfx: float = -0.35,
) -> tuple[tuple[float, float, float, float, float, float], list[tuple[float, float, float]], bool]:
    """
    Multi-step brake extension: brake straight until stop or max_steps.
    Returns (final_state, path_points, stopped) where stopped=True if v < stop_v.
    """
    from ru_racer_dynamics import ru_racer_propagate
    path_pts = [(x, y, psi)]
    xt, yt, psi_t, vxt, vyt, dotpsi_t = x, y, psi, vx, vy, dotpsi
    delta = 0.0  # straight braking
    for _ in range(max_steps):
        xt, yt, psi_t, vxt, vyt, dotpsi_t = ru_racer_propagate(
            xt, yt, psi_t, vxt, vyt, dotpsi_t, delta, sfx, dt
        )
        path_pts.append((xt, yt, psi_t))
        v = math.sqrt(vxt * vxt + vyt * vyt)
        if v < stop_v:
            return (xt, yt, psi_t, vxt, vyt, dotpsi_t), path_pts, True
    return (xt, yt, psi_t, vxt, vyt, dotpsi_t), path_pts, False


def ru_racer_extend(
    x: float, y: float, psi: float, vx: float, vy: float, dotpsi: float,
    target_x: float, target_y: float,
    dt: float,
    prev_delta: float = 0.0,
    sfx_override: float | None = None,
) -> tuple[tuple[float, float, float, float, float, float], list[tuple[float, float, float]]]:
    """
    RU-Racer EoM extend: steer toward target using Bike.m dynamics.
    State: (x, y, psi, vx, vy, dotpsi). Input: (delta, sfx).
    Returns ((x_new, y_new, psi_new, vx_new, vy_new, dotpsi_new), path_points).
    """
    from ru_racer_dynamics import ru_racer_propagate

    desired_psi = math.atan2(target_y - y, target_x - x)
    psi_err = _normalize_angle(desired_psi - psi)
    dist = math.hypot(target_x - x, target_y - y)
    v = math.sqrt(vx * vx + vy * vy)
    v_safe = max(0.5, v)

    # Steering: proportional to heading error, rate-limited (Bike.m Controller uses kp~2)
    kp = 3.0
    delta_des = kp * psi_err
    delta_des = max(-MAX_STEER_RAD, min(MAX_STEER_RAD, delta_des))
    max_ddelta = MAX_STEER_RATE_RAD_S * dt
    delta = max(prev_delta - max_ddelta, min(prev_delta + max_ddelta, delta_des))

    # Longitudinal slip: coast when far, mild brake when close (Bike.m sfx_max~0.1)
    # sfx_override: use for drift-to-collision (braking reduces impact energy)
    if sfx_override is not None:
        sfx = sfx_override
    else:
        sfx = 0.0 if dist > 3.0 else -0.05

    # Integrate with sub-steps for path points (finer for narrow passages)
    n = max(3, int(v * dt / 0.3))
    dt_step = dt / n
    path_pts = [(x, y, psi)]
    xt, yt, psi_t, vxt, vyt, dotpsi_t = x, y, psi, vx, vy, dotpsi
    for _ in range(n - 1):
        xt, yt, psi_t, vxt, vyt, dotpsi_t = ru_racer_propagate(
            xt, yt, psi_t, vxt, vyt, dotpsi_t, delta, sfx, dt_step
        )
        path_pts.append((xt, yt, psi_t))
    x_new, y_new, psi_new, vx_new, vy_new, dotpsi_new = ru_racer_propagate(
        xt, yt, psi_t, vxt, vyt, dotpsi_t, delta, sfx, dt_step
    )
    path_pts.append((x_new, y_new, psi_new))
    return (x_new, y_new, psi_new, vx_new, vy_new, dotpsi_new), path_pts


def _vehicle_rect_corners(x: float, y: float, psi: float) -> list[tuple[float, float]]:
    """Return 4 corners of vehicle rectangle (x,y center, psi heading)."""
    hl = VEH_LENGTH * VEH_SCALE * VEH_LENGTH_SCALE / 2
    hw = VEH_WIDTH * VEH_SCALE * VEH_WIDTH_SCALE / 2
    c, s = math.cos(psi), math.sin(psi)
    return [
        (x + hl * c - hw * s, y + hl * s + hw * c),
        (x + hl * c + hw * s, y + hl * s - hw * c),
        (x - hl * c + hw * s, y - hl * s - hw * c),
        (x - hl * c - hw * s, y - hl * s + hw * c),
    ]


def _point_in_rect(px: float, py: float, x: float, y: float, psi: float) -> bool:
    """True if (px,py) is inside vehicle rectangle centered at (x,y) with heading psi."""
    hl = VEH_LENGTH * VEH_SCALE * VEH_LENGTH_SCALE / 2
    hw = VEH_WIDTH * VEH_SCALE * VEH_WIDTH_SCALE / 2
    dx = px - x
    dy = py - y
    local_x = dx * math.cos(psi) + dy * math.sin(psi)
    local_y = -dx * math.sin(psi) + dy * math.cos(psi)
    return abs(local_x) <= hl and abs(local_y) <= hw


def check_rect_collision(
    x: float, y: float, psi: float,
    grid, ymax: float, ppm: float,
) -> bool:
    """True if vehicle rectangle at (x,y,psi) overlaps obstacle."""
    import numpy as np
    h, w = grid.shape
    corners = _vehicle_rect_corners(x, y, psi)
    cols = [int(round(cx * ppm)) for cx, _ in corners]
    rows = [int(round((ymax - cy) * ppm)) for _, cy in corners]
    c0, c1 = max(0, min(cols)), min(w - 1, max(cols))
    r0, r1 = max(0, min(rows)), min(h - 1, max(rows))
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if grid[r, c] == 0:
                cx_w = (c + 0.5) / ppm
                cy_w = ymax - (r + 0.5) / ppm
                if _point_in_rect(cx_w, cy_w, x, y, psi):
                    return True
    return False


def check_path_collision(
    path_pts: list[tuple[float, float, float]],
    grid, ymax: float, ppm: float,
) -> bool:
    """True if vehicle footprint at any path point collides with obstacle."""
    for x, y, psi in path_pts:
        if check_rect_collision(x, y, psi, grid, ymax, ppm):
            return True
    return False


def _compute_energy(node: tuple, m: float = VEH_MASS, I: float = VEH_INERTIA) -> float:
    """Kinetic energy: 0.5*m*v² + 0.5*I*ω². Node: (x,y,psi,vx,vy,omega)."""
    x, y, psi, vx, vy, omega = node
    v = math.sqrt(vx * vx + vy * vy)
    return 0.5 * m * v * v + 0.5 * I * omega * omega


def _compute_momentum(node: tuple, m: float = VEH_MASS) -> float:
    """Momentum magnitude m*||v|| for impact metric. Objects assumed stopped; ego momentum = collision impact.
    Node: (x,y,psi,vx,vy,omega). Returns m * sqrt(vx² + vy²)."""
    x, y, psi, vx, vy, omega = node
    v = math.sqrt(vx * vx + vy * vy)
    return m * v


# Collision cost prioritization: prefer accidents to road objects over borders
# Tune BORDER_COST_FACTOR: higher = stronger preference for object collisions
BORDER_COLLISION_MARGIN = 8.0  # m: collision within this of map edge = border
BORDER_COST_FACTOR = 2.5       # multiplier for border collisions (higher = worse)
OBJECT_COST_FACTOR = 1.0       # multiplier for object collisions (preferred)


def _is_border_collision(
    x: float, y: float,
    xmin: float, xmax: float, ymin: float, ymax: float,
    margin: float = BORDER_COLLISION_MARGIN,
) -> bool:
    """True if collision point is near map boundary (road border)."""
    return (
        x - xmin < margin or xmax - x < margin
        or y - ymin < margin or ymax - y < margin
    )


def _collision_cost(
    node: tuple,
    xmin: float, xmax: float, ymin: float, ymax: float,
    border_factor: float = BORDER_COST_FACTOR,
    object_factor: float = OBJECT_COST_FACTOR,
    border_margin: float = BORDER_COLLISION_MARGIN,
) -> float:
    """Collision cost: momentum * location factor. Higher = worse.
    Prefer accidents to road objects (lower cost) over borders (higher cost)."""
    mom = _compute_momentum(node)
    x, y = node[0], node[1]
    if _is_border_collision(x, y, xmin, xmax, ymin, ymax, border_margin):
        return mom * border_factor
    return mom * object_factor


def sparse_rrt(
    nodes: list,
    edges: list,
    terminal_collision: set,
    parent: dict,
    path: list,
    goal: tuple[float, float],
    sparse_resolution: float = 2,
    v_resolution: float = 1.0,
    psi_resolution: float = 1,
    vy_resolution: float = 1,
    omega_resolution: float = 1,
    adaptive_critical_dist: float = 12.0,
    adaptive_factor: float = 0.5,
) -> tuple[list, list, set, dict, list]:
    """
    Sparsify RRT: keep one node per (x, y, v, psi, vy, omega) bin for complete search.
    Adaptive resolution: finer (smaller radius) near goal and for collision nodes.
    adaptive_critical_dist: distance to goal (m) for critical region. adaptive_factor: 0.5 = 2x finer when critical.
    Returns (nodes, edges, terminal_collision, parent, path).
    """
    from collections import defaultdict

    def _is_critical(i: int, x: float, y: float) -> bool:
        if i in terminal_collision:
            return True
        return math.hypot(x - goal[0], y - goal[1]) < adaptive_critical_dist

    def bin_key(x: float, y: float, psi: float, vx: float, vy: float, omega: float, is_crit: bool) -> tuple:
        r = sparse_resolution * (adaptive_factor if is_crit else 1.0)
        vr = v_resolution * (adaptive_factor if is_crit else 1.0)
        pr = psi_resolution * (adaptive_factor if is_crit else 1.0)
        vyr = vy_resolution * (adaptive_factor if is_crit else 1.0)
        omr = omega_resolution * (adaptive_factor if is_crit else 1.0)
        ix = int(math.floor(x / r))
        iy = int(math.floor(y / r))
        v = math.sqrt(vx * vx + vy * vy)
        iv = int(math.floor(v / vr))
        psi_n = _normalize_angle(psi)
        ipsi = int(math.floor(psi_n / pr))
        ivy = int(math.floor(vy / vyr))
        iomega = int(math.floor(omega / omr))
        return (ix, iy, iv, ipsi, ivy, iomega, is_crit)

    # Compute depth from start for each node
    depth = {}
    for i in range(len(nodes)):
        d = 0
        j = i
        while parent.get(j, -1) >= 0:
            d += 1
            j = parent[j]
        depth[i] = d

    # Group nodes by (x, y, v, psi, vy, omega) bin with adaptive resolution
    bins = defaultdict(list)
    for i, n in enumerate(nodes):
        x, y, psi, vx, vy, omega = n[0], n[1], n[2], n[3], n[4], n[5]
        is_crit = _is_critical(i, x, y)
        bk = bin_key(x, y, psi, vx, vy, omega, is_crit)
        is_coll = i in terminal_collision
        e = _compute_energy(n) if is_coll else None
        bins[bk].append((i, is_coll, e, depth[i]))

    # Select kept indices per bin
    kept = set()
    kept.add(0)  # always keep start
    for bk, candidates in bins.items():
        coll = [(idx, e) for idx, is_coll, e, _ in candidates if is_coll and e is not None]
        non_coll = [(idx, d) for idx, is_coll, e, d in candidates if not is_coll]
        # Collision: keep min energy
        if coll:
            best_coll = min(coll, key=lambda x: x[1])
            kept.add(best_coll[0])
        # Non-collision: keep one with minimum depth (preserve main path)
        if non_coll:
            best_non = min(non_coll, key=lambda x: x[1])
            kept.add(best_non[0])

    # Build new structures
    old_to_new = {}
    new_nodes = []
    for i in sorted(kept):
        old_to_new[i] = len(new_nodes)
        new_nodes.append(nodes[i])

    def find_kept_ancestor(idx: int) -> int:
        """Follow parent chain until we find a kept node."""
        while idx >= 0 and idx not in kept:
            idx = parent.get(idx, -1)
        return idx

    new_edges = []
    new_parent = {}
    for oi, oj in parent.items():
        if oj < 0:
            if oi in kept:
                new_parent[old_to_new[oi]] = -1
            continue
        if oi not in kept:
            continue
        anc = find_kept_ancestor(oj)
        if anc >= 0:
            new_edges.append(((nodes[anc][0], nodes[anc][1]), (nodes[oi][0], nodes[oi][1])))
            new_parent[old_to_new[oi]] = old_to_new[anc]

    new_terminal_collision = {old_to_new[i] for i in terminal_collision if i in kept}

    # Rebuild path: find node closest to goal among kept non-collision, trace back to start
    new_path = []
    goal_dist = 3.5
    candidates_goal = []
    for i in kept:
        if i in terminal_collision:
            continue
        n = nodes[i]
        d = math.hypot(n[0] - goal[0], n[1] - goal[1])
        if d < goal_dist:
            candidates_goal.append((i, d))
    if candidates_goal:
        best_goal_idx = min(candidates_goal, key=lambda x: x[1])[0]
        chain = []
        idx = best_goal_idx
        while idx >= 0:
            if idx in kept:
                chain.append(idx)
            idx = parent.get(idx, -1)
        if chain:
            path_nodes = [nodes[i] for i in reversed(chain)]
            new_path = [(p[0], p[1], p[2]) for p in path_nodes]
    if not new_path and path:
        new_path = path

    return new_nodes, new_edges, new_terminal_collision, new_parent, new_path


def _make_tree(nodes_init: list | None = None) -> dict:
    """
    Parallel tree structure for green and orange trees.
    Returns dict with: nodes (list), edges (list of ((x1,y1),(x2,y2))), parent (dict).
    Green: parent[i] = index in same tree. Orange: parent[i] = index in green tree.
    """
    nodes = list(nodes_init) if nodes_init else []
    return {"nodes": nodes, "edges": [], "parent": {0: -1} if nodes else {}}


def rrt_double_tree(
    start: tuple[float, float, float, float],  # (x, y, psi, v)
    goal: tuple[float, float],
    grid, xmin: float, xmax: float, ymin: float, ymax: float,
    ppm: float,
    max_nodes: int = 5000,
    dt: float = 0.1,
    goal_sample_rate: float = 0.55,  # exploitation: more goal bias for feasible tree
    border_cost_factor: float = BORDER_COST_FACTOR,
    object_cost_factor: float = OBJECT_COST_FACTOR,
    border_margin: float = BORDER_COLLISION_MARGIN,
    sample_boxes: dict | None = None,
) -> tuple[dict, dict, list, int]:
    """
    Green-orange double-tree RRT. Both trees use parallel structure: {nodes, edges, parent}.
    - Green: exploitation toward goal, collision-free.
    - Orange: no goal; parent=less energy; Obj1 stop, Obj2 min momentum.
    sample_boxes overrides the map-specific sampling regions ("junction",
    "left", "right"), each ((x0, x1), (y0, y1)); defaults are the T-junction map.
    Returns (tree_green, tree_orange, best_path, n_drained).
    """
    v0 = start[3]
    start_ru = (start[0], start[1], start[2], v0, 0.0, 0.0)

    # Parallel structure for both trees
    tree_green = _make_tree([start_ru])
    tree_orange = _make_tree([])  # orange has no root; parents point to green indices
    tree_orange["parent"] = {}
    best_path = []
    n_stop = 0  # orange nodes that stopped (Obj1)
    STOP_V_THRESHOLD = 3.0

    junction_x, junction_y = (40.0, 48.0), (15.0, 45.0)
    left_x, left_y = (5.0, 38.0), (10.0, 50.0)
    right_x, right_y = (42.0, 48.0), (10.0, 50.0)
    if sample_boxes:
        junction_x, junction_y = sample_boxes.get("junction", (junction_x, junction_y))
        left_x, left_y = sample_boxes.get("left", (left_x, left_y))
        right_x, right_y = sample_boxes.get("right", (right_x, right_y))
    junction_rate = 0.25
    left_rate = 0.20
    right_rate = 0.20
    stop_rate = 0.25  # orange: emergency brake-to-stop (fail-safe)

    ng, eg, pg = tree_green["nodes"], tree_green["edges"], tree_green["parent"]
    no, eo, po = tree_orange["nodes"], tree_orange["edges"], tree_orange["parent"]

    for _ in range(max_nodes):
        # --- GREEN TREE (exploitation: goal-biased) ---
        r = random.random()
        if r < goal_sample_rate:
            rand_xy = goal
        elif r < goal_sample_rate + junction_rate:
            rand_xy = (
                junction_x[0] + random.random() * (junction_x[1] - junction_x[0]),
                junction_y[0] + random.random() * (junction_y[1] - junction_y[0]),
            )
        else:
            rand_xy = (
                xmin + random.random() * (xmax - xmin),
                ymin + random.random() * (ymax - ymin),
            )

        best_i = 0
        best_d = float("inf")
        for i, n in enumerate(ng):
            d = math.hypot(rand_xy[0] - n[0], rand_xy[1] - n[1])
            if d < best_d:
                best_d = d
                best_i = i

        near = ng[best_i]
        x_n, y_n, psi_n, vx_n, vy_n, dotpsi_n = near[0], near[1], near[2], near[3], near[4], near[5]
        v_n = math.sqrt(vx_n * vx_n + vy_n * vy_n)
        prev_delta = math.atan(dotpsi_n * WHEELBASE / max(0.5, v_n)) if v_n > 0.5 else 0.0

        (x_new, y_new, psi_new, vx_new, vy_new, dotpsi_new), path_pts = ru_racer_extend(
            x_n, y_n, psi_n, vx_n, vy_n, dotpsi_n, rand_xy[0], rand_xy[1], dt, prev_delta, sfx_override=None
        )
        if vx_new >= 0.3 and xmin <= x_new <= xmax and ymin <= y_new <= ymax:
            if not any(pt[0] < xmin or pt[0] > xmax or pt[1] < ymin or pt[1] > ymax for pt in path_pts):
                in_collision = check_path_collision(path_pts, grid, ymax, ppm)
                if not in_collision:
                    new_state = (x_new, y_new, psi_new, vx_new, vy_new, dotpsi_new)
                    ng.append(new_state)
                    eg.append(((x_n, y_n), (x_new, y_new)))
                    pg[len(ng) - 1] = best_i
                    if math.hypot(x_new - goal[0], y_new - goal[1]) < 3.5:
                        path = []
                        idx = len(ng) - 1
                        while idx >= 0:
                            n = ng[idx]
                            path.append((n[0], n[1], n[2]))
                            idx = pg.get(idx, -1)
                        path.reverse()
                        if not best_path or len(path) < len(best_path):
                            best_path = path

        # --- ORANGE TREE (fail-safe: Obj1 stop, Obj2 min-momentum collision) ---
        r2 = random.random()
        # Emergency brake-to-stop (fail-safe): multi-step straight braking
        if r2 < stop_rate and len(ng) > 0:
            i_stop = random.randint(0, len(ng) - 1)
            n_stop_src = ng[i_stop]
            x_s, y_s, psi_s, vx_s, vy_s, dp_s = n_stop_src[0], n_stop_src[1], n_stop_src[2], n_stop_src[3], n_stop_src[4], n_stop_src[5]
            v_s = math.sqrt(vx_s * vx_s + vy_s * vy_s)
            if v_s >= STOP_V_THRESHOLD:
                state_stop, path_stop, stopped = _brake_to_stop_extend(
                    x_s, y_s, psi_s, vx_s, vy_s, dp_s, dt=0.15, max_steps=35, stop_v=STOP_V_THRESHOLD, sfx=-0.35
                )
                if stopped:
                    in_coll_stop = check_path_collision(path_stop, grid, ymax, ppm)
                    if not in_coll_stop and xmin <= state_stop[0] <= xmax and ymin <= state_stop[1] <= ymax:
                        if not any(pt[0] < xmin or pt[0] > xmax or pt[1] < ymin or pt[1] > ymax for pt in path_stop):
                            no.append(state_stop)
                            eo.append(((x_s, y_s), (state_stop[0], state_stop[1])))
                            po[len(no) - 1] = i_stop
                            n_stop += 1
        else:
            # Collision sampling (drift-to-collision)
            if r2 < stop_rate + left_rate:
                rand_xy_acc = (
                    left_x[0] + random.random() * (left_x[1] - left_x[0]),
                    left_y[0] + random.random() * (left_y[1] - left_y[0]),
                )
            elif r2 < stop_rate + left_rate + right_rate:
                rand_xy_acc = (
                    right_x[0] + random.random() * (right_x[1] - right_x[0]),
                    right_y[0] + random.random() * (right_y[1] - right_y[0]),
                )
            else:
                rand_xy_acc = (
                    xmin + random.random() * (xmax - xmin),
                    ymin + random.random() * (ymax - ymin),
                )

            # Parent selection: prefer lower energy among k-nearest
            candidates_acc = []
            for i, n in enumerate(ng):
                d = math.hypot(rand_xy_acc[0] - n[0], rand_xy_acc[1] - n[1])
                e = _compute_energy(n)
                candidates_acc.append((i, d, e))
            candidates_acc.sort(key=lambda x: x[1])
            k_near = min(10, len(candidates_acc))
            near_cands = candidates_acc[:k_near]
            near_cands.sort(key=lambda x: x[2])
            k_acc = min(3, len(near_cands))
            best_i_acc = near_cands[random.randint(0, k_acc - 1)][0] if k_acc > 0 else 0

            near_acc = ng[best_i_acc]
            x_na, y_na, psi_na, vx_na, vy_na, dotpsi_na = near_acc[0], near_acc[1], near_acc[2], near_acc[3], near_acc[4], near_acc[5]
            v_na = math.sqrt(vx_na * vx_na + vy_na * vy_na)
            prev_delta_acc = math.atan(dotpsi_na * WHEELBASE / max(0.5, v_na)) if v_na > 0.5 else 0.0

            (x_acc, y_acc, psi_acc, vx_acc, vy_acc, dp_acc), path_acc = ru_racer_extend(
                x_na, y_na, psi_na, vx_na, vy_na, dotpsi_na, rand_xy_acc[0], rand_xy_acc[1], dt, prev_delta_acc,
                sfx_override=-0.18,
            )
            v_acc = math.sqrt(vx_acc * vx_acc + vy_acc * vy_acc)
            if xmin <= x_acc <= xmax and ymin <= y_acc <= ymax:
                if not any(pt[0] < xmin or pt[0] > xmax or pt[1] < ymin or pt[1] > ymax for pt in path_acc):
                    in_coll = check_path_collision(path_acc, grid, ymax, ppm)
                    state_acc = (x_acc, y_acc, psi_acc, vx_acc, vy_acc, dp_acc)
                    if not in_coll and v_acc < STOP_V_THRESHOLD:
                        no.append(state_acc)
                        eo.append(((x_na, y_na), (x_acc, y_acc)))
                        po[len(no) - 1] = best_i_acc  # parent = green index
                        n_stop += 1
                    elif in_coll and vx_acc >= 0.3:
                        no.append(state_acc)
                        eo.append(((x_na, y_na), (x_acc, y_acc)))
                        po[len(no) - 1] = best_i_acc

    # Sparsify green tree
    n_before = len(ng)
    ng, eg, _, pg, best_path = sparse_rrt(
        ng, eg, set(), pg, best_path, goal,
        sparse_resolution=0.3, v_resolution=1.0, psi_resolution=0.35,
        vy_resolution=0.6, omega_resolution=0.2,
        adaptive_critical_dist=15.0, adaptive_factor=0.4,
    )
    tree_green["nodes"], tree_green["edges"], tree_green["parent"] = ng, eg, pg
    n_drained_green = n_before - len(ng)

    # Sparsify orange: one stop per bin, min-cost per bin for collisions
    from collections import defaultdict
    n_acc_before = len(no)
    stop_bins = defaultdict(list)
    for i in range(n_stop):
        x, y = no[i][0], no[i][1]
        r = 1.0
        ix, iy = int(math.floor(x / r)), int(math.floor(y / r))
        stop_bins[(ix, iy)].append(i)
    kept_stop = sorted({cands[0] for cands in stop_bins.values()})
    n_stop_kept = len(kept_stop)
    acc_bins = defaultdict(list)
    for i, n in enumerate(no):
        if i >= n_stop:
            x, y = n[0], n[1]
            r = 0.5
            ix, iy = int(math.floor(x / r)), int(math.floor(y / r))
            cost = _collision_cost(n, xmin, xmax, ymin, ymax,
                                   border_factor=border_cost_factor,
                                   object_factor=object_cost_factor,
                                   border_margin=border_margin)
            acc_bins[(ix, iy)].append((i, cost))
    kept_coll = sorted({min(cands, key=lambda x: x[1])[0] for cands in acc_bins.values()})
    no_new = [no[i] for i in kept_stop] + [no[i] for i in kept_coll]
    eo_new = [eo[i] for i in kept_stop] + [eo[i] for i in kept_coll]
    po_new = {new_i: po[kept_stop[new_i]] for new_i in range(n_stop_kept)}
    for new_i, old_i in enumerate(kept_coll):
        po_new[n_stop_kept + new_i] = po[old_i]
    tree_orange["nodes"], tree_orange["edges"], tree_orange["parent"] = no_new, eo_new, po_new
    n_drained = n_drained_green + n_acc_before - len(no_new)
    return tree_green, tree_orange, best_path, n_drained, n_stop_kept


def rrt_dynamical(
    start: tuple[float, float, float, float],  # (x, y, psi, v)
    goal: tuple[float, float],
    grid, xmin: float, xmax: float, ymin: float, ymax: float,
    ppm: float,
    max_nodes: int = 5000,
    dt: float = 0.1,
    goal_sample_rate: float = 0.1,
    drift_collision_rate: float = 0.12,
    drift_collision_retry: bool = True,
) -> tuple[list, list[tuple], list, set]:
    """
    Dynamical RRT with RU-Racer EoM (Bike.m: Pacejka tires, slip forces).
    drift_collision_rate: fraction of extends that use braking (sfx=-0.1) to explore
        drift-to-collision trajectories with lower impact energy.
    drift_collision_retry: when a collision occurs, retry with heavier braking (sfx=-0.12);
        if that also collides with lower energy, add it as an extra node.
    Returns (nodes, edges, path, terminal_collision).
    nodes: list of (x, y, psi, vx, vy, omega)
    edges: list of ((x1,y1),(x2,y2)) for plotting
    path: list of (x, y) from start to goal
    terminal_collision: set of node indices that ended in collision (don't grow from these)
    """
    # Start: (x,y,psi,v) -> RU-Racer state (x,y,psi,vx,vy,0) with vx=v, vy=0
    v0 = start[3]
    start_ru = (start[0], start[1], start[2], v0, 0.0, 0.0)
    nodes = [start_ru]
    edges = []
    parent = {0: -1}
    best_path = []
    terminal_collision = set()

    # Junction region for narrow-passage bias (T-junction middle)
    junction_x = (40.0, 48.0)
    junction_y = (15.0, 45.0)
    junction_sample_rate = 0.28
    # Left region: drift-to-accident exploration
    left_x = (5.0, 38.0)
    left_y = (10.0, 50.0)
    left_sample_rate = 0.08
    left_drift_prob = 0.85
    # Right region: drift-to-accident exploration
    right_x = (42.0, 48.0)
    right_y = (10.0, 50.0)
    right_sample_rate = 0.08
    right_drift_prob = 0.85

    for _ in range(max_nodes):
        r = random.random()
        if r < goal_sample_rate:
            rand_xy = goal
            use_drift = False
        elif r < goal_sample_rate + junction_sample_rate:
            rand_xy = (
                junction_x[0] + random.random() * (junction_x[1] - junction_x[0]),
                junction_y[0] + random.random() * (junction_y[1] - junction_y[0]),
            )
            use_drift = random.random() < 0.15  # 15% drift for turning exploration
        elif r < goal_sample_rate + junction_sample_rate + left_sample_rate:
            rand_xy = (
                left_x[0] + random.random() * (left_x[1] - left_x[0]),
                left_y[0] + random.random() * (left_y[1] - left_y[0]),
            )
            use_drift = random.random() < left_drift_prob
        elif r < goal_sample_rate + junction_sample_rate + left_sample_rate + right_sample_rate:
            rand_xy = (
                right_x[0] + random.random() * (right_x[1] - right_x[0]),
                right_y[0] + random.random() * (right_y[1] - right_y[0]),
            )
            use_drift = random.random() < right_drift_prob
        else:
            rand_xy = (
                xmin + random.random() * (xmax - xmin),
                ymin + random.random() * (ymax - ymin),
            )
            use_drift = False

        best_i = 0
        best_d = float("inf")
        for i, n in enumerate(nodes):
            if i in terminal_collision:
                continue
            d = math.hypot(rand_xy[0] - n[0], rand_xy[1] - n[1])
            if d < best_d:
                best_d = d
                best_i = i

        near = nodes[best_i]
        x_n, y_n, psi_n, vx_n, vy_n, dotpsi_n = near[0], near[1], near[2], near[3], near[4], near[5]
        # Estimate prev_delta from steady-state: omega = vx*tan(delta)/L
        v_n = math.sqrt(vx_n * vx_n + vy_n * vy_n)
        prev_delta = math.atan(dotpsi_n * WHEELBASE / max(0.5, v_n)) if v_n > 0.5 else 0.0
        # Drift-to-collision: use braking for left/right-region samples or drift_collision_rate
        sfx_override = -0.1 if (use_drift or random.random() < drift_collision_rate) else None
        (x_new, y_new, psi_new, vx_new, vy_new, dotpsi_new), path_pts = ru_racer_extend(
            x_n, y_n, psi_n, vx_n, vy_n, dotpsi_n, rand_xy[0], rand_xy[1], dt, prev_delta, sfx_override
        )
        # Reject if vx too low (RU-Racer EoM requires vx>0 for slip calculations)
        if vx_new < 0.3:
            continue
        # Reject nodes outside workspace — tree does not continue beyond bounds
        if x_new < xmin or x_new > xmax or y_new < ymin or y_new > ymax:
            continue
        # Reject if any path point leaves workspace (tree stops at boundary)
        if any(
            pt[0] < xmin or pt[0] > xmax or pt[1] < ymin or pt[1] > ymax
            for pt in path_pts
        ):
            continue
        x_new = max(xmin, min(xmax, x_new))
        y_new = max(ymin, min(ymax, y_new))

        in_collision = check_path_collision(path_pts, grid, ymax, ppm)
        new_state = (x_new, y_new, psi_new, vx_new, vy_new, dotpsi_new)
        nodes.append(new_state)
        edges.append(((x_n, y_n), (x_new, y_new)))
        parent[len(nodes) - 1] = best_i

        if in_collision:
            terminal_collision.add(len(nodes) - 1)
            # Drift retry: extend from same parent with heavier braking for lower-momentum collision
            if drift_collision_retry and v_n > 5.0:
                (x_br, y_br, psi_br, vx_br, vy_br, dp_br), path_br = ru_racer_extend(
                    x_n, y_n, psi_n, vx_n, vy_n, dotpsi_n, rand_xy[0], rand_xy[1], dt, prev_delta,
                    sfx_override=-0.12,
                )
                if vx_br >= 0.3 and xmin <= x_br <= xmax and ymin <= y_br <= ymax:
                    in_br = check_path_collision(path_br, grid, ymax, ppm)
                    if in_br:
                        state_br = (x_br, y_br, psi_br, vx_br, vy_br, dp_br)
                        mom_orig = _compute_momentum(new_state)
                        mom_br = _compute_momentum(state_br)
                        if mom_br < mom_orig:
                            nodes.append(state_br)
                            edges.append(((x_n, y_n), (x_br, y_br)))
                            parent[len(nodes) - 1] = best_i
                            terminal_collision.add(len(nodes) - 1)
            continue

        if math.hypot(x_new - goal[0], y_new - goal[1]) < 2.0:
            path = []
            idx = len(nodes) - 1
            while idx >= 0:
                n = nodes[idx]
                path.append((n[0], n[1], n[2]))
                idx = parent[idx]
            path.reverse()
            if not best_path or len(path) < len(best_path):
                best_path = path
        # Continue exploring to build denser tree of feasible trajectories

    # Sparsify: keep one node per spatial bin; for collision nodes keep min-momentum (lowest impact)
    n_before_sparse = len(nodes)
    nodes, edges, terminal_collision, parent, best_path = sparse_rrt(
        nodes, edges, terminal_collision, parent, best_path, goal,
        sparse_resolution=0.2,   # finer spatial bins (was 0.5) -> fewer nodes drained
        v_resolution=0.2,
        psi_resolution=0.1,
        vy_resolution=0.1,
        omega_resolution=0.1,
        adaptive_critical_dist=15.0,
        adaptive_factor=0.1,      # finer bins in critical regions
    )
    n_drained = n_before_sparse - len(nodes)
    return nodes, edges, best_path, terminal_collision, n_drained


def main():
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scenario_dir = THIS_DIR
    map_pgm = os.path.join(scenario_dir, "maps", "tjunction_map.pgm")
    map_png = os.path.join(scenario_dir, "maps", "tjunction_map.png")
    ppm = 10.0

    # Regenerate PGM from PNG: black=obstacle (0), gray/white=free (255)
    if os.path.exists(map_png):
        try:
            from PIL import Image
            img = Image.open(map_png).convert("L")
            arr = np.array(img)
            w, h = img.size
            obstacle_threshold = 50  # pixel < 50 -> obstacle
            occupancy = np.where(arr < obstacle_threshold, 0, 255).astype(np.uint8)
            with open(map_pgm, "wb") as f:
                f.write(f"P5\n{w} {h}\n255\n".encode())
                f.write(occupancy.tobytes())
        except ImportError:
            pass  # fall back to existing PGM if PIL not available

    if not os.path.exists(map_pgm):
        print("ERROR: Map not found. Run from scenario-validation dir.")
        return 1

    grid, xmin, xmax, ymin, ymax = load_occupancy(map_pgm, ppm)
    # Start above junction, goal at bottom (slightly lower and right of original)
    # State: (x, y, psi, v) — psi=+1.57 rad = +90° facing +y toward goal
    start = (41.8, 5.2, 1.57, 12.0)
    goal = (44.0, 48.0)

    if check_rect_collision(start[0], start[1], start[2], grid, ymax, ppm):
        print("WARNING: Start position is in collision with obstacles (vehicle rectangle overlaps).")
        print(f"  Vehicle: {VEH_LENGTH * VEH_SCALE * VEH_LENGTH_SCALE:.2f} x {VEH_WIDTH * VEH_SCALE * VEH_WIDTH_SCALE:.2f} m")

    random.seed(123)
    tree_green, tree_orange, path, n_drained, n_stop = rrt_double_tree(
        start, goal, grid, xmin, xmax, ymin, ymax, ppm,
        max_nodes=10000, dt=0.03, goal_sample_rate=0.55,
    )
    nodes_feasible = tree_green["nodes"]
    edges_feasible = tree_green["edges"]
    nodes_accident = tree_orange["nodes"]
    edges_accident = tree_orange["edges"]

    # Smooth path: B-spline interpolation
    path_smooth = []
    path_smooth_psi = []
    if path and len(path) > 2:
        try:
            from scipy.interpolate import splprep, splev
            path_arr = np.array([(p[0], p[1]) for p in path])
            tck, u = splprep([path_arr[:, 0], path_arr[:, 1]], s=2.0, k=min(3, len(path) - 1))
            u_smooth = np.linspace(0, 1, max(50, len(path) * 2))
            x_smooth, y_smooth = splev(u_smooth, tck)
            path_smooth = list(zip(x_smooth, y_smooth))
            # Derive psi from path direction
            for i in range(len(path_smooth)):
                if i + 1 < len(path_smooth):
                    psi_i = math.atan2(path_smooth[i + 1][1] - path_smooth[i][1],
                                      path_smooth[i + 1][0] - path_smooth[i][0])
                else:
                    psi_i = path_smooth_psi[-1] if path_smooth_psi else start[2]
                path_smooth_psi.append(psi_i)
        except ImportError:
            path_smooth = [(p[0], p[1]) for p in path]
            path_smooth_psi = [p[2] for p in path]
        except Exception:
            path_smooth = [(p[0], p[1]) for p in path]
            path_smooth_psi = [p[2] for p in path]
    else:
        path_smooth = [(p[0], p[1]) for p in path] if path else []
        path_smooth_psi = [p[2] for p in path] if path else []

    # Plot
    from matplotlib.patches import Polygon
    from matplotlib.collections import LineCollection
    import matplotlib.cm as mcm

    def _node_speed(n):
        return math.sqrt(n[3] * n[3] + n[4] * n[4])

    # Speed range for colormap (all nodes)
    all_speeds = [_node_speed(n) for n in nodes_feasible] + [_node_speed(n) for n in nodes_accident]
    v_min, v_max = min(all_speeds), max(all_speeds)
    if v_max <= v_min:
        v_max = v_min + 1e-6
    cmap = plt.cm.viridis
    norm_speed = lambda v: (v - v_min) / (v_max - v_min)

    # Position -> speed (for edge coloring; use child node speed)
    def _pos_key(x, y):
        return (round(x, 3), round(y, 3))

    pos_to_speed_g = {_pos_key(n[0], n[1]): _node_speed(n) for n in nodes_feasible}
    pos_to_speed_o = {_pos_key(n[0], n[1]): _node_speed(n) for n in nodes_accident}

    def _speed_at_pos(p, green=True):
        d = pos_to_speed_g if green else pos_to_speed_o
        k = _pos_key(p[0], p[1])
        if k in d:
            return d[k]
        # Fallback: nearest
        best, best_d = None, float("inf")
        for kk, vv in d.items():
            dd = (p[0] - kk[0]) ** 2 + (p[1] - kk[1]) ** 2
            if dd < best_d:
                best_d, best = dd, vv
        return best if best is not None else v_min

    fig, ax = plt.subplots(figsize=(10, 10))
    if os.path.exists(map_png):
        img = np.array(plt.imread(map_png))
        ax.imshow(img, extent=[xmin, xmax, ymin, ymax], origin="upper", alpha=0.9)
    else:
        ax.imshow(grid, extent=[xmin, xmax, ymin, ymax], origin="upper", cmap="gray", alpha=0.8)

    # Positions where orange branches start (green nodes that are parents of orange)
    orange_branch_starts = list({(p1[0], p1[1]) for p1, p2 in edges_accident})
    tol = 0.05

    def _pos_is_orange_branch(p: tuple) -> bool:
        for o in orange_branch_starts:
            if abs(p[0] - o[0]) < tol and abs(p[1] - o[1]) < tol:
                return True
        return False

    # Feasible tree: color by speed (viridis: dark=slow, bright=fast)
    segs_g, colors_g = [], []
    for (p1, p2) in edges_feasible:
        if _pos_is_orange_branch(p2):
            continue
        v = _speed_at_pos(p2, green=True)
        segs_g.append([[p1[0], p1[1]], [p2[0], p2[1]]])
        colors_g.append(cmap(norm_speed(v)))
    if segs_g:
        lc_g = LineCollection(segs_g, colors=colors_g, linewidths=0.6, alpha=0.85)
        ax.add_collection(lc_g)
    # Feasible nodes (dots, color by speed)
    if nodes_feasible:
        gx = [n[0] for n in nodes_feasible]
        gy = [n[1] for n in nodes_feasible]
        g_colors = [cmap(norm_speed(_node_speed(n))) for n in nodes_feasible]
        ax.scatter(gx, gy, c=g_colors, s=10, alpha=0.9, edgecolors="none")

    # Orange tree: color by speed (same viridis scale)
    segs_o, colors_o = [], []
    for (p1, p2) in edges_accident:
        v = _speed_at_pos(p2, green=False)
        segs_o.append([[p1[0], p1[1]], [p2[0], p2[1]]])
        colors_o.append(cmap(norm_speed(v)))
    if segs_o:
        lc_o = LineCollection(segs_o, colors=colors_o, linewidths=0.6, alpha=0.85)
        ax.add_collection(lc_o)
    if nodes_accident:
        ox = [n[0] for n in nodes_accident]
        oy = [n[1] for n in nodes_accident]
        o_colors = [cmap(norm_speed(_node_speed(n))) for n in nodes_accident]
        ax.scatter(ox, oy, c=o_colors, s=14, alpha=0.9, edgecolors="none")
    if len(nodes_accident) > n_stop:
        coll_nodes = nodes_accident[n_stop:]
        min_cost = min(_collision_cost(n, xmin, xmax, ymin, ymax) for n in coll_nodes)
        for n in coll_nodes:
            if _collision_cost(n, xmin, xmax, ymin, ymax) == min_cost:
                x, y, psi, vx, vy, omega = n[0], n[1], n[2], n[3], n[4], n[5]
                corners = _vehicle_rect_corners(x, y, psi)
                poly = Polygon(corners, facecolor="orange", edgecolor="darkorange", alpha=0.35, linewidth=1.5)
                ax.add_patch(poly)
                ax.plot(n[0], n[1], "o", color="orange", markersize=5, label="Min-cost collision (prefer object)")
                # Velocity arrow (car velocity direction at impact)
                v = math.sqrt(vx * vx + vy * vy)
                if v >= 0.1:
                    VEL_ARROW_SCALE = 0.5  # m per m/s
                    dx, dy = vx * VEL_ARROW_SCALE, vy * VEL_ARROW_SCALE
                    ax.annotate("", xy=(x + dx, y + dy), xytext=(x, y),
                                arrowprops=dict(arrowstyle="->", color="darkred", lw=1.5, alpha=0.9))
                break
    ax.plot([], [], "-", color="green", linewidth=2, label="Normal (goal)")
    ax.plot([], [], "-", color="orange", linewidth=2, label="Fail-safe (stop + collision)")
    ax.plot([], [], "-", color="darkred", linewidth=2, label="Velocity (min-cost)")

    # Best path (raw + smoothed): color by speed (same viridis scale)
    path_speeds = [_speed_at_pos((p[0], p[1]), green=True) for p in path] if path else []
    if path and len(path) > 1:
        segs_path = [[[path[i][0], path[i][1]], [path[i+1][0], path[i+1][1]]] for i in range(len(path)-1)]
        colors_path = [cmap(norm_speed(path_speeds[i+1])) for i in range(len(path)-1)]
        lc_path = LineCollection(segs_path, colors=colors_path, linewidths=1.2, alpha=0.7)
        ax.add_collection(lc_path)
    path_smooth_speeds = []
    if path_smooth:
        if path and len(path_speeds) > 0:
            for i in range(len(path_smooth)):
                t = i / max(1, len(path_smooth) - 1)
                j = min(int(round(t * (len(path_speeds) - 1))), len(path_speeds) - 1)
                path_smooth_speeds.append(path_speeds[j])
        else:
            path_smooth_speeds = [v_min] * len(path_smooth)
        if len(path_smooth) > 1:
            segs_smooth = [[[path_smooth[i][0], path_smooth[i][1]], [path_smooth[i+1][0], path_smooth[i+1][1]]] for i in range(len(path_smooth)-1)]
            colors_smooth = [cmap(norm_speed(path_smooth_speeds[i+1])) for i in range(len(path_smooth)-1)]
            lc_smooth = LineCollection(segs_smooth, colors=colors_smooth, linewidths=3, alpha=0.9)
            ax.add_collection(lc_smooth)
        # Vehicle bounding boxes along best path (single color)
        for i in range(0, len(path_smooth), max(1, len(path_smooth) // 12)):
            x, y = path_smooth[i]
            psi = path_smooth_psi[i] if i < len(path_smooth_psi) else start[2]
            corners = _vehicle_rect_corners(x, y, psi)
            poly = Polygon(corners, facecolor="lime", edgecolor="darkgreen", alpha=0.25, linewidth=0.8)
            ax.add_patch(poly)
    else:
        print("No path found; showing exploration tree only.")

    start_xy = (start[0], start[1])
    ax.plot(start_xy[0], start_xy[1], "go", markersize=12, label="Start")
    psi0 = start[2]
    ax.annotate("", xy=(start_xy[0] + 2 * math.cos(psi0), start_xy[1] + 2 * math.sin(psi0)),
                xytext=start_xy, arrowprops=dict(arrowstyle="->", color="green", lw=2))
    # Label: Initial state (this area is designated start, base map unchanged)
    ax.annotate("Initial state", xy=(start_xy[0], start_xy[1] - 3), fontsize=10, ha="center",
                color="darkgreen", weight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="darkgreen", alpha=0.9))
    ax.plot(goal[0], goal[1], "r*", markersize=14, label="Goal")
    # Axis limits: full map, ensure all nodes/edges visible (sparse may place nodes at map edges)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)  # flip y for image (origin upper)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Green-orange double-tree RRT: Normal operation + Fail-safe (stop / min-cost collision)")
    ax.legend(loc="upper left")
    # Colorbar for speed (nodes and branches)
    sm = mcm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=v_min, vmax=v_max))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
    cbar.set_label("Speed (m/s)")
    plt.tight_layout()

    out_path = os.path.join(scenario_dir, "..", "..", "figs", "rrt_tjunction_paths.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")
    print(f"Feasible: {len(nodes_feasible)} nodes | Orange: {len(nodes_accident)} ({n_stop} stop, {len(nodes_accident)-n_stop} collision) | Path: {len(path)}")
    print(f"Collision cost: border={BORDER_COST_FACTOR}x, object={OBJECT_COST_FACTOR}x (prefer object over border)")
    print(f"Nodes drained (removed by sparse): {n_drained}")
    v0 = start[3]
    k_at_v = max_curvature_at_speed(v0)
    r_max = v0 * k_at_v
    ay_max = v0 * v0 * k_at_v
    print("Dynamical limits (at v=%.0f m/s):" % v0)
    print("  Max steering angle: %.2f rad (%.0f deg)" % (MAX_STEER_RAD, math.degrees(MAX_STEER_RAD)))
    print("  Max curvature: %.4f rad/m (lateral-accel limited)" % k_at_v)
    print("  Max yaw rate: %.2f rad/s (%.0f deg/s)" % (r_max, math.degrees(r_max)))
    print("  Max lateral accel: %.2f m/s^2 (%.2f g)" % (ay_max, ay_max / 9.81))
    return 0


if __name__ == "__main__":
    sys.exit(main())
