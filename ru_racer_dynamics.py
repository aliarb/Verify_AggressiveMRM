"""
RU-Racer vehicle dynamics from https://github.com/ru-racer/ru-racer

Equations of motion from matlab/Bike.m - bicycle model with Pacejka tire forces.
State: [x, y, psi, vx, vy, dotpsi] (body-frame velocities vx, vy)
Input: [delta, sfx] (steering angle, longitudinal slip)
"""

from __future__ import annotations

import math

# RU-Racer Bike.m parameters (scaled for full-size: m=1500, Izz=2500, L=2.8)
# Original: m=6.8, Izz=0.15, lf=0.15, lr=0.25 -> ratio lf:lr = 3:5
LF = 1.05   # m (front axle to CoG)
LR = 1.75   # m (rear axle to CoG)
M = 1500.0  # kg
IZZ = 2500.0  # kg·m²
G = 9.81
H = 0.5  # m (CoG height, typical full-size)

# Pacejka tire params (from Bike.m: tiref/tirer)
TIRE_F = {"B": 10.275, "C": 1.56, "D": -0.85}  # front (Bike.m sign convention)
TIRE_R = {"B": 10.275, "C": 1.56, "D": -0.79}  # rear


def _normalize_angle(a: float) -> float:
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def ru_racer_dynamics(
    x: float, y: float, psi: float, vx: float, vy: float, dotpsi: float,
    delta: float, sfx: float,
) -> tuple[float, float, float, float, float, float]:
    """
    RU-Racer EoM from Bike.m Dynamic().
    Returns (dotx, doty, dotpsi, dotvx, dotvy, dotdotpsi).
    """
    cd = math.cos(delta)
    sd = math.sin(delta)
    srx = sfx

    # Slip ratios (Bike.m)
    denom = vx * cd + (vy + LF * dotpsi) * sd
    if abs(denom) < 1e-6:
        denom = 1e-6 if denom >= 0 else -1e-6
    sfy = ((vy + LF * dotpsi) * cd - vx * sd) / denom
    if abs(vx) < 1e-6:
        sry = 0.0
    else:
        sry = (vy - LR * dotpsi) / vx

    sf = math.sqrt(sfy * sfy + sfx * sfx)
    sr = math.sqrt(sry * sry + srx * srx)

    # Pacejka: mu = D*sin(C*atan(B*s))
    muf = TIRE_F["D"] * math.sin(TIRE_F["C"] * math.atan(TIRE_F["B"] * sf))
    mur = TIRE_R["D"] * math.sin(TIRE_R["C"] * math.atan(TIRE_R["B"] * sr))

    if abs(sf) > 0.01:
        mufx = -muf * (sfx / sf)
        mufy = -muf * (sfy / sf)
    else:
        mufx = 0.0
        mufy = 0.0

    if abs(sr) > 0.001:
        murx = -mur * (srx / sr)
        mury = -mur * (sry / sr)
    else:
        murx = 0.0
        mury = 0.0

    # Normal forces (Bike.m)
    denom_fz = LF + LR + H * (mufx * cd - mufy * sd - murx)
    if abs(denom_fz) < 1e-6:
        denom_fz = 1e-6 if denom_fz >= 0 else -1e-6
    Ffz = M * G * (LR - murx * H) / denom_fz
    Frz = M * G - Ffz

    Ffx = mufx * Ffz
    Frx = murx * Frz
    Ffy = -mufy * Ffz
    Fry = -mury * Frz

    # EoM
    dotx = vx * math.cos(psi) - vy * math.sin(psi)
    doty = vx * math.sin(psi) + vy * math.cos(psi)
    dotvx = (Ffx * cd - Ffy * sd + Frx) / M + vy * dotpsi
    dotvy = (Ffx * sd + Ffy * cd + Fry) / M - vx * dotpsi
    dotdotpsi = (LF * (Ffx * sd + Ffy * cd) - LR * Fry) / IZZ

    return dotx, doty, dotpsi, dotvx, dotvy, dotdotpsi


def ru_racer_propagate(
    x: float, y: float, psi: float, vx: float, vy: float, dotpsi: float,
    delta: float, sfx: float, dt: float,
) -> tuple[float, float, float, float, float, float]:
    """Euler integrate RU-Racer dynamics for one step."""
    dx, dy, dpsi, dvx, dvy, ddotpsi = ru_racer_dynamics(
        x, y, psi, vx, vy, dotpsi, delta, sfx
    )
    x_new = x + dx * dt
    y_new = y + dy * dt
    psi_new = _normalize_angle(psi + dpsi * dt)
    vx_new = vx + dvx * dt
    vy_new = vy + dvy * dt
    dotpsi_new = dotpsi + ddotpsi * dt
    return x_new, y_new, psi_new, vx_new, vy_new, dotpsi_new


def v_omega_to_vx_vy(v: float, omega: float) -> tuple[float, float]:
    """Approximate (v, omega) from simplified model to (vx, vy) for RU-Racer.
    For near-straight: vx≈v, vy≈0. For turning: vx=v*cos(beta), vy=v*sin(beta).
    Simple mapping: vx=v, vy=0 when omega≈0; otherwise estimate sideslip."""
    if abs(omega) < 1e-6:
        return v, 0.0
    v_safe = max(0.5, abs(v))
    vx = v
    vy = 0.0  # RU-Racer will compute vy from dynamics
    return vx, vy


def vx_vy_to_v_omega(vx: float, vy: float, dotpsi: float) -> tuple[float, float]:
    """Convert RU-Racer state to (v, omega) for RRT node."""
    v = math.sqrt(vx * vx + vy * vy)
    return v, dotpsi
