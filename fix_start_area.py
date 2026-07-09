#!/usr/bin/env python3
"""
Restore the start area in the map PNG to original appearance (dark/obstacle-like).
The display will show the base map; run_rrt_tjunction patches the collision grid
so the start area is treated as free for path planning.
"""
import os
import numpy as np
from PIL import Image

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MAP_PNG = os.path.join(THIS_DIR, "maps", "tjunction_map.png")
MAP_PGM = os.path.join(THIS_DIR, "maps", "tjunction_map.pgm")
PPM = 10

# Start position and vehicle dimensions (match run_rrt_tjunction.py)
START_X, START_Y = 41.5, 5.0
VEH_LENGTH, VEH_WIDTH = 2.80, 1.80
VEH_SCALE, VEH_LENGTH_SCALE = 2.8, 1.2
hl = VEH_LENGTH * VEH_SCALE * VEH_LENGTH_SCALE / 2
hw = VEH_WIDTH * VEH_SCALE / 2
margin = 3

x_min = START_X - hw - margin
x_max = START_X + hw + margin
y_min = START_Y - hl - margin
y_max = START_Y + hl + margin

img = Image.open(MAP_PNG)
arr = np.array(img)
h, w = arr.shape[0], arr.shape[1]
ymax_world = h / PPM

c0 = max(0, int(x_min * PPM))
c1 = min(w, int(x_max * PPM) + 1)
r0 = max(0, int((ymax_world - y_max) * PPM))
r1 = min(h, int((ymax_world - y_min) * PPM) + 1)
r0, r1 = min(r0, r1), max(r0, r1)

# Restore to dark gray (obstacle-like) for original base map appearance
DARK = 90
if arr.ndim == 3:
    arr[r0:r1, c0:c1, :3] = DARK
    if arr.shape[2] == 4:
        arr[r0:r1, c0:c1, 3] = 255
else:
    arr[r0:r1, c0:c1] = DARK

Image.fromarray(arr).save(MAP_PNG)
print(f"Restored start area to original appearance in {MAP_PNG}")

if os.path.exists(MAP_PGM):
    os.remove(MAP_PGM)
    print(f"Removed {MAP_PGM} (will be regenerated on next run)")
