# Scenario Validation Maps

Maps for scenario validation. File-based maps are used when `map_region.type: file` in scenario YAML.

## T-Junction Map (EMRM Scene 1)

**File:** `tjunction_map.png` (and `tjunction_map.pgm` for ru_racer)

**Workspace:** Scaled for full-size vehicle on a 2-sided (2-lane) road. ~47 m × 53 m.

**Occupancy (obstacle_threshold: 50):**
- **Black (pixel < 50):** Obstacles (truck, vehicles, curbs, buildings)
- **Gray / white (pixel ≥ 50):** Drivable (road surface, lane markings, initial state area)

**Scenario setup:**
- **Initial state:** Bottom-right car (our controlled vehicle), facing up
- **Obstacle:** Truck in middle of intersection

**Usage:** Referenced by `EMRM_Scene1_tjunction.yaml` via `map_file: maps/tjunction_map.png`.
