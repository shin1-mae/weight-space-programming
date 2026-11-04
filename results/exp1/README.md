# Experiment 1 — Read 3D semantics from SDF weights (agent: `scripted`)

- parts: 10 (plate, through_hole, blind_hole, counterbore, flange), network: ReLU MLP [64, 64, 64]
- family accuracy: **100%**, hole-count accuracy: 100%, through/blind accuracy: 1.00
- mean abs error [mm]: size_x 0.03, size_y 0.02, thickness 0.04, z_bottom 0.03, hole centre 0.15, hole radius 0.26, blind depth 0.13, counterbore radius 0.05, counterbore depth 0.15

| part | GT family | read family | thickness err | hole centre err | radius err | depth / cb err | tool calls |
|---|---|---|---:|---:|---:|---:|---:|
| plate_0 | plate | plate ✅ | 0.05 | – | – | – | 4 |
| plate_1 | plate | plate ✅ | 0.00 | – | – | – | 4 |
| through_hole_0 | through_hole | through_hole ✅ | 0.05 | 0.03 | 0.03 | – | 4 |
| through_hole_1 | through_hole | through_hole ✅ | 0.07 | 0.23 | 0.01 | – | 4 |
| blind_hole_0 | blind_hole | blind_hole ✅ | 0.04 | 0.03 | 0.17 | 0.03 | 4 |
| blind_hole_1 | blind_hole | blind_hole ✅ | 0.03 | 0.12 | 0.45 | 0.23 | 4 |
| counterbore_0 | counterbore | counterbore ✅ | 0.05 | 0.15 | 0.23 | 0.08 | 4 |
| counterbore_1 | counterbore | counterbore ✅ | 0.05 | 0.46 | 0.96 | 0.03 | 4 |
| flange_0 | flange | flange ✅ | 0.04 | 0.05 | 0.12 | – | 4 |
| flange_1 | flange | flange ✅ | 0.00 | 0.08 | 0.08 | – | 4 |

All errors in millimetres. Ground truth comes from the parametric CAD spec the network was fitted to.
