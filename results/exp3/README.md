# Experiment 3 — Fix the tolerance-blind hole classifier (agent: `scripted`)

Seeds: [0, 1, 2]. Test sets: 2,000 tenant-A and 2,000 tenant-B holes (held out). Few-shot: 16 tenant-B samples. Forward budget: 3000.

| method | B accuracy | B リーマ recall | B キリ precision | A accuracy (regression) | A リーマ recall | changed weights | gradient steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| before | 83.1% ± 4.0% | 34.7% ± 16.1% | 70.3% ± 5.2% | 100.0% ± 0.0% | 100.0% ± 0.0% | 0.0000% | 0 |
| wsp_route | 100.0% ± 0.0% | 100.0% ± 0.0% | 100.0% ± 0.0% | 100.0% ± 0.0% | 100.0% ± 0.0% | 0.9092% | 0 |
| wsp_route_ablate | 100.0% ± 0.0% | 100.0% ± 0.0% | 100.0% ± 0.0% | 83.3% ± 0.6% | 35.9% ± 2.3% | 1.7525% | 0 |
| ft_head | 99.5% ± 0.8% | 97.9% ± 3.0% | 98.7% ± 1.9% | 99.9% ± 0.1% | 100.0% ± 0.0% | 2.8945% | 200 |
| ft_all | 99.6% ± 0.6% | 98.4% ± 2.3% | 99.0% ± 1.5% | 99.9% ± 0.2% | 100.0% ± 0.0% | 69.9038% | 200 |

Per seed (wsp_route): forward passes used / edits / changed parameters

- seed 0: 293 forward passes, 12 edits, 69 / 7589 parameters changed, patch `patches/seed0_wsp_route.{json,npz}`
- seed 1: 295 forward passes, 12 edits, 69 / 7589 parameters changed, patch `patches/seed1_wsp_route.{json,npz}`
- seed 2: 373 forward passes, 14 edits, 69 / 7589 parameters changed, patch `patches/seed2_wsp_route.{json,npz}`
