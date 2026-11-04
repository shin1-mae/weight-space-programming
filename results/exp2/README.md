# Experiment 2 — ECN edits in weight space (agent: `scripted`)

Chamfer distance / P95 in mm against the analytic CAD of the *changed* part (lower is better). `grad` = gradient steps used. Read-back = the Exp. 1 reader run on the resulting network.

| case | ECN | method | Chamfer | P95 | IoU | watertight | read-back Ø err | read-back t err | grad | sec |
|---|---|---|---:|---:|---:|:---:|---:|---:|---:|---:|
| ecn1_translate_z | 部品全体をZ方向に+5mmオフセットせよ。 | no_edit | 3.370 | 5.10 | 0.354 | ✅ | 0.06 | 0.05 | 0 | 0.0 |
| ecn1_translate_z |  | wsp_edit | 0.227 | 0.43 | 0.920 | ✅ | 0.06 | 0.05 | 0 | 7.9 |
| ecn1_translate_z |  | write_from_read | 0.215 | 0.41 | 0.896 | ✅ | 0.06 | 0.05 | 0 | 0.8 |
| ecn1_translate_z |  | retrain | 0.232 | 0.45 | 0.919 | ✅ | 0.12 | 0.02 | 3000 | 15.0 |
| ecn2_readme_example | 中心の穴の直径を2倍に広げ、Z軸方向に+5mmオフセットせよ。 | no_edit | 3.688 | 10.54 | 0.270 | ✅ | 20.94 | 0.05 | 0 | 0.0 |
| ecn2_readme_example |  | wsp_edit | 0.207 | 0.39 | 0.906 | ✅ | 0.13 | 0.01 | 0 | 9.6 |
| ecn2_readme_example |  | write_from_read | 0.202 | 0.39 | 0.884 | ✅ | 0.13 | 0.05 | 0 | 0.8 |
| ecn2_readme_example |  | retrain | 0.215 | 0.40 | 0.897 | ✅ | 0.00 | 0.10 | 3000 | 15.6 |
| ecn3_thickness | 板厚を15mmに変更せよ（下面基準）。 | no_edit | 0.878 | 2.07 | 0.789 | ✅ | – | 2.00 | 0 | 0.0 |
| ecn3_thickness |  | wsp_edit | 0.247 | 0.47 | 0.914 | ✅ | – | 0.00 | 0 | 8.6 |
| ecn3_thickness |  | write_from_read | 0.240 | 0.47 | 0.986 | ✅ | – | 0.00 | 0 | 0.6 |
| ecn3_thickness |  | retrain | 0.248 | 0.47 | 0.910 | ✅ | – | 0.00 | 3000 | 17.0 |
| ecn4_offset_flange | 外形全体を0.5mm肉盛り（オフセット）せよ。 | no_edit | 0.549 | 0.72 | 0.926 | ✅ | 0.77 | 0.96 | 0 | 0.0 |
| ecn4_offset_flange |  | wsp_edit | 0.212 | 0.40 | 0.956 | ✅ | 0.23 | 0.03 | 0 | 14.7 |
| ecn4_offset_flange |  | write_from_read | 0.208 | 0.40 | 0.906 | ✅ | 0.23 | 0.04 | 0 | 0.8 |
| ecn4_offset_flange |  | retrain | 0.217 | 0.41 | 0.941 | ✅ | 0.10 | 0.04 | 3000 | 17.7 |
| ecn5_add_hole | (x=+15, y=-10) にφ8の貫通穴を追加せよ。 | no_edit | 0.252 | 0.43 | 0.884 | ✅ | – | 0.05 | 0 | 0.0 |
| ecn5_add_hole |  | wsp_edit | 0.198 | 0.38 | 0.898 | ✅ | 0.00 | 0.04 | 0 | 9.7 |
| ecn5_add_hole |  | write_from_read | 0.195 | 0.37 | 0.886 | ✅ | 0.00 | 0.05 | 0 | 0.9 |
| ecn5_add_hole |  | retrain | 0.204 | 0.39 | 0.896 | ✅ | 0.00 | 0.02 | 3000 | 16.2 |
| ecn6_shrink_hole | 穴径をφ10に縮小せよ。 | no_edit | 0.304 | 0.54 | 0.855 | ✅ | 6.02 | 0.07 | 0 | 0.0 |
| ecn6_shrink_hole |  | wsp_edit | 0.220 | 0.42 | 0.879 | ✅ | 0.00 | 0.09 | 0 | 11.4 |
| ecn6_shrink_hole |  | write_from_read | 0.218 | 0.42 | 0.860 | ✅ | 0.00 | 0.07 | 0 | 0.8 |
| ecn6_shrink_hole |  | retrain | 0.227 | 0.44 | 0.875 | ✅ | 0.15 | 0.05 | 3000 | 14.7 |
| ecn7_enlarge_flange_bore | 中心の穴の直径を1.5倍に広げよ。 | no_edit | 0.583 | 5.13 | 0.799 | ✅ | 10.33 | 0.00 | 0 | 0.0 |
| ecn7_enlarge_flange_bore |  | wsp_edit | 0.192 | 0.36 | 0.923 | ✅ | 0.25 | 0.01 | 0 | 10.2 |
| ecn7_enlarge_flange_bore |  | write_from_read | 0.187 | 0.36 | 0.882 | ✅ | 0.25 | 0.00 | 0 | 1.4 |
| ecn7_enlarge_flange_bore |  | retrain | 0.200 | 0.37 | 0.922 | ✅ | 0.01 | 0.01 | 3000 | 14.2 |

## Mean over cases

| method | Chamfer [mm] | P95 [mm] | IoU | watertight | gradient steps | seconds |
|---|---:|---:|---:|---:|---:|---:|
| no_edit | 1.375 | 3.51 | 0.697 | 100% | 0 | 0.0 |
| wsp_edit | 0.215 | 0.41 | 0.914 | 100% | 0 | 10.3 |
| write_from_read | 0.209 | 0.40 | 0.900 | 100% | 0 | 0.9 |
| retrain | 0.220 | 0.42 | 0.909 | 100% | 3000 | 15.8 |
