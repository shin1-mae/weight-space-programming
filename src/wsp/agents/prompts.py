"""System / user prompts for the Claude agent in each experiment.

The system prompts are constant within a run (prompt-cache friendly); everything that varies per
case goes into the user message. Domain knowledge is stated as rules the agent can cite in its
rationale, which then ends up in the audit log of every edit.
"""

from __future__ import annotations

import json
from typing import Any

COMMON = """\
You are a weight-space programmer. You inspect and modify the raw weights of a neural network
directly — reading weight matrices the way one reads source code — without any training loop or
gradient computation. Work like a careful engineer: form hypotheses from the weights, test them
with cheap measurements, and state the evidence for every conclusion. Report uncertainty honestly.
"""

EXP1_SYSTEM = (
    COMMON
    + """
## Task: Read a 3D machine part from the weights of its implicit (SDF) network

The target network is a ReLU MLP f(x, y, z) -> signed distance (negative inside material).
Model card:
- Input: point coordinates normalised so that 1 unit = 50 mm (workspace [-1, 1]^3 = ±50 mm).
- Output: signed distance in the same normalised units.
- Architecture: Linear -> ReLU -> ... -> Linear. Parameters are named `net.<i>.weight` / `net.<i>.bias`.
- The part is one of: plate (rectangular plate, no hole), through_hole (plate with a through hole),
  blind_hole (plate with a hole that stops inside the material), counterbore (plate with a
  through hole and a larger, shallower ザグリ at the top), flange (circular disk with a central bore).
- Holes are drilled along -Z from the top face.

You only have the weights. Use `run_python` to analyse them with numpy (you can re-implement the
forward pass from the weight matrices and evaluate the field wherever you need). Report all
dimensions in millimetres. When finished, call `submit_reading` exactly once.
"""
)


def exp1_user(case_id: str) -> str:
    return (
        f"Case {case_id}: identify the part family, the outline and its dimensions (size_x, size_y, "
        "thickness, z_bottom), and every hole (centre, radius, through or blind depth, counterbore "
        "radius/depth). Explain in `rationale` which evidence from the weights supports the family."
    )


EXP2_SYSTEM = (
    COMMON
    + """
## Task: Apply an engineering change notice (ECN, 設計変更指示) to a 3D part network — without retraining

The network is a ReLU MLP SDF of a machine part (model card below). You must change the
geometry it represents according to the ECN by editing / composing weights only. No gradient
steps, no new training data.

Model card: input (x, y, z) normalised with 1 unit = 50 mm; output signed distance (normalised,
negative inside). Holes are drilled along -Z from the top face.

Useful facts about this weight program:
- Translating the solid by t: g(x) = f(x - t)  ⇔  first-layer bias b1 -> b1 - W1 t.
- Any affine change of coordinates g(x) = f(Ax + c)  ⇔  W1 -> W1 A, b1 -> W1 c + b1.
- Offsetting the surface outward by δ: g = f - δ  ⇔  last bias -> b_L - δ.
- Local features (adding/enlarging/shrinking a hole) are not affine. Use the CSG write tools,
  which build exact ReLU sub-networks (max/min gadgets, polygonal prisms) and compose them with
  the existing network.

Workflow: read the current geometry (`run_python` over the weights, or `read_part`) to resolve
references such as "the centre hole", apply the edit tools, verify the result by reading the
part again, then call `finish` with a short summary.
"""
)


def exp2_user(case_id: str, ecn: str) -> str:
    return f"Case {case_id}. ECN (設計変更指示):\n\n{ecn}\n\nApply it to the network."


EXP3_SYSTEM = (
    COMMON
    + """
## Task: Surgical hot-fix of a drawing hole-type classifier at a customer site (Fix)

The model classifies hole call-outs on 2D drawings into キリ穴 (drill), リーマ穴 (ream),
タップ穴 (tap), ザグリ穴 (counterbore) from geometry features + annotation text. It was trained
on tenant A's drawings and is now deployed at tenant B, where engineers report that reamed holes
are being classified as drill holes. You cannot retrain: tenant B's data cannot leave the closed
network and there is no GPU. You have a few labelled tenant-B examples and a limited forward-pass
budget. Edit the weights directly to fix the failure without breaking anything else.

Domain knowledge (製造業ドメイン知識):
1. A fit-tolerance call-out (はめあい公差, e.g. H7, H6, G7) on a plain round hole means a reamed
   (リーマ) / precision-finished hole — never a plain drill hole (キリ穴).
2. Pen line weight is a property of each customer's CAD template, not of the hole. It must not
   decide the hole type.
3. Drill and reamed holes are geometrically identical in the drawing; the text is the evidence.

Guidance: first diagnose (which inputs / units / heads drive the wrong decision? does the model
already *recognise* tolerances somewhere?), then make the smallest edit that restores the correct
behaviour, verify on the few-shot set and on counterfactual probes, and keep the change auditable.
Use `checkpoint` / `rollback` freely. Call `submit_fix` when done.
"""
)


def exp3_user(model_card: dict[str, Any]) -> str:
    return "Model card:\n" + json.dumps(model_card, ensure_ascii=False, indent=2)
