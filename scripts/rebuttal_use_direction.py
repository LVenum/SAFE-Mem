"""Rebuttal: direction of use-classifier errors against human labels. Calls no model.

.venv/bin/python scripts/rebuttal_use_direction.py
"""

import json

from ams.eval.attribution import use_error_direction
from ams.eval.labels import _norm, _read

labels = {
    i: _norm(r["kind"]) for i, r in _read("data/labels/label_set_v2_use_final.csv", "id").items()
}
key = json.load(open("data/labels/label_key_v2_use.json"))
cols = ["n", "exact", "under", "under_by_1", "over", "boundary_under", "boundary_over"]
for bb in ("gpt56", "4omini"):
    print(f"\n### {bb}\n\n| group | " + " | ".join(cols) + " |\n|" + "---|" * (len(cols) + 1))
    for g, c in sorted(use_error_direction(labels, key, bb).items()):
        print(f"| {g} | " + " | ".join(str(c[k]) for k in cols) + " |")
