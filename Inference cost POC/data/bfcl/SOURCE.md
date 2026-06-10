# BFCL data provenance

These files are a subset of the **Berkeley Function-Calling Leaderboard (BFCL)** dataset,
fetched 2026-06-06 from the public Gorilla repository:

- Source: https://github.com/ShishirPatil/gorilla (`berkeley-function-call-leaderboard/bfcl_eval/data/`)
- Files vendored here (the single-turn, AST-checkable `simple_python` category only):
  - `BFCL_v4_simple_python.json` — 400 tasks (questions + available functions)
  - `possible_answer/BFCL_v4_simple_python.json` — ground-truth call(s) per task

We vendor only this small, judge-free subset for the Phase 1 smoke test and Phase 2 sweep. The
full benchmark (multi-turn, live, etc.) is not included; see the BFCL caveat in the top-level
README. BFCL is released under the Gorilla repo's license (Apache-2.0); this subset is included
for reproducibility of a public benchmark, not redistributed as a new dataset.
