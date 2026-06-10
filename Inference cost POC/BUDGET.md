# BUDGET.md — live cost ledger + pricing lookups

**Hard wall: $30 total Modal compute.** This is a wall, not a target. No GPU phase starts if its
estimated cost would push the running total over $30. The majority is reserved for the single
headline 7B run (Phase 3).

## Running total

| Date | Phase | Action | Est. $ | Measured $ (Modal) | Cumulative $ | Notes |
|------|-------|--------|--------|--------------------|--------------|-------|
| 2026-06-05 | 0 | Scaffold + mock pipeline | 0.00 | 0.00 | **0.00** | $0 — no GPU. Full sweep ran on deterministic mock. |
| 2026-06-08 | 1 | Deploy + smoke test (1.5B/L4) | 0.11 | ~0.07–0.13 (est; confirm on dashboard) | **~0.13** | Deployed vLLM 0.21.0; ran 5-task smoke. ~1–2 cold starts (~150s each) due to scaledown between manual probes. App stopped after. |
| 2026-06-08 | 2 | Small-model sweep (1.5B/L4) | 0.06 | ~0.06–0.07 (164s warmup + 93s sweep + ~30s idle ≈ 287s alive) | **~0.20** | 40 tasks × 5 strategies × 3 seeds = 600 results / 2640 calls, concurrent (16 workers). Projection (~$0.06) held almost exactly. App stopped after. |
| 2026-06-09 | 3 | Headline 7B/L4: tau-bench debug + BFCL-hard sweep | 0.12 (BFCL) | ~0.45–0.55 (many debugging cold starts; see note) | **~0.70** | 7B on L4. BFCL parallel_multiple sweep (40×5×3, ~$0.12) + tau-bench integration debugging (several 7B cold starts ~150–446s each incl. one 15GB weight download) + sample transcripts. App stopped after. |

**Cumulative spend: ~$0.70 / $30.00** (estimate; exact figure on the Modal dashboard). ~$29.30
remains — the project came in at ~2.3% of the budget wall.

### Phase 3 reconciliation — honest about the debugging overhead
The BFCL-hard sweep itself was cheap (~$0.12, projection held). The rest of Phase 3 cost went to
**iterating on the tau-bench integration against the real 7B**: a deploy bug (the `MODEL_NAME` env
var didn't propagate to the container — it served the 1.5B and 404'd) cost an extra cold start, and
getting the agent loop right took several single-episode diagnostics, each paying a 7B cold start
(~150s cached; ~446s the first time including the one-time 15GB weight download). A killed sweep
(piped through `tail`, so no logs + no checkpoint) also burned partial GPU time. Lessons recorded:
write runs to a log file (never `tail`), checkpoint long runs, and verify the served model before
the batch. Total Phase 3 ≈ $0.45–0.55; the avoidable debugging overhead was perhaps half of that.

### Phase 2 reconciliation — the concurrency lever worked
- **Projected** (alive-time model, 16 workers): ~$0.06 measured, ~271s alive.
- **Actual:** 164s cold start + 93s concurrent sweep + ~30s idle ≈ **287s alive ≈ $0.064.** The
  projection was within ~6% — the Phase-1-calibrated model is trustworthy for Phase 3 planning.
- Concurrency (continuous batching) ran 2,640 calls in **93s** of sweep wall-clock; the same work
  sequential was projected at ~977s. The GPU rental, not the tokens, is what concurrency saves.

### Phase 1 cost reconciliation (the two-cost-views caveat, made concrete)
This is the central methodological payoff and it showed up cleanly:
- **Pre-run estimate:** ~$0.11 (container alive-time × L4 rate).
- **Per-request attribution** (my harness, summed request wall-clock at steady state): **$0.0016.**
  This captures ONLY inference compute on a warm server — it does **not** see the GPU rental
  overhead (cold start: ~150s weight-load + CUDA-graph compile).
- **Actual alive-time bill:** ~$0.07–0.13, **dominated by cold start**, which the per-request view
  entirely misses.
- The ~40–80× gap between the per-request view ($0.0016) and the alive-time bill (~$0.10) **is**
  the measurement caveat this project exists to expose. Self-hosting cost is rental-time, not
  token-time; the normalized-token view and the measured-GPU view diverge for exactly this reason.

**Process lessons (recorded so Phase 2/3 are clean):**
- `scaledown_window=60s` is too short for stepwise manual work — the container scaled to zero
  between probes and re-paid cold start. For the sweep: warm once, then run the full batch
  immediately in one process, or raise `scaledown_window` for the run and `modal app stop` after.
- The cold start (~150s) is a fixed per-cold-start tax; amortize it by running all
  tasks/strategies/seeds in a single warm window rather than many short invocations.

### Phase 1 also surfaced a measurement BUG (fixed, $0 to verify)
The smoke test's real model outputs revealed the BFCL AST checker was **under-counting ~4×**: it
ignored positional arguments and stripped namespace-qualified names (`math.factorial`). Re-scoring
the *same saved outputs* with the fixed checker (no GPU re-spend) moved baseline from 0.20 → 0.80.
Fixes + regression tests committed. Had this not been caught, the entire Phase 2/3 sweep would have
reported success rates 4× too low. This is the payoff of smoke-testing before the big spend.

---

## Modal GPU pricing (looked up 2026-06-05)

Source: https://modal.com/pricing (cross-checked against cloudgpuprices.com/vendors/modal).
**Billing is per-second; $0 while a container is idle / scaled to zero.**

| GPU | $/second | ≈ $/hour | Fits Qwen2.5-1.5B? | Fits Qwen2.5-7B (bf16)? |
|-----|----------|----------|--------------------|--------------------------|
| T4 (16GB)    | 0.000164 | ~0.59 | yes | tight/awkward |
| **L4 (24GB)** | **0.000222** | **~0.80** | yes | yes (~15GB weights in 24GB) |
| A10G (24GB)  | 0.000306 | ~1.10 | yes | yes (more throughput headroom) |
| A100 40GB    | 0.000583 | ~2.10 | overkill | yes |
| A100 80GB    | 0.000694 | ~2.50 | overkill | yes |
| H100         | 0.001097 | ~3.95 | overkill | yes |

Rates are mirrored in `src/cost.py:MODAL_GPU_USD_PER_SEC`. **Default GPU = L4** (cheapest that
fits the 1.5B comfortably and the 7B in bf16). Fall back to A10G only if L4 OOMs or throughput is
poor for the headline run.

### Cost-control settings baked into `modal_app.py`
- Weights cached in a `modal.Volume` at `/root/.cache/huggingface` → downloaded **once**.
- vLLM compile cache in a second Volume → faster cold starts.
- `scaledown_window=60s` → container scales to zero shortly after idle (no warm-GPU billing).
- `@modal.concurrent(max_inputs=64)` → many requests batched per GPU so we push tokens through
  during the rented window rather than paying for idle time.

## Reference token prices (normalized cost view)

Configurable placeholders in `src/cost.py:TokenPrice` (default 0.20 / 0.60 USD per 1M prompt /
completion tokens). **Override per run** with `--prompt-price` / `--completion-price` and record
the provider + date here when used for a reported number. These are NOT the real bill; see the
two-cost-views caveat in the README.

---

## Phase cost estimates (to be filled before each GPU phase)

### Phase 1 — smoke test (target < $1)  [ESTIMATE READY — awaiting go-ahead]
- Model: Qwen2.5-1.5B-Instruct on **L4** @ $0.000222/sec ($0.80/hr), per-second, scale-to-zero.
- Workload: ~5 BFCL `simple_python` tasks × (baseline + self_consistency n=3) × 1 seed ≈ 20 calls.
- **Cost is container ALIVE-time × rate** (cold start + serving + idle-until-scaledown), not just
  the 20 inference calls. Breakdown (conservative):
  | component | ~seconds | note |
  |---|---|---|
  | weight download (1.5B ~3GB → Volume) | ~90 | one-time; cached after first run |
  | model load + CUDA-graph compile | ~120 | every cold start |
  | serving ~20 calls | ~60 | low concurrency |
  | idle until scaledown (window=60s) | ~60 | billed before scale-to-zero |
  - **Worst-case single run ≈ 330s → ~$0.073.** With 50% safety margin → **~$0.11.**
- Plus a one-time **image build** (CUDA base + vLLM pip install) on `modal deploy`; Modal builds
  images on its builder infra — typically not GPU-billed — but recorded here and reconciled
  against the actual Modal bill after the run.
- Hard kill-switch: `scripts/smoke_test.py --budget 1.00` aborts mid-run if measured cost exceeds
  $1. Comfortably under the <$1 target and far under the $30 wall.
- **STATUS: ✅ DONE (2026-06-08).** Deployed, smoke-tested 5 tasks, app stopped. Actual ~$0.07–0.13
  (cold-start-dominated; estimate held). Surfaced + fixed a 4× checker under-count. See the
  reconciliation + bug notes in the running-total section above.

### Phase 2 — small-model sweep ✅ DONE (2026-06-08)
- Qwen2.5-1.5B, 40 BFCL simple_python tasks, all 5 strategies, 3 seeds, concurrent (16 workers).
- Actual ~$0.06 (projection held within 6%). App stopped after.
- **Scientific result (honest, and a little anticlimactic):** the 1.5B is already strong on
  single-turn BFCL simple (baseline 0.825), so the reliability strategies barely move the needle —
  self_consistency / verifier_rerank reach 0.842 (+1.7pp, within overlapping CIs), self_correct
  *drops* to 0.817 (reflexion sometimes breaks a correct answer). reliability-per-dollar therefore
  strongly favors baseline (~18k successes/$ normalized) over verifier_rerank (~1.6k). The
  reliability tax is real and steep, but its ROI depends entirely on **baseline headroom** — on an
  easy task there's almost nothing to buy.
- **abstain never fired:** the model's mean logprob stayed in [−0.33, −0.00] (mean −0.06), all above
  the −0.8 threshold, so abstain ≡ baseline. Not a bug (logprobs were captured); the threshold is
  inert when the model is confident-and-correct. Tuning the threshold / using harder tasks is the
  knob.
- **Implication for Phase 3:** a *better* model (7B) on the *same* easy tasks would saturate even
  more, hiding strategy value. To make the pass@1-vs-pass@k and strategy-separation stories land,
  Phase 3 needs tasks with real headroom (harder BFCL categories: parallel / multiple /
  parallel_multiple, or multi-turn; or tau-bench). See README → Phase 3 plan.

### Phase 3 — headline run ✅ DONE (2026-06-09)
- Qwen2.5-7B-Instruct on L4. Two deliverables:
  1. **Quantitative headline:** BFCL `parallel_multiple` (harder, single-turn, AST-checkable), all
     5 strategies, 40 tasks × 3 seeds. baseline **0.725** — the "productive middle" (real headroom,
     not saturated like 1.5B/simple). **Yet every strategy is dominated by baseline:** self_consistency
     0.717, self_correct 0.700, verifier_rerank 0.725 (8–10× cost), abstain never fires. The Pareto
     frontier is a single point (baseline).
  2. **tau-bench (airline):** the multi-turn integration runs faithfully (coherent agent, instrumented
     user-sim, programmatic reward) but the 7B scores **0** — too hard. Kept as a validated capability
     with clean sample transcripts (`results/taubench_airline_7b_transcripts.txt`); full quantitative
     tau-bench is future work with a stronger model.
- **Why strategies don't help (measured, not hand-waved):** the 7B is near-deterministic on function
  calling — across 120 task-seeds, all 5 samples were identical in 98 (mean unique answers 1.24/5),
  and pass@k barely exceeds pass@1. Low output diversity ⇒ self-consistency/verifier-rerank have
  almost nothing to exploit. The tax is real; what it buys depends on diversity/error-structure, and
  programmatic tool-use is a low-diversity regime.

---

## Reconciliation note (measured vs. attributed GPU-seconds)
Our per-result GPU-seconds (sum of request wall-clock) **over-counts** under request concurrency,
because overlapping requests share one GPU. So `measured_gpu_usd` from `src/cost.py` is an
**upper bound** on attributed cost. The ground truth is Modal's reported usage for the run window.
After each GPU phase, record BOTH numbers above and note the gap.
