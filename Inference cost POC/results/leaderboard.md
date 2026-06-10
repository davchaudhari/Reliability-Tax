# Leaderboard — reliability-per-dollar

Aggregated across matched runs. `rpd_norm` = successful tasks per USD at the reference
token price (normalized view). Higher success is NOT the same as higher reliability/$.

| strategy | success | 95% CI | abstain | calls/task | norm $/task | reliability/$ |
|---|---|---|---|---|---|---|
| baseline | 0.725 | [0.57, 0.85] | 0.00 | 1.0 | 7.00e-05 | 10354 |
| abstain | 0.725 | [0.57, 0.85] | 0.00 | 1.0 | 7.00e-05 | 10363 |
| verifier_rerank | 0.725 | [0.59, 0.86] | 0.00 | 10.0 | 7.85e-04 | 924 |
| self_consistency | 0.717 | [0.57, 0.85] | 0.00 | 5.0 | 3.52e-04 | 2039 |
| self_correct | 0.700 | [0.56, 0.83] | 0.00 | 3.1 | 2.89e-04 | 2425 |
