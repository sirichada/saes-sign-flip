# Signed Re-Analysis: Per-Layer Reversals & Rank Stability

Extends `amplify_vs_suppress_comparison.md` per CLAUDE.md Step 1. Uses only cached JSON scores, no GPU.

**Caveat:** the raw `output_score` is non-negative by construction (`rank_output_score * top_token_score`, both in [0,1]) — confirmed empirically, min=0 in every cache. So 'reversal' below means the *paired delta* (amplify score − suppress score) goes negative, i.e. suppression scores higher than amplification for that feature — not that the raw score itself goes negative. A true below-*unsteered*-baseline test would need an amp_factor=0 cache, which does not exist yet.

## Pooled vs. per-layer correlation (Simpson's-paradox check)

Pooled across all layers: Spearman rho = -0.2717 (p=3.90e-18), Pearson r = 0.1064 (p=5.40e-08). The existing comparison report only cites the Pearson value (weak positive) as evidence output_score isn't dominated by a shared confound; the pooled Spearman is actually *negative*, which is if anything stronger support for that claim per CLAUDE.md's preference for rank correlation near a floor. But per-layer Spearman below is mostly **positive** (often 0.5-0.8) in early/mid layers — the pooled negative correlation is a between-layer artifact (early layers have low amplify + variable suppress; late layers have high amplify + suppress collapsed near-uniformly to ~0), not evidence within any single layer. Read the per-layer numbers, not the pooled one.

## Per-layer reversal fraction and rank correlation (amp_factor=-10)

| layer | n | reversals | rev% | spearman(amp,sup) | pearson(amp,sup) | amp mean | sup mean |
|---|---|---|---|---|---|---|---|
| 0 | 100 | 4 | 4.0% | 0.534 | 0.175 | 0.0093 | 0.0014 |
| 1 | 100 | 6 | 6.0% | 0.310 | 0.580 | 0.0106 | 0.0041 |
| 2 | 100 | 15 | 15.0% | 0.734 | 0.453 | 0.0029 | 0.0007 |
| 3 | 100 | 14 | 14.0% | 0.509 | 0.797 | 0.0100 | 0.0023 |
| 4 | 100 | 13 | 13.0% | 0.653 | -0.010 | 0.0159 | 0.0023 |
| 5 | 100 | 13 | 13.0% | 0.635 | 0.669 | 0.0076 | 0.0019 |
| 6 | 100 | 11 | 11.0% | 0.595 | 0.352 | 0.0191 | 0.0035 |
| 7 | 100 | 11 | 11.0% | 0.584 | 0.042 | 0.0241 | 0.0027 |
| 8 | 100 | 9 | 9.0% | 0.603 | 0.790 | 0.0113 | 0.0006 |
| 9 | 100 | 11 | 11.0% | 0.656 | 0.323 | 0.0089 | 0.0010 |
| 10 | 100 | 6 | 6.0% | 0.729 | 0.436 | 0.0144 | 0.0014 |
| 11 | 100 | 7 | 7.0% | 0.649 | 0.801 | 0.0021 | 0.0000 |
| 12 | 100 | 2 | 2.0% | 0.602 | 0.986 | 0.0042 | 0.0006 |
| 13 | 100 | 2 | 2.0% | 0.760 | 0.384 | 0.0265 | 0.0050 |
| 14 | 100 | 3 | 3.0% | 0.695 | 0.267 | 0.0312 | 0.0046 |
| 15 | 100 | 5 | 5.0% | 0.502 | 0.572 | 0.0693 | 0.0012 |
| 16 | 100 | 1 | 1.0% | 0.423 | 0.308 | 0.0398 | 0.0006 |
| 17 | 100 | 0 | 0.0% | 0.147 | 0.133 | 0.1629 | 0.0014 |
| 18 | 100 | 0 | 0.0% | 0.169 | 0.218 | 0.2031 | 0.0039 |
| 19 | 100 | 0 | 0.0% | -0.142 | 0.344 | 0.1747 | 0.0002 |
| 20 | 100 | 0 | 0.0% | 0.093 | 0.180 | 0.1600 | 0.0001 |
| 21 | 100 | 0 | 0.0% | -0.205 | 0.115 | 0.1948 | 0.0041 |
| 22 | 100 | 0 | 0.0% | -0.003 | 0.121 | 0.2480 | 0.0000 |
| 23 | 100 | 0 | 0.0% | 0.012 | -0.005 | 0.2330 | 0.0000 |
| 24 | 100 | 0 | 0.0% | 0.047 | -0.116 | 0.2622 | 0.0000 |
| 25 | 100 | 0 | 0.0% | 0.114 | 0.059 | 0.2096 | 0.0000 |

## Reversal rate across the magnitude sweep

| amp_factor | reversals | reversal % |
|---|---|---|
| -2 | 130 | 5.0% |
| -10 | 133 | 5.1% |
| -20 | 66 | 2.5% |

Features reversing at all three magnitudes: **16** (12.0% of the 133 that reverse at -10). Reversing only at -10: **53**.


**Most of those 16 'stable' reversals are floor-noise ties** (all four scores round to ~0.0000 at 4dp — both conditions are at the metric floor, so which one is nominally larger is meaningless). Requiring at least one score to clear 0.01 leaves only **4** features:

| layer_feature | amp | sup(-2) | sup(-10) | sup(-20) |
|---|---|---|---|---|
| 14_6669 | 0.0727 | 0.1832 | 0.2343 | 0.3867 |
| 2_13823 | 0.0002 | 0.0004 | 0.0202 | 0.0007 |
| 7_12166 | 0.0001 | 0.0004 | 0.0025 | 0.0218 |
| 7_6944 | 0.0002 | 0.1582 | 0.0902 | 0.0106 |

Only `14_6669` (layer 14) shows a clean, monotonically-increasing-with-magnitude reversal (0.073 → 0.183 → 0.234 → 0.387 as amp_factor goes 10 → -2 → -10 → -20) — the single strongest candidate for a genuine bidirectional/suppression-sensitive feature in this dataset, and worth a qualitative generation spot-check per CLAUDE.md Step 4. The other three non-trivial candidates are small and non-monotonic across magnitude, consistent with noise rather than a real effect.

## Rank stability of suppression scores across magnitude

- suppress(-2) vs suppress(-10): Spearman rho = 0.7828
- suppress(-10) vs suppress(-20): Spearman rho = 0.8662
- suppress(-2) vs suppress(-20): Spearman rho = 0.5841
