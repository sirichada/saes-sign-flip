*AI-assistance disclosure: see [`AI_ASSISTANCE.md`](./AI_ASSISTANCE.md).*

# Reconstruction Error & Off-Manifold Distance

Uses only the `diagnostics` blocks cached by `output_score.py --log_recon_error`, no GPU.

**Method:** for each feature, at the last token, we log the L2 norms `clean_recon_err` = ‖a_i − dec(enc(a_i))‖ (amp-independent SAE fidelity), `steer_dist` = ‖ã_i − a_i‖ (how far the intervention moved the activation), and `offmanifold_err` = ‖ã_i − dec(enc(ã_i))‖ (how far the *steered* point lands off the SAE manifold — the key artifact metric), plus the post-Eq.6 `steered_latent`.

**Caveat:** JumpReLU enforces non-negative activations, so any negative `steered_latent` (all suppression conditions on a neutral prompt) is structurally novel territory the decoder never saw in training. A pure null cannot by itself distinguish genuine asymmetry from a decoder off-distribution artifact — that is exactly what the amplify-vs-suppress and zero-anchor comparisons below are for.

## Sanity check

`clean_recon_err` is amp-independent, so it must match across conditions for the same feature. Max per-feature relative spread across the 6 present conditions: **0.00e+00** (≈0 expected).

## Per-layer off-manifold error — amplify(+10) vs suppress(−10)

If suppression is a generic off-distribution artifact, the two columns track each other. If it is a sign-specific JumpReLU effect, suppress ≫ amplify. Wilcoxon signed-rank on the paired per-feature difference within each layer.

| layer | n | amp off-mani | sup off-mani | sup−amp | wilcoxon p |
|---|---|---|---|---|---|
| 0 | 100 | 2674.3224 | 2162.4817 | -511.8406 | 4.25e-04 |
| 1 | 100 | 1415.1991 | 1172.7026 | -242.4964 | 7.22e-06 |
| 2 | 100 | 721.7128 | 542.3447 | -179.3681 | 1.55e-04 |
| 3 | 100 | 1346.4507 | 1184.9180 | -161.5326 | 1.16e-01 |
| 4 | 100 | 1151.8141 | 938.9177 | -212.8965 | 1.83e-01 |
| 5 | 100 | 1472.3894 | 1289.2614 | -183.1280 | 5.33e-02 |
| 6 | 100 | 1180.7980 | 1221.1470 | 40.3490 | 3.91e-02 |
| 7 | 100 | 1197.0044 | 1256.6881 | 59.6837 | 3.25e-02 |
| 8 | 100 | 900.5607 | 982.9152 | 82.3544 | 6.85e-05 |
| 9 | 100 | 887.9443 | 1006.4240 | 118.4797 | 2.68e-04 |
| 10 | 100 | 928.3019 | 969.4660 | 41.1641 | 4.58e-02 |
| 11 | 100 | 798.2864 | 916.5122 | 118.2258 | 9.12e-05 |
| 12 | 100 | 1087.9852 | 1157.7774 | 69.7922 | 4.66e-03 |
| 13 | 100 | 1037.9004 | 1049.9367 | 12.0363 | 1.18e-01 |
| 14 | 100 | 927.2856 | 963.2962 | 36.0106 | 4.21e-02 |
| 15 | 100 | 1353.7743 | 1496.6278 | 142.8535 | 4.81e-03 |
| 16 | 100 | 1475.2214 | 1447.6034 | -27.6180 | 2.88e-01 |
| 17 | 100 | 2316.9587 | 2603.6757 | 286.7170 | 3.92e-05 |
| 18 | 100 | 2319.4172 | 2700.6683 | 381.2511 | 2.79e-04 |
| 19 | 100 | 2161.0821 | 2724.7285 | 563.6464 | 5.14e-08 |
| 20 | 100 | 1982.1839 | 2625.7054 | 643.5214 | 9.55e-09 |
| 21 | 100 | 1940.2097 | 2350.5683 | 410.3585 | 2.02e-06 |
| 22 | 100 | 2249.3601 | 2608.6797 | 359.3196 | 4.96e-04 |
| 23 | 100 | 2812.1156 | 3223.0462 | 410.9306 | 1.58e-04 |
| 24 | 100 | 3071.3761 | 3162.5407 | 91.1646 | 2.41e-01 |
| 25 | 100 | 1311.0774 | 1094.8677 | -216.2097 | 5.11e-05 |

## Off-manifold error vs suppression magnitude

Does the steered activation fall further off-manifold as |s| grows (magnitude-driven), independent of the reversal story?

| condition | mean off-mani | mean steer_dist | mean |steered_latent| |
|---|---|---|---|
| suppress(-2) | 129.8749 | 108.7585 | 108.7337 |
| suppress(-10) | 1648.2116 | 543.7925 | 543.7676 |
| suppress(-20) | 4596.1793 | 1087.5850 | 1087.5601 |

## Anchor comparison — does weirdness start at zero, or only past it?

`unsteered(0)` leaves the latent natural; `zero-ablate` sets it to 0 (removes the feature's contribution, sits at the JumpReLU boundary); `suppress(-10)` pushes it negative. If off-manifold error is already high at zero-ablation, the effect is removal, not negativity. If it only rises once past zero into negative s, that is the sign-specific signature.

| condition | mean off-mani | mean steer_dist |
|---|---|---|
| unsteered(0) | 63.3730 | 0.0000 |
| zero-ablate | 63.3732 | 0.0248 |
| suppress(-10) | 1648.2116 | 543.7925 |

## Spot-check candidates vs their layer population

Is each candidate's off-manifold error an outlier within its layer, or ordinary? This quantitatively firms up `step4_coherence_spotcheck.md`'s manual coherence call on `14_6669` (whose output score rose monotonically with |s| as generation degraded).

Reference condition: **suppress(-10)**. Percentile = fraction of that layer's features with lower off-manifold error than the candidate.

| feature | off-mani | layer mean | layer pctile | steered_latent |
|---|---|---|---|---|
| 14_6669 | 734.8422 | 963.2962 | 58.0% | -341.0607 |
| 2_13823 | 124.3919 | 542.3447 | 0.0% | -201.6973 |
| 7_12166 | 1605.8407 | 1256.6881 | 73.0% | -260.3003 |
| 7_6944 | 1255.7278 | 1256.6881 | 61.0% | -260.3003 |

## Summary interpretation

**Per-layer amp-vs-suppress: mixed, not uniform.** Layers 0–2 and 25 show suppression's off-manifold error *significantly below* amplification's (sup−amp negative, p<1.6e-04) — the opposite of the sign-specific prediction. Layers 6–23 show the reverse: suppression significantly exceeds amplification, most strongly and consistently in layers 17–23 (sup−amp of +287 to +644, all p<4e-04) and in layers 8, 9, 11 (p<3e-04). Layers 3, 4, 13, 16, 24 show no significant difference either way. So the sign-specific effect is real but **layer-localized to the middle-to-late layers (roughly 6–23, strongest at 17–23)**, not a repo-wide phenomenon — the earliest layers (0–2) and the final layer (25) actually behave oppositely. This is exactly the kind of pattern the "report per layer" guardrail exists to surface: a single pooled statistic would have averaged the early-layer reversal against the late-layer effect and likely produced a misleadingly muted (or sign-ambiguous) aggregate.

**Magnitude sweep: superlinear, not just proportional.** off-mani/steer_dist rises from ≈1.19 (s=−2) to ≈3.03 (s=−10) to ≈4.23 (s=−20). If off-manifold error were simply proportional to how far the intervention moved the point, this ratio would stay flat. Instead it grows with |s|, meaning larger negative pushes land disproportionately further off the decoder's manifold per unit of displacement — consistent with pushing progressively deeper into territory JumpReLU's non-negativity constraint never let the decoder see in training.

**Zero-ablation anchor: the effect does not appear at zero.** `zero-ablate` (off-mani=63.3732) is statistically indistinguishable from `unsteered(0)` (63.3730) — a difference of 0.0002 against a `suppress(-10)` value of 1648.21 (26x higher). This directly resolves the "removal vs. negativity" question from the plan: reconstructing "this feature's contribution set to exactly 0" is just as easy for the decoder as reconstructing the natural unsteered activation. The off-manifold blow-up appears **only once the latent is pushed past zero into negative territory**, not at the JumpReLU boundary itself. This is the strongest single piece of evidence for the sign-specific (not generic-removal) story.

**14_6669 and the other 3 candidates are not outliers in their own layers.** Percentiles vs. their layer's suppress(-10) population: 14_6669 58%, 7_6944 61%, 7_12166 73%, 2_13823 0% (i.e. 2_13823's off-manifold error is near the *minimum* of layer 2, not elevated). None of the four sit in the tail of their layer's distribution — three are within 25 points of the median (50th percentile) and 2_13823 is unusually *clean*, not unusually corrupted. This means this report does **not** support attributing 14_6669's "reversal" (flagged in `signed_reanalysis.md` and `step4_coherence_spotcheck.md` — output score rising monotonically with |s| as generation degraded into repetition) to an exceptional off-manifold reconstruction artifact for that specific feature — its decoder fidelity under suppression is ordinary for a layer-14 feature. The degenerate-generation explanation from `step4_coherence_spotcheck.md` remains the better-supported account for that candidate specifically, even though the *population*-level effect (previous two paragraphs) is real.

**Decision gate: stays closed for the flagged candidates; partially open only for a layer-scoped population claim.** The project's decision gate opens the "true suppression" extension only if off-distribution novelty looks like it's masking a real effect, and says to skip it if the signal looks partial or symmetric. The signal here is explicitly partial: sign-specific in layers ~6–23 (strongest 17–23) but reversed in layers 0–2 and 25, and — critically for prioritizing follow-up work — the specific features flagged as candidate reversals by `signed_reanalysis.md` and `step4_coherence_spotcheck.md` show no elevated off-manifold error at all. Per the gate's own "skip if partial" criterion, this does **not** justify pursuing "true suppression" (activating-prompt) validation for the 4 spot-check candidates. If the extension is pursued at all, the population-level layer-17–23 effect is the more defensible target — but that is a different, broader claim than the one motivating `step4_coherence_spotcheck.md`'s circularity concern, and would need its own feature selection rather than reusing `spot_check_reversal_candidates.json`.
