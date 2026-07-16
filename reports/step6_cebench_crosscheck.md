*AI-assistance disclosure: see [`AI_ASSISTANCE.md`](./AI_ASSISTANCE.md).*

# Step 6: CE-Bench Per-Feature Interpretability Cross-Check

## Summary

*(placeholder — write after reading the tables below)*

## What this is (and is not)

This step applies **CE-Bench's contrastive methodology per-feature** (Gulko, Peng &
Kumar, arXiv:2509.00691; `docs/papers/gulko-2025-ce-bench.md`) to the 4 spot-check
candidates — it is **not** "running the CE-Bench benchmark": CE-Bench's released
pipeline persists only the max-pooled per-SAE scalar, which cannot answer a
per-feature question. It is the only measurement in this project taken with
**steering switched off** — every other readout (output score, Step 4 coherence,
Step 5 metrics) observes steered forward passes.

**Construct guardrail:** this measures *feature interpretability* (unsteered
activation divergence on contrastive story pairs), not *steering effectiveness*
(the output score). A **low** percentile is steering-free corroboration that the
candidate's apparent reversal was noise, not signal. A **high** percentile only
rules out "the feature is meaningless" — it does **not** resolve the
generation-artifact question (Steps 3–5's artifact reading stands either way);
it would instead sharpen the illustration of the output score's blind spot:
a genuinely interpretable feature whose bidirectional score is still an artifact.

**Validity caveats:**
1. CE-Bench was validated against SAEBench only at the SAE-scalar (max-pooled)
   level; single-neuron CE-Bench scores have no per-feature ground-truth
   validation. Results here are descriptive and internally calibrated: each
   candidate vs. the other latents of its own dictionary, same data, same math.
2. CE-Bench's own gemma-scope-2b experiments (its §4.4–4.5) ran on the
   inference-only testbed with no SAEBench ground truth, so even SAE-level
   validity on this exact SAE suite is CE-Bench agreeing with itself.

**Method.** Dataset `GulkoA/contrastive-stories-v4` (5000 pairs analyzed); 
model google/gemma-2-2b (unsteered); SAEs `gemma-scope-2b-pt-res-canonical`,
`layer_{2,7,14}/width_16k/canonical`. Per pair: V1/V2 = mean SAE activation over
each story's tokens; contrastive C = |V1−V2|; independence D = |I1 − I2| with
I1 = token-weighted mean over both stories and I2 = leave-one-subject-out
baseline; combined T = C + D. Each per-pair vector is z-scored over neurons.
**Pinned per-feature statistic: max over pairs** of the z-scored value
(CE-Bench's pooling direction, applied over pairs); sensitivity alternatives:
mean over pairs, and max under the paper's min-max normalization.

**Paper/code discrepancies pinned to the code** (recorded per the plan):
z-score (code) vs. min-max (paper §2.2) per-pair normalization; I1 =
token-weighted both-stories mean (code) vs. V1+V2 (paper §2.3); the published
v2 independence-baseline loop is buggy (iterates an empty dict) — we implement
its evident leave-one-subject-out intent, which itself differs from the paper's
global I_avg.

## Layer 2

SAE-level scalars (CE-Bench's benchmark quantity, pre sparsity penalty, mean over pairs of per-pair neuron max): contrastive = 32.75, independence = 39.94, interpretability = 34.90.
Top neuron per score: contrastive: `15089` (max 99.6), independence: `15089` (max 94.8), interpretability: `15089` (max 99.6).

| candidate | score | max (pinned) | pctile | mean | pctile | min-max max | pctile |
|---|---|---|---|---|---|---|---|
| `2_13823` | contrastive | 71.91 | 100.0 | 4.814 | 99.8 | 1.000 | 98.0 |
| `2_13823` | independence | 41.63 | 99.6 | 7.132 | 99.9 | 1.000 | 98.3 |
| `2_13823` | interpretability | 58.83 | 100.0 | 6.387 | 99.9 | 1.000 | 98.3 |

## Layer 7

SAE-level scalars (CE-Bench's benchmark quantity, pre sparsity penalty, mean over pairs of per-pair neuron max): contrastive = 41.18, independence = 51.98, interpretability = 44.49.
Top neuron per score: contrastive: `4879` (max 105.5), independence: `2182` (max 108.3), interpretability: `2182` (max 101.7).

| candidate | score | max (pinned) | pctile | mean | pctile | min-max max | pctile |
|---|---|---|---|---|---|---|---|
| `7_12166` | contrastive | -0.07 | 5.2 | -0.283 | 4.8 | 0.007 | 6.1 |
| `7_12166` | independence | -0.12 | 4.5 | -0.309 | 4.5 | 0.002 | 4.3 |
| `7_12166` | interpretability | -0.15 | 4.4 | -0.324 | 4.6 | 0.004 | 4.1 |
| `7_6944` | contrastive | 23.00 | 92.7 | -0.225 | 29.4 | 0.441 | 84.4 |
| `7_6944` | independence | 16.15 | 88.2 | -0.178 | 40.6 | 0.425 | 89.2 |
| `7_6944` | interpretability | 21.53 | 91.7 | -0.223 | 35.2 | 0.435 | 86.9 |

## Layer 14

SAE-level scalars (CE-Bench's benchmark quantity, pre sparsity penalty, mean over pairs of per-pair neuron max): contrastive = 46.54, independence = 47.80, interpretability = 44.10.
Top neuron per score: contrastive: `12858` (max 98.9), independence: `71` (max 91.5), interpretability: `15887` (max 85.4).

| candidate | score | max (pinned) | pctile | mean | pctile | min-max max | pctile |
|---|---|---|---|---|---|---|---|
| `14_6669` | contrastive | 0.05 | 8.7 | -0.262 | 4.5 | 0.008 | 7.1 |
| `14_6669` | independence | -0.07 | 8.7 | -0.293 | 4.5 | 0.005 | 8.8 |
| `14_6669` | interpretability | 0.01 | 9.5 | -0.303 | 4.5 | 0.010 | 11.1 |

## Reading the percentiles

Percentile = share of the candidate's own dictionary (d_sae latents, same
data, same statistic) scoring strictly below it. Uniform-ish percentiles are
expected for arbitrary latents; a candidate must sit in the extreme upper
tail before "responds to real semantic contrasts" is supportable. Per-layer
only — never compare raw values across layers (each layer is z-scored and
distributed differently).
