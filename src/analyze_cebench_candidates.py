"""
Step 6 (Track C) — CPU analysis pass for the CE-Bench per-feature cross-check.
Consumes the caches written by src/cebench_extract.py; no GPU, no model, no steering.

Scoring is pinned to CE-Bench's CODE (ce_bench/CE_Bench_v2.py), with each
paper/code discrepancy recorded in the report:
  - per-pair normalization is a z-score over neurons (code), not the paper's
    min-max (paper Sec 2.2); min-max is reported as a sensitivity check.
  - I1 is the token-weighted mean over BOTH stories' tokens (code), not V1+V2
    (paper Sec 2.3).
  - the code's independence baseline build loop is buggy as published
    (`subject_averaged_activations` starts empty and iterates over itself, so it
    never fills); we implement its evident intent — per-subject averages of I1,
    baseline I2 for a pair = mean of all OTHER subjects' averages
    (leave-one-subject-out) — which also differs from the paper's global I_avg.

Per-feature statistic (pinned, per plan): a feature's contrastive (resp.
independence, interpretability) score = MAX over story pairs of its per-pair
normalized value — CE-Bench's pooling direction (max over the score vector),
applied over pairs; the MEAN over pairs is reported as the sensitivity
alternative. Each candidate is placed as a percentile within its own SAE
dictionary (the other d_sae-1 latents on the same data = the null distribution).

Construct guardrail: this measures FEATURE INTERPRETABILITY (unsteered
activation divergence on contrastive text). It is not steering effectiveness
(the output score) and must never be conflated with it.
"""
import argparse
import json
import os
from collections import defaultdict
from datetime import date

import numpy as np

CACHE_DIR = "data/cebench"
CANDIDATES_PATH = "data/features/spot_check_reversal_candidates.json"
REPORT = "reports/step6_cebench_crosscheck.md"
OUT_JSON = "data/output_scores/step6_cebench_metrics.json"


def zscore_rows(x):
    """Per-pair (row-wise) z-score over neurons — CE_Bench_v2.py's normalization."""
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (x - mu) / sd


def minmax_rows(x):
    """Per-pair min-max normalization — the PAPER's stated normalization (sensitivity)."""
    lo = x.min(axis=1, keepdims=True)
    rng = x.max(axis=1, keepdims=True) - lo
    rng[rng == 0] = 1.0
    return (x - lo) / rng


def percentile_of(stat_vec, idx):
    """Percentile rank of feature idx within its dictionary (0-100, higher = stronger)."""
    return 100.0 * (stat_vec < stat_vec[idx]).mean()


def load_layer(layer, n_pairs):
    V1 = np.load(os.path.join(CACHE_DIR, f"V1_layer{layer}.npy"), mmap_mode="r")[:n_pairs]
    V2 = np.load(os.path.join(CACHE_DIR, f"V2_layer{layer}.npy"), mmap_mode="r")[:n_pairs]
    return np.asarray(V1, dtype=np.float64), np.asarray(V2, dtype=np.float64)


def analyze_layer(V1, V2, n1, n2, subjects):
    """Returns dict of per-feature stat vectors + per-layer SAE-level scalars."""
    # contrastive: C = |V1 - V2| per pair, z-scored per pair (code); min-max sensitivity
    C = np.abs(V1 - V2)
    Cz = zscore_rows(C)

    # independence: I1 = token-weighted mean over both stories (code);
    # baseline = leave-one-subject-out mean of per-subject averages (code intent)
    w1 = (n1 / (n1 + n2))[:, None]
    I1 = w1 * V1 + (1 - w1) * V2
    subj_idx = defaultdict(list)
    for i, s in enumerate(subjects):
        subj_idx[s].append(i)
    subj_names = list(subj_idx)
    subj_avg = np.stack([I1[subj_idx[s]].mean(axis=0) for s in subj_names])
    subj_pos = {s: k for k, s in enumerate(subj_names)}
    total = subj_avg.sum(axis=0)
    n_subj = len(subj_names)
    if n_subj < 2:
        raise ValueError("need >=2 subjects for the leave-one-subject-out baseline")
    I2 = np.stack([(total - subj_avg[subj_pos[s]]) / (n_subj - 1) for s in subjects])
    D = np.abs(I1 - I2)
    Dz = zscore_rows(D)

    # combined interpretability distance (code: contrast + independence, then z)
    T = C + D
    Tz = zscore_rows(T)

    stats = {}
    for name, z, dist in (("contrastive", Cz, C), ("independence", Dz, D), ("interpretability", Tz, T)):
        stats[name] = {
            "max": z.max(axis=0),            # pinned per-feature stat
            "mean": z.mean(axis=0),          # sensitivity alternative
            "minmax_max": minmax_rows(dist).max(axis=0),  # paper-normalization sensitivity
        }
    # SAE-level scalar, CE-Bench's actual benchmark quantity (pre sparsity penalty):
    # mean over pairs of the per-pair max over neurons
    sae_scalars = {name: float(z.max(axis=1).mean())
                   for name, z in (("contrastive", Cz), ("independence", Dz), ("interpretability", Tz))}
    return stats, sae_scalars


def selftest():
    """Validate the scoring math on synthetic data: a planted high-contrast neuron
    must land at the top percentile; an average neuron must sit mid-distribution."""
    rng = np.random.default_rng(0)
    n_pairs, d = 200, 512
    subjects = [f"s{i // 5}" for i in range(n_pairs)]  # 5 pairs per subject
    V1 = rng.gamma(1.0, 1.0, (n_pairs, d))
    V2 = rng.gamma(1.0, 1.0, (n_pairs, d))
    planted = 7
    V1[:, planted] += 25.0  # strong contrast: story1 always activates it, story2 never
    n1 = rng.integers(50, 200, n_pairs).astype(float)
    n2 = rng.integers(50, 200, n_pairs).astype(float)
    stats, scalars = analyze_layer(V1, V2, n1, n2, subjects)
    p_planted = percentile_of(stats["contrastive"]["max"], planted)
    p_null = percentile_of(stats["contrastive"]["max"], 42)
    assert p_planted > 99.5, f"planted neuron percentile {p_planted} (expected >99.5)"
    assert 1 < p_null < 99, f"null neuron percentile {p_null} (expected mid-distribution)"
    # the planted neuron should also determine the SAE-level contrastive scalar
    assert np.argmax(stats["contrastive"]["max"]) == planted
    print(f"selftest OK: planted neuron pct={p_planted:.1f}, null neuron pct={p_null:.1f}, "
          f"SAE contrastive scalar={scalars['contrastive']:.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true", help="validate scoring math on synthetic data")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return

    with open(os.path.join(CACHE_DIR, "progress.json")) as f:
        prog = json.load(f)
    n_pairs = prog["next"]
    if n_pairs < prog["n_pairs"]:
        print(f"[warn] extraction incomplete: {n_pairs}/{prog['n_pairs']} pairs — analyzing what exists")
    n1 = np.array(prog["n1"][:n_pairs], dtype=np.float64)
    n2 = np.array(prog["n2"][:n_pairs], dtype=np.float64)
    subjects = prog["subjects"][:n_pairs]

    with open(CANDIDATES_PATH) as f:
        candidates = json.load(f)

    results = {}
    for layer_str, feats in candidates.items():
        layer = int(layer_str)
        V1, V2 = load_layer(layer, n_pairs)
        stats, sae_scalars = analyze_layer(V1, V2, n1, n2, subjects)
        layer_res = {"sae_scalars": sae_scalars, "candidates": {}}
        for score_name, variants in stats.items():
            top = int(np.argmax(variants["max"]))
            layer_res.setdefault("top_neuron", {})[score_name] = {
                "index": top, "max": float(variants["max"][top])}
        for feat in feats:
            key = f"{layer}_{feat}"
            layer_res["candidates"][key] = {
                score_name: {
                    variant: {"value": float(vec[feat]), "percentile": percentile_of(vec, feat)}
                    for variant, vec in variants.items()}
                for score_name, variants in stats.items()}
        results[layer_str] = layer_res
        print(f"layer {layer}: done ({len(feats)} candidate(s), {n_pairs} pairs)")

    write_report(results, prog, n_pairs)
    meta = {"script": "src/analyze_cebench_candidates.py", "date": str(date.today()),
            "n_pairs": n_pairs, "extraction_meta": prog["meta"],
            "pinned_stat": "max over pairs of per-pair z-scored value (code normalization)",
            "sensitivity_stats": ["mean over pairs (z-score)", "max over pairs (min-max, paper normalization)"],
            "independence_baseline": "leave-one-subject-out mean of per-subject I1 averages (code intent; published code loop is buggy)"}
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "results": results}, f, indent=1)
    print(f"wrote {REPORT} and {OUT_JSON}")


def write_report(results, prog, n_pairs):
    lines = [
        "*AI-assistance disclosure: see [`AI_ASSISTANCE.md`](./AI_ASSISTANCE.md).*", "",
        "# Step 6: CE-Bench Per-Feature Interpretability Cross-Check", "",
        "## Summary", "", "*(placeholder — write after reading the tables below)*", "",
        "## What this is (and is not)", "",
        "This step applies **CE-Bench's contrastive methodology per-feature** (Gulko, Peng &",
        "Kumar, arXiv:2509.00691; `docs/papers/gulko-2025-ce-bench.md`) to the 4 spot-check",
        "candidates — it is **not** \"running the CE-Bench benchmark\": CE-Bench's released",
        "pipeline persists only the max-pooled per-SAE scalar, which cannot answer a",
        "per-feature question. It is the only measurement in this project taken with",
        "**steering switched off** — every other readout (output score, Step 4 coherence,",
        "Step 5 metrics) observes steered forward passes.", "",
        "**Construct guardrail:** this measures *feature interpretability* (unsteered",
        "activation divergence on contrastive story pairs), not *steering effectiveness*",
        "(the output score). A **low** percentile is steering-free corroboration that the",
        "candidate's apparent reversal was noise, not signal. A **high** percentile only",
        "rules out \"the feature is meaningless\" — it does **not** resolve the",
        "generation-artifact question (Steps 3–5's artifact reading stands either way);",
        "it would instead sharpen the illustration of the output score's blind spot:",
        "a genuinely interpretable feature whose bidirectional score is still an artifact.", "",
        "**Validity caveats:**",
        "1. CE-Bench was validated against SAEBench only at the SAE-scalar (max-pooled)",
        "   level; single-neuron CE-Bench scores have no per-feature ground-truth",
        "   validation. Results here are descriptive and internally calibrated: each",
        "   candidate vs. the other latents of its own dictionary, same data, same math.",
        "2. CE-Bench's own gemma-scope-2b experiments (its §4.4–4.5) ran on the",
        "   inference-only testbed with no SAEBench ground truth, so even SAE-level",
        "   validity on this exact SAE suite is CE-Bench agreeing with itself.", "",
        "**Method.** Dataset `GulkoA/contrastive-stories-v4`" + f" ({n_pairs} pairs analyzed); ",
        "model google/gemma-2-2b (unsteered); SAEs `gemma-scope-2b-pt-res-canonical`,",
        "`layer_{2,7,14}/width_16k/canonical`. Per pair: V1/V2 = mean SAE activation over",
        "each story's tokens; contrastive C = |V1−V2|; independence D = |I1 − I2| with",
        "I1 = token-weighted mean over both stories and I2 = leave-one-subject-out",
        "baseline; combined T = C + D. Each per-pair vector is z-scored over neurons.",
        "**Pinned per-feature statistic: max over pairs** of the z-scored value",
        "(CE-Bench's pooling direction, applied over pairs); sensitivity alternatives:",
        "mean over pairs, and max under the paper's min-max normalization.", "",
        "**Paper/code discrepancies pinned to the code** (recorded per the plan):",
        "z-score (code) vs. min-max (paper §2.2) per-pair normalization; I1 =",
        "token-weighted both-stories mean (code) vs. V1+V2 (paper §2.3); the published",
        "v2 independence-baseline loop is buggy (iterates an empty dict) — we implement",
        "its evident leave-one-subject-out intent, which itself differs from the paper's",
        "global I_avg.", "",
    ]

    for layer_str, res in results.items():
        lines += [f"## Layer {layer_str}", "",
                  "SAE-level scalars (CE-Bench's benchmark quantity, pre sparsity penalty, "
                  "mean over pairs of per-pair neuron max): "
                  + ", ".join(f"{k} = {v:.2f}" for k, v in res["sae_scalars"].items()) + ".",
                  "Top neuron per score: "
                  + ", ".join(f"{k}: `{v['index']}` (max {v['max']:.1f})"
                              for k, v in res["top_neuron"].items()) + ".", "",
                  "| candidate | score | max (pinned) | pctile | mean | pctile | min-max max | pctile |",
                  "|---|---|---|---|---|---|---|---|"]
        for key, scores in res["candidates"].items():
            for score_name, variants in scores.items():
                m, mn, mm = variants["max"], variants["mean"], variants["minmax_max"]
                lines.append(
                    f"| `{key}` | {score_name} | {m['value']:.2f} | {m['percentile']:.1f} "
                    f"| {mn['value']:.3f} | {mn['percentile']:.1f} "
                    f"| {mm['value']:.3f} | {mm['percentile']:.1f} |")
        lines.append("")

    lines += ["## Reading the percentiles", "",
              "Percentile = share of the candidate's own dictionary (d_sae latents, same",
              "data, same statistic) scoring strictly below it. Uniform-ish percentiles are",
              "expected for arbitrary latents; a candidate must sit in the extreme upper",
              "tail before \"responds to real semantic contrasts\" is supportable. Per-layer",
              "only — never compare raw values across layers (each layer is z-scored and",
              "distributed differently).", ""]

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
