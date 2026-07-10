"""
Signed re-analysis of the amplify/suppress output_score caches (CLAUDE.md Step 1).

Uses only the existing JSON caches under data/output_scores/ — no GPU, no model
loading. Extends amplify_vs_suppress_comparison.md with:
  - per-layer reversal fractions (suppress > amplify), not just the aggregate
  - decays-to-floor vs. crosses-below-baseline split, using amplify score as the
    baseline proxy (no amp_factor=0 / unsteered cache exists yet to do this properly)
  - Spearman rank-stability across the -2/-10/-20 magnitude sweep, per CLAUDE.md's
    guidance to prefer rank over raw magnitude near a floor
  - whether the reversal set is a stable population across magnitudes or noise
"""
import json
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr, pearsonr, wilcoxon

CACHE_DIR = "data/output_scores"


def load_scores(name):
    with open(f"{CACHE_DIR}/{name}.json") as f:
        d = json.load(f)
    return d.get("scores", d)


def layer_of(key):
    return int(key.split("_")[0])


def main():
    amplify = load_scores("gemma2_2b_amplify")
    sup10 = load_scores("gemma2_2b_suppress")
    sup2 = load_scores("gemma2_2b_suppress_amp2")
    sup20 = load_scores("gemma2_2b_suppress_amp20")

    keys = sorted(set(amplify) & set(sup10) & set(sup2) & set(sup20))
    print(f"common keys across all 4 caches: {len(keys)}")
    dropped = (set(amplify) | set(sup10) | set(sup2) | set(sup20)) - set(keys)
    if dropped:
        print(f"WARNING: {len(dropped)} keys not present in all 4 caches, excluded: {sorted(dropped)[:10]}...")

    amp = np.array([amplify[k] for k in keys])
    s2 = np.array([sup2[k] for k in keys])
    s10 = np.array([sup10[k] for k in keys])
    s20 = np.array([sup20[k] for k in keys])
    layers = np.array([layer_of(k) for k in keys])

    # ---- 1. Signed delta (amp - suppress), baseline proxy = amplify score ----
    # "Baseline" here is the amplify condition, since no true unsteered (amp=0) run
    # is cached. This is NOT the off-distribution-safe baseline CLAUDE.md ultimately
    # wants (see caveat printed at the end) but it's the best available signed split
    # with existing data.
    delta10 = amp - s10
    reversal10 = delta10 < 0  # suppress > amplify: "crosses below baseline"
    decay10 = ~reversal10  # suppress <= amplify: "decays toward floor"

    print("\n=== Overall (amp_factor=-10) ===")
    print(f"n={len(keys)}, reversals={reversal10.sum()} ({reversal10.mean():.1%}), "
          f"decays={decay10.sum()} ({decay10.mean():.1%})")

    rho, p = spearmanr(amp, s10)
    r, pp = pearsonr(amp, s10)
    print(f"amplify vs suppress(-10): Spearman rho={rho:.4f} (p={p:.2e}), Pearson r={r:.4f} (p={pp:.2e})")

    # ---- 2. Per-layer reversal fraction + rank correlation ----
    print("\n=== Per-layer: reversal fraction and rank correlation (amp_factor=-10) ===")
    print(f"{'layer':>5} {'n':>5} {'revers.':>8} {'rev%':>6} {'spearman':>9} {'pearson':>8} "
          f"{'amp_mean':>9} {'sup_mean':>9}")
    layer_stats = {}
    for layer in sorted(set(layers)):
        mask = layers == layer
        n = mask.sum()
        rev = reversal10[mask].sum()
        rho_l, _ = spearmanr(amp[mask], s10[mask]) if n > 2 else (float("nan"), None)
        r_l, _ = pearsonr(amp[mask], s10[mask]) if n > 2 else (float("nan"), None)
        layer_stats[layer] = dict(n=int(n), reversals=int(rev), rev_frac=rev / n,
                                   spearman=rho_l, pearson=r_l,
                                   amp_mean=float(amp[mask].mean()), sup_mean=float(s10[mask].mean()))
        print(f"{layer:>5} {n:>5} {rev:>8} {rev/n:>6.1%} {rho_l:>9.3f} {r_l:>8.3f} "
              f"{amp[mask].mean():>9.4f} {s10[mask].mean():>9.4f}")

    # ---- 3. Magnitude sweep: is the reversal set stable, or floor noise? ----
    print("\n=== Reversal rate across magnitude sweep (suppress > amplify) ===")
    for mag, s in [(-2, s2), (-10, s10), (-20, s20)]:
        rev = (amp - s) < 0
        print(f"amp_factor={mag:>4}: reversals={rev.sum()} ({rev.mean():.1%})")

    rev2 = (amp - s2) < 0
    rev20 = (amp - s20) < 0
    stable_all3 = reversal10 & rev2 & rev20
    print(f"\nfeatures reversing at ALL THREE magnitudes (-2, -10, -20): {stable_all3.sum()} "
          f"of {reversal10.sum()} that reverse at -10 "
          f"({stable_all3.sum() / max(reversal10.sum(),1):.1%} overlap)")
    only_at_10 = reversal10 & ~rev2 & ~rev20
    print(f"features reversing ONLY at -10 (not at -2 or -20 -- likely noise near the floor): "
          f"{only_at_10.sum()}")

    # Most "stable" reversals are ties between values that round to ~0 at 4dp --
    # both conditions are at the metric floor, not a real effect. Require at least
    # one of the four scores to clear a magnitude threshold to call it non-trivial.
    max_score = np.maximum.reduce([amp, s2, s10, s20])
    nontrivial_stable = stable_all3 & (max_score > 0.01)
    print(f"\nof those, non-trivial (some score > 0.01, i.e. not just floor-noise ties): "
          f"{nontrivial_stable.sum()}")
    nontrivial_rows = []
    for i in np.where(nontrivial_stable)[0]:
        row = (keys[i], float(amp[i]), float(s2[i]), float(s10[i]), float(s20[i]))
        nontrivial_rows.append(row)
        print(f"  {row[0]:>10}: amp={row[1]:.4f}  sup(-2)={row[2]:.4f}  sup(-10)={row[3]:.4f}  sup(-20)={row[4]:.4f}")

    # ---- 4. Rank stability across magnitudes among suppression scores themselves ----
    print("\n=== Rank stability of suppression scores across magnitude (Spearman) ===")
    rho_2_10, _ = spearmanr(s2, s10)
    rho_10_20, _ = spearmanr(s10, s20)
    rho_2_20, _ = spearmanr(s2, s20)
    print(f"suppress(-2) vs suppress(-10): rho={rho_2_10:.4f}")
    print(f"suppress(-10) vs suppress(-20): rho={rho_10_20:.4f}")
    print(f"suppress(-2) vs suppress(-20): rho={rho_2_20:.4f}")

    # ---- 5. Wilcoxon per layer (does the aggregate significance hold within layers?) ----
    print("\n=== Per-layer Wilcoxon signed-rank test (amp vs suppress(-10)) ===")
    for layer in sorted(set(layers)):
        mask = layers == layer
        try:
            stat, p = wilcoxon(amp[mask], s10[mask])
        except ValueError:
            stat, p = float("nan"), float("nan")
        print(f"layer {layer:>2}: n={mask.sum():>4}  p={p:.3e}")

    # ---- Write markdown report ----
    lines = []
    lines.append("# Signed Re-Analysis: Per-Layer Reversals & Rank Stability\n")
    lines.append("Extends `amplify_vs_suppress_comparison.md` per CLAUDE.md Step 1. "
                  "Uses only cached JSON scores, no GPU.\n")
    lines.append("**Caveat:** the raw `output_score` is non-negative by construction "
                  "(`rank_output_score * top_token_score`, both in [0,1]) — confirmed empirically, "
                  "min=0 in every cache. So 'reversal' below means the *paired delta* "
                  "(amplify score − suppress score) goes negative, i.e. suppression scores higher "
                  "than amplification for that feature — not that the raw score itself goes negative. "
                  "A true below-*unsteered*-baseline test would need an amp_factor=0 cache, which "
                  "does not exist yet.\n")

    lines.append(f"## Pooled vs. per-layer correlation (Simpson's-paradox check)\n")
    lines.append(f"Pooled across all layers: Spearman rho = {rho:.4f} (p={p:.2e}), "
                  f"Pearson r = {r:.4f} (p={pp:.2e}). The existing comparison report only cites the "
                  f"Pearson value (weak positive) as evidence output_score isn't dominated by a shared "
                  f"confound; the pooled Spearman is actually *negative*, which is if anything stronger "
                  f"support for that claim per CLAUDE.md's preference for rank correlation near a floor. "
                  f"But per-layer Spearman below is mostly **positive** (often 0.5-0.8) in early/mid "
                  f"layers — the pooled negative correlation is a between-layer artifact (early layers "
                  f"have low amplify + variable suppress; late layers have high amplify + suppress "
                  f"collapsed near-uniformly to ~0), not evidence within any single layer. Read the "
                  f"per-layer numbers, not the pooled one.\n")

    lines.append("## Per-layer reversal fraction and rank correlation (amp_factor=-10)\n")
    lines.append("| layer | n | reversals | rev% | spearman(amp,sup) | pearson(amp,sup) | amp mean | sup mean |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for layer in sorted(layer_stats):
        s = layer_stats[layer]
        lines.append(f"| {layer} | {s['n']} | {s['reversals']} | {s['rev_frac']:.1%} | "
                      f"{s['spearman']:.3f} | {s['pearson']:.3f} | {s['amp_mean']:.4f} | {s['sup_mean']:.4f} |")

    lines.append("\n## Reversal rate across the magnitude sweep\n")
    lines.append("| amp_factor | reversals | reversal % |")
    lines.append("|---|---|---|")
    for mag, s in [(-2, s2), (-10, s10), (-20, s20)]:
        rev = (amp - s) < 0
        lines.append(f"| {mag} | {rev.sum()} | {rev.mean():.1%} |")

    lines.append(f"\nFeatures reversing at all three magnitudes: **{stable_all3.sum()}** "
                  f"({stable_all3.sum() / max(reversal10.sum(),1):.1%} of the {reversal10.sum()} "
                  f"that reverse at -10). Reversing only at -10: **{only_at_10.sum()}**.\n")

    lines.append(f"\n**Most of those {stable_all3.sum()} 'stable' reversals are floor-noise ties** "
                  f"(all four scores round to ~0.0000 at 4dp — both conditions are at the metric "
                  f"floor, so which one is nominally larger is meaningless). Requiring at least one "
                  f"score to clear 0.01 leaves only **{nontrivial_stable.sum()}** features:\n")
    lines.append("| layer_feature | amp | sup(-2) | sup(-10) | sup(-20) |")
    lines.append("|---|---|---|---|---|")
    for row in nontrivial_rows:
        lines.append(f"| {row[0]} | {row[1]:.4f} | {row[2]:.4f} | {row[3]:.4f} | {row[4]:.4f} |")
    lines.append("\nOnly `14_6669` (layer 14) shows a clean, monotonically-increasing-with-magnitude "
                  "reversal (0.073 → 0.183 → 0.234 → 0.387 as amp_factor goes 10 → -2 → -10 → -20) — "
                  "the single strongest candidate for a genuine bidirectional/suppression-sensitive "
                  "feature in this dataset, and worth a qualitative generation spot-check per CLAUDE.md "
                  "Step 4. The other three non-trivial candidates are small and non-monotonic across "
                  "magnitude, consistent with noise rather than a real effect.\n")

    lines.append("## Rank stability of suppression scores across magnitude\n")
    lines.append(f"- suppress(-2) vs suppress(-10): Spearman rho = {rho_2_10:.4f}\n"
                  f"- suppress(-10) vs suppress(-20): Spearman rho = {rho_10_20:.4f}\n"
                  f"- suppress(-2) vs suppress(-20): Spearman rho = {rho_2_20:.4f}\n")

    with open(f"{CACHE_DIR}/signed_reanalysis.md", "w") as f:
        f.write("\n".join(lines))
    print(f"\nWrote {CACHE_DIR}/signed_reanalysis.md")


if __name__ == "__main__":
    main()
