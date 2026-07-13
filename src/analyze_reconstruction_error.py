"""
Reconstruction-error / activation-distance analysis (CLAUDE.md Step 3).

Consumes the `diagnostics` blocks in the `_diag` output_score caches plus the two
anchor caches (unsteered amp=0, zero-ablation ã_i=0). Uses only cached JSON, no GPU.

Distinguishes two explanations for suppression's effects that the output score alone
cannot separate:
  - off-distribution decoder artifact: off-manifold error comparably bad in BOTH
    steering directions (decoder just misbehaves off-manifold), vs.
  - sign-specific JumpReLU effect: error worse specifically for negative s, because
    JumpReLU enforces non-negative activations so any negative latent is novel.

Reports per-layer (not just aggregate), sweeps |s|, compares the zero-ablation and
unsteered anchors against negative s, and spotlights the 4 spot-check candidates.
"""
import json
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon

CACHE_DIR = "data/output_scores"
CANDIDATES = {"2_13823", "7_12166", "7_6944", "14_6669"}

# (label, cache basename). Missing files are skipped with a warning.
CONDITIONS = [
    ("amplify(+10)", "gemma2_2b_amplify_diag"),
    ("suppress(-2)", "gemma2_2b_suppress_amp2_diag"),
    ("suppress(-10)", "gemma2_2b_suppress_diag"),
    ("suppress(-20)", "gemma2_2b_suppress_amp20_diag"),
    ("zero-ablate", "gemma2_2b_zero_ablation"),
    ("unsteered(0)", "gemma2_2b_unsteered"),
]

FIELDS = ["clean_recon_err", "steer_dist", "offmanifold_err", "steered_latent", "max_act_value"]


def load_diagnostics(name):
    """Return the diagnostics dict for a cache, or None if the file/field is absent."""
    try:
        with open(f"{CACHE_DIR}/{name}.json") as f:
            d = json.load(f)
    except FileNotFoundError:
        return None
    diag = d.get("diagnostics")
    return diag if diag else None


def layer_of(key):
    return int(key.split("_")[0])


def main():
    diags = {label: load_diagnostics(name) for label, name in CONDITIONS}
    present = {label: d for label, d in diags.items() if d}
    missing = [label for label, d in diags.items() if not d]
    if missing:
        print(f"WARNING: no diagnostics for: {missing} (run those conditions with --log_recon_error)")
    if not present:
        print("No diagnostics caches found. Produce the _diag caches first (see the Step 3 plan).")
        return

    keys = sorted(set.intersection(*[set(d) for d in present.values()]))
    print(f"common keys across {len(present)} present conditions: {len(keys)}")
    layers = np.array([layer_of(k) for k in keys])

    # field arrays: arr[label][field] aligned to `keys`
    arr = {}
    for label, d in present.items():
        arr[label] = {f: np.array([d[k].get(f, np.nan) for k in keys]) for f in FIELDS}

    amp_label = "amplify(+10)" if "amplify(+10)" in present else None
    sup_label = "suppress(-10)" if "suppress(-10)" in present else None

    lines = []
    lines.append("# Step 3 — Reconstruction Error & Off-Manifold Distance\n")
    lines.append("CLAUDE.md Step 3. Uses only the `diagnostics` blocks cached by `output_score.py "
                 "--log_recon_error`, no GPU.\n")
    lines.append("**Method:** for each feature, at the last token, we log the L2 norms "
                 "`clean_recon_err` = ‖a_i − dec(enc(a_i))‖ (amp-independent SAE fidelity), "
                 "`steer_dist` = ‖ã_i − a_i‖ (how far the intervention moved the activation), and "
                 "`offmanifold_err` = ‖ã_i − dec(enc(ã_i))‖ (how far the *steered* point lands off "
                 "the SAE manifold — the key artifact metric), plus the post-Eq.6 `steered_latent`.\n")
    lines.append("**Caveat:** JumpReLU enforces non-negative activations, so any negative "
                 "`steered_latent` (all suppression conditions on a neutral prompt) is structurally "
                 "novel territory the decoder never saw in training. A pure null cannot by itself "
                 "distinguish genuine asymmetry from a decoder off-distribution artifact — that is "
                 "exactly what the amplify-vs-suppress and zero-anchor comparisons below are for.\n")

    # ---- 1. Sanity: clean_recon_err stable across conditions ----
    print("\n=== Sanity: clean_recon_err spread across conditions (should be ~identical) ===")
    if len(present) > 1:
        stack = np.vstack([arr[l]["clean_recon_err"] for l in present])
        spread = np.nanmax(stack, axis=0) - np.nanmin(stack, axis=0)
        rel = spread / (np.nanmean(stack, axis=0) + 1e-12)
        print(f"max per-feature relative spread of clean_recon_err: {np.nanmax(rel):.2e} "
              f"(should be ~0; large => bug)")
        lines.append("## Sanity check\n")
        lines.append(f"`clean_recon_err` is amp-independent, so it must match across conditions "
                     f"for the same feature. Max per-feature relative spread across the "
                     f"{len(present)} present conditions: **{np.nanmax(rel):.2e}** (≈0 expected).\n")

    # ---- 2. Per-layer off-manifold error: amplify vs suppress ----
    if amp_label and sup_label:
        print("\n=== Per-layer off-manifold error: amplify(+10) vs suppress(-10) ===")
        a = arr[amp_label]["offmanifold_err"]
        s = arr[sup_label]["offmanifold_err"]
        lines.append("## Per-layer off-manifold error — amplify(+10) vs suppress(−10)\n")
        lines.append("If suppression is a generic off-distribution artifact, the two columns track "
                     "each other. If it is a sign-specific JumpReLU effect, suppress ≫ amplify. "
                     "Wilcoxon signed-rank on the paired per-feature difference within each layer.\n")
        lines.append("| layer | n | amp off-mani | sup off-mani | sup−amp | wilcoxon p |")
        lines.append("|---|---|---|---|---|---|")
        print(f"{'layer':>5} {'n':>5} {'amp':>10} {'sup':>10} {'sup-amp':>10} {'wilcox_p':>10}")
        for layer in sorted(set(layers)):
            mask = layers == layer
            am, sm = np.nanmean(a[mask]), np.nanmean(s[mask])
            try:
                _, p = wilcoxon(a[mask], s[mask])
            except ValueError:
                p = float("nan")
            print(f"{layer:>5} {mask.sum():>5} {am:>10.4f} {sm:>10.4f} {sm-am:>10.4f} {p:>10.2e}")
            lines.append(f"| {layer} | {mask.sum()} | {am:.4f} | {sm:.4f} | {sm-am:.4f} | {p:.2e} |")

    # ---- 3. Off-manifold error vs |s| sweep ----
    sweep = [l for l in ["suppress(-2)", "suppress(-10)", "suppress(-20)"] if l in present]
    if len(sweep) >= 2:
        print("\n=== Off-manifold error vs suppression magnitude ===")
        lines.append("\n## Off-manifold error vs suppression magnitude\n")
        lines.append("Does the steered activation fall further off-manifold as |s| grows "
                     "(magnitude-driven), independent of the reversal story?\n")
        lines.append("| condition | mean off-mani | mean steer_dist | mean |steered_latent| |")
        lines.append("|---|---|---|---|")
        for l in sweep:
            om = np.nanmean(arr[l]["offmanifold_err"])
            sd = np.nanmean(arr[l]["steer_dist"])
            sl = np.nanmean(np.abs(arr[l]["steered_latent"]))
            print(f"{l:>14}: off-mani={om:.4f}  steer_dist={sd:.4f}  |latent|={sl:.4f}")
            lines.append(f"| {l} | {om:.4f} | {sd:.4f} | {sl:.4f} |")

    # ---- 4. Anchor comparison: zero-ablation vs unsteered vs negative s ----
    anchor_labels = [l for l in ["unsteered(0)", "zero-ablate", "suppress(-10)"] if l in present]
    if len(anchor_labels) >= 2:
        print("\n=== Anchor comparison (off-manifold error) ===")
        lines.append("\n## Anchor comparison — does weirdness start at zero, or only past it?\n")
        lines.append("`unsteered(0)` leaves the latent natural; `zero-ablate` sets it to 0 "
                     "(removes the feature's contribution, sits at the JumpReLU boundary); "
                     "`suppress(-10)` pushes it negative. If off-manifold error is already high at "
                     "zero-ablation, the effect is removal, not negativity. If it only rises once "
                     "past zero into negative s, that is the sign-specific signature.\n")
        lines.append("| condition | mean off-mani | mean steer_dist |")
        lines.append("|---|---|---|")
        for l in anchor_labels:
            om = np.nanmean(arr[l]["offmanifold_err"])
            sd = np.nanmean(arr[l]["steer_dist"])
            print(f"{l:>14}: off-mani={om:.4f}  steer_dist={sd:.4f}")
            lines.append(f"| {l} | {om:.4f} | {sd:.4f} |")

    # ---- 5. Spotlight the 4 spot-check candidates ----
    cand_present = sorted(CANDIDATES & set(keys))
    if cand_present:
        print("\n=== Spot-check candidates: off-manifold error vs layer population ===")
        lines.append("\n## Spot-check candidates vs their layer population\n")
        lines.append("Is each candidate's off-manifold error an outlier within its layer, or "
                     "ordinary? This quantitatively firms up Step 4's manual coherence call on "
                     "`14_6669` (whose output score rose monotonically with |s| as generation "
                     "degraded).\n")
        ref = sup_label or (sweep[-1] if sweep else None)
        if ref:
            lines.append(f"Reference condition: **{ref}**. Percentile = fraction of that layer's "
                         f"features with lower off-manifold error than the candidate.\n")
            lines.append("| feature | off-mani | layer mean | layer pctile | steered_latent |")
            lines.append("|---|---|---|---|---|")
            om_all = arr[ref]["offmanifold_err"]
            sl_all = arr[ref]["steered_latent"]
            for k in cand_present:
                i = keys.index(k)
                lyr = layer_of(k)
                mask = layers == lyr
                pct = float(np.nanmean(om_all[mask] < om_all[i]))
                print(f"  {k:>10}: off-mani={om_all[i]:.4f}  layer_mean={np.nanmean(om_all[mask]):.4f}  "
                      f"pctile={pct:.1%}  latent={sl_all[i]:.4f}")
                lines.append(f"| {k} | {om_all[i]:.4f} | {np.nanmean(om_all[mask]):.4f} | "
                             f"{pct:.1%} | {sl_all[i]:.4f} |")

    lines.append("\n## Summary interpretation\n")
    lines.append("_Fill in from the tables above once the caches are produced: state per-layer "
                 "whether suppression's off-manifold error tracks amplification (generic artifact) "
                 "or exceeds it (sign-specific), whether the zero-ablation anchor already shows the "
                 "effect, and whether the decision gate for the 'true suppression' extension opens._\n")

    out = f"{CACHE_DIR}/step3_reconstruction_error.md"
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
