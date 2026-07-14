"""
Dead-salmon control analysis (CLAUDE.md Step 2). CPU-only, cached JSON, no GPU.

Direction-specificity test (complement to Step 3's sign-specificity): do the
paper's SELECTED max-activating features fall further off-manifold under steering
than a random per-layer control POOL of SAE dictionary latents, or is the collapse
generic (a dead salmon)? Per sign s in {-10,+10} and per layer, compare the
selected population against the control pool with an UNPAIRED Mann-Whitney U test
(the controls are random, not matched to individual features, so a paired test
would be inert); BH-FDR across the 26 layers. Primary metric offmanifold_err;
secondary metric the output score.

Baseline note: at the unsteered anchor (s=0) no latent is modified, so
offmanifold_err there is a per-LAYER constant identical for every feature. It
shifts both populations equally and cannot change the selected-vs-control contrast;
we load it only to report the per-layer baseline for reference.
"""
import json
from collections import defaultdict

import numpy as np
from scipy.stats import mannwhitneyu

CACHE_DIR = "data/output_scores"
FEATURE_DIR = "data/features"
CANDIDATES = {"2_13823", "7_12166", "7_6944", "14_6669"}

# sign -> (selected diag cache basename, salmon-pool diag cache basename)
CONDITIONS = {
    "-10": ("gemma2_2b_suppress_diag", "gemma2_2b_salmon_neg10_diag"),
    "+10": ("gemma2_2b_amplify_diag", "gemma2_2b_salmon_10_diag"),
}


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def layer_of(key):
    return int(key.split("_")[0])


def bh_fdr(pvals):
    """Benjamini-Hochberg FDR-adjusted p-values (nan-safe)."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    idx = np.where(~np.isnan(p))[0]
    m = len(idx)
    if m == 0:
        return out
    order = idx[np.argsort(p[idx])]
    adj = p[order] * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out[order] = np.clip(adj, 0, 1)
    return out


def field_dict(diag, field):
    return {k: v.get(field, np.nan) for k, v in diag.items()}


def by_layer(values):
    """{"{layer}_{idx}": scalar} -> {layer: [non-nan scalars]}."""
    out = defaultdict(list)
    for k, v in values.items():
        if v == v:  # drop NaN
            out[layer_of(k)].append(v)
    return out


def unpaired_layer_test(sel_vals, ctrl_vals):
    """Per layer: Mann-Whitney U (two-sided) of the selected population vs the
    control pool. Returns (rows, pvals), rows = [layer, n_sel, n_ctrl, sel_mean,
    ctrl_mean, sel-ctrl]."""
    sel_by, ctrl_by = by_layer(sel_vals), by_layer(ctrl_vals)
    rows, pvals = [], []
    for lyr in sorted(set(sel_by) | set(ctrl_by)):
        s = np.array(sel_by.get(lyr, []), float)
        c = np.array(ctrl_by.get(lyr, []), float)
        if len(s) and len(c):
            try:
                _, p = mannwhitneyu(s, c, alternative="two-sided")
            except ValueError:
                p = np.nan
        else:
            p = np.nan
        sm = float(np.mean(s)) if len(s) else np.nan
        cm = float(np.mean(c)) if len(c) else np.nan
        rows.append([lyr, len(s), len(c), sm, cm, sm - cm])
        pvals.append(p)
    return rows, pvals


def emit_metric(lines, title, blurb, rows, pvals, baseline_by_layer=None):
    p_fdr = bh_fdr(pvals)
    lines.append(f"\n{title}\n")
    lines.append(blurb + "\n")
    hdr = "| layer | n_sel | n_ctrl | sel | ctrl | sel-ctrl | mwu p | p_fdr |"
    sep = "|---|---|---|---|---|---|---|---|"
    if baseline_by_layer is not None:
        hdr += " unsteered base |"
        sep += "---|"
    lines.append(hdr)
    lines.append(sep)
    for (lyr, ns, nc, sm, cm, diff), p, pf in zip(rows, pvals, p_fdr):
        p_s = f"{p:.2e}" if p == p else "nan"
        row = f"| {lyr} | {ns} | {nc} | {sm:.4f} | {cm:.4f} | {diff:+.4f} | {p_s} | {pf:.2e} |"
        if baseline_by_layer is not None:
            b = baseline_by_layer.get(lyr, np.nan)
            row += f" {b:.4f} |" if b == b else " n/a |"
        lines.append(row)
    n_sig = int(np.nansum(np.asarray(p_fdr) < 0.05))
    n_pos = sum(1 for r in rows if r[5] == r[5] and r[5] > 0)
    lines.append(f"\n**{n_sig}/{len(rows)} layers FDR<0.05; sel>ctrl in {n_pos}/{len(rows)} layers.**\n")
    return n_sig, n_pos


def main():
    meta = load_json(f"{FEATURE_DIR}/salmon_control_features.meta.json")
    if meta is None:
        print("Missing salmon_control_features.meta.json — run make_salmon_features.py first.")
        return

    unsteered = load_json(f"{CACHE_DIR}/gemma2_2b_unsteered.json")
    baseline_by_layer = {}
    if unsteered and unsteered.get("diagnostics"):
        for key, d in unsteered["diagnostics"].items():
            baseline_by_layer.setdefault(layer_of(key), d.get("offmanifold_err", np.nan))

    norms = load_json(f"{FEATURE_DIR}/decoder_norms.json")
    row_norms = norms["row_norms"] if norms else {}

    lines = ["# Step 2 — Dead-Salmon Random-Direction Control\n"]
    lines.append("CLAUDE.md Step 2. Tests DIRECTION-specificity of the suppression off-manifold "
                 "effect (complement to Step 3's sign-specificity): do the paper's SELECTED "
                 "max-activating features fall further off-manifold than a random per-layer POOL "
                 "of SAE dictionary latents steered identically, or is the collapse generic (a "
                 "dead salmon)?\n")
    lines.append(f"Control = a shared pool of n_pool={meta['n_pool']} random dictionary latents "
                 f"per layer (seed {meta['seed']}). The bump is feature-agnostic, so controls are "
                 "not matched to individual selected features; the test is therefore an UNPAIRED "
                 "per-layer Mann-Whitney U (selected population vs. control pool), p_fdr = "
                 "Benjamini-Hochberg across the 26 layers. Primary metric `offmanifold_err`; "
                 "output score reported secondarily.\n")

    for sign, (sel_name, sal_name) in CONDITIONS.items():
        sel_cache = load_json(f"{CACHE_DIR}/{sel_name}.json")
        sal_cache = load_json(f"{CACHE_DIR}/{sal_name}.json")
        if not (sel_cache and sal_cache and sel_cache.get("diagnostics") and sal_cache.get("diagnostics")):
            lines.append(f"\n## s = {sign}: MISSING caches ({sel_name} / {sal_name}) — skipped\n")
            print(f"skip s={sign}: missing caches")
            continue
        sd, ld = sel_cache["diagnostics"], sal_cache["diagnostics"]

        # Primary: offmanifold_err
        rows, pv = unpaired_layer_test(field_dict(sd, "offmanifold_err"),
                                       field_dict(ld, "offmanifold_err"))
        emit_metric(lines, f"## s = {sign} — off-manifold error (primary)",
                    "Positive `sel-ctrl` = selected features fall FURTHER off-manifold than the "
                    "random control pool (direction-specific). ~0 = dead salmon (generic).",
                    rows, pv, baseline_by_layer)

        # Secondary: output score
        rows2, pv2 = unpaired_layer_test(sel_cache["scores"], sal_cache["scores"])
        emit_metric(lines, f"## s = {sign} — output score (secondary)",
                    "Same unpaired per-layer comparison on the (signed) output score. Logit-lens "
                    "token sets always exist for Gemma, so no feature is dropped; sign "
                    "interpretation on a neutral prompt is murky, hence secondary.",
                    rows2, pv2)

        # Candidate spotlight: each candidate vs its layer's control pool
        lines.append(f"\n### Candidate spotlight (s = {sign}, off-manifold)\n")
        lines.append("Percentile = fraction of the layer's control pool with LOWER off-manifold "
                     "error than the candidate (near 50% = ordinary, i.e. fails specificity).\n")
        lines.append("| candidate | sel off-mani | pool mean | pool max | pctile vs pool |")
        lines.append("|---|---|---|---|---|")
        pool_off = by_layer(field_dict(ld, "offmanifold_err"))
        for cand in sorted(CANDIDATES):
            if cand not in sd:
                continue
            lyr = layer_of(cand)
            sel_off = sd[cand].get("offmanifold_err", np.nan)
            pool = np.array(pool_off.get(lyr, []), float)
            pct = float(np.nanmean(pool < sel_off)) if len(pool) else np.nan
            pm = np.nanmean(pool) if len(pool) else np.nan
            px = np.nanmax(pool) if len(pool) else np.nan
            lines.append(f"| {cand} | {sel_off:.4f} | {pm:.4f} | {px:.4f} | {pct:.0%} |")

    # Decoder-norm covariate
    lines.append("\n## Decoder-norm covariate\n")
    if row_norms:
        lines.append("The only residual magnitude confound is ‖W_dec[feature]‖ (the bump is "
                     "otherwise feature-agnostic). If the selected and control-pool decoder-norm "
                     "distributions match per layer, the contrast above is not a norm artifact.\n")
        sel_n, ctrl_n = defaultdict(list), defaultdict(list)
        selected = load_json(f"{FEATURE_DIR}/{meta['selected_file'].split('/')[-1]}") or {}
        controls = load_json(f"{FEATURE_DIR}/salmon_control_features.json") or {}
        for lyr_str, idxs in selected.items():
            for i in idxs:
                key = f"{lyr_str}_{int(i)}"
                if key in row_norms:
                    sel_n[int(lyr_str)].append(row_norms[key])
        for lyr_str, idxs in controls.items():
            for i in idxs:
                key = f"{lyr_str}_{int(i)}"
                if key in row_norms:
                    ctrl_n[int(lyr_str)].append(row_norms[key])
        lines.append("| layer | mean sel ‖W_dec‖ | mean ctrl ‖W_dec‖ | ratio |")
        lines.append("|---|---|---|---|")
        for lyr in sorted(set(sel_n) | set(ctrl_n)):
            s = float(np.mean(sel_n[lyr])) if sel_n[lyr] else np.nan
            c = float(np.mean(ctrl_n[lyr])) if ctrl_n[lyr] else np.nan
            ratio = s / c if (c == c and c) else np.nan
            lines.append(f"| {lyr} | {s:.4f} | {c:.4f} | {ratio:.3f} |")
    else:
        lines.append("_decoder_norms.json not found — run check_decoder_norms.py on the GPU box._\n")

    lines.append("\n## Summary interpretation\n_Fill in: per sign, in how many layers do selected "
                 "features fall significantly further off-manifold than the random control pool "
                 "(direction-specific) vs. indistinguishable (dead salmon)? Does any survival "
                 "concentrate in layers ~17-23 (Step 3's band)? Do the 4 candidates sit inside "
                 "their layer's control distribution (pctile near 50%) — failing the specificity "
                 "test — consistent with the Step 3/Step 4 artifact reading?_\n")

    out = f"{CACHE_DIR}/step2_dead_salmon.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
