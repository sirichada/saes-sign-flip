"""
Decoder-norm check + covariate dump for the dead-salmon control (CLAUDE.md Step 2).

Loads each layer's gemma-scope SAE (reuses src/utils.get_sae) and computes
‖W_dec[feature]‖ for the selected features and their salmon controls. Answers the
one residual confound: is the additive steering bump the same size for selected vs
control (decoder ~norm-homogeneous), or does it differ via decoder norm? Also
reports whether W_dec rows are ~unit-norm dictionary-wide.

Run on the GPU box (SAEs cached there); CPU works too but downloads the SAEs.
Output data/features/decoder_norms.json is consumed (optionally) by
analyze_salmon_control.py as a covariate. Run as `python src/check_decoder_norms.py`.
"""
import argparse
import json

import torch
from utils import get_sae, get_features_by_layers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_type", default="gemma2_2b")
    ap.add_argument("--selected_file", default="data/features/gemma_2b_features.json")
    ap.add_argument("--salmon_file", default="data/features/salmon_control_features.json")
    ap.add_argument("--out_file", default="data/features/decoder_norms.json")
    args = ap.parse_args()

    selected = get_features_by_layers(args.selected_file)   # {int_layer: [int idx]}
    controls = get_features_by_layers(args.salmon_file)
    saes = {}
    norms = {}          # "{layer}_{feature}" -> ‖W_dec[feature]‖
    layer_summary = {}  # str_layer -> dict-wide stats

    for layer in sorted(selected):
        sae = get_sae(args.model_type, layer, saes)
        W = sae.W_dec.detach().cpu().to(torch.float64)      # [n_features, d_model]
        row_norms = torch.linalg.vector_norm(W, dim=1)
        layer_summary[str(layer)] = {
            "mean": float(row_norms.mean()), "std": float(row_norms.std()),
            "min": float(row_norms.min()), "max": float(row_norms.max()),
        }
        idxs = set(selected.get(layer, [])) | set(controls.get(layer, []))
        for i in idxs:
            norms[f"{layer}_{i}"] = float(row_norms[i])
        saes.pop(layer, None)
        s = layer_summary[str(layer)]
        print(f"layer {layer}: dict-wide ‖W_dec‖ mean={s['mean']:.4f} std={s['std']:.4f} "
              f"range=[{s['min']:.4f},{s['max']:.4f}]")

    with open(args.out_file, "w") as f:
        json.dump({"row_norms": norms, "layer_summary": layer_summary}, f)
    means = [v["mean"] for v in layer_summary.values()]
    stds = [v["std"] for v in layer_summary.values()]
    print(f"\nWrote {args.out_file}: {len(norms)} feature norms across {len(layer_summary)} layers")
    print(f"Across layers: mean ‖W_dec‖ in [{min(means):.4f},{max(means):.4f}], max per-layer "
          f"std {max(stds):.4f}. If std<<mean and mean~const => norm-homogeneous, no magnitude "
          f"confound; else use norms as covariate.")


if __name__ == "__main__":
    main()
