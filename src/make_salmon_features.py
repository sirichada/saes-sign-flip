"""
Generate the dead-salmon random-direction control POOL (CLAUDE.md Step 2).

CPU-only, no model/SAE load. For each layer, draw ONE shared pool of n_pool random
SAE dictionary latents (excluding that layer's selected features), from a fixed
logged seed. This is a per-layer *population* compared against the selected
population (unpaired), NOT k controls matched to each selected feature: the
steering bump is feature-agnostic (sae_utils.py:49), so a random control shares no
pair-specific trait with any selected feature and a paired design would be inert
(and ~10x more compute). n_pool defaults to 100 to match the per-layer selected
count.

Writes:
  - a flat scorer-ready features file {layer: [pool indices]}, and
  - a sidecar with seed / n_pool / dict_width / selected_file for provenance.
"""
import argparse
import json
import random
from collections import OrderedDict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selected_file", default="data/features/gemma_2b_features.json")
    ap.add_argument("--out_file", default="data/features/salmon_control_features.json")
    ap.add_argument("--meta_file", default="data/features/salmon_control_features.meta.json")
    ap.add_argument("--n_pool", type=int, default=100)        # per-layer control pool size
    ap.add_argument("--dict_width", type=int, default=16384)  # gemma-scope 16k
    ap.add_argument("--seed", type=int, default=20260714)
    args = ap.parse_args()

    with open(args.selected_file) as f:
        selected = json.load(f)  # {layer_str: [indices]}

    # stdlib random (Mersenne Twister) — deterministic across platforms/CPython
    # versions for a fixed seed, and dependency-free so this data-prep step runs
    # anywhere (incl. the no-GPU authoring box).
    rng = random.Random(args.seed)
    flat = OrderedDict()  # layer_str -> sorted control pool indices (for scorer)

    for layer_str in sorted(selected, key=int):
        sel_set = set(int(x) for x in selected[layer_str])
        # Exclude this layer's selected features so a "random" control is never
        # itself one of the paper's selected features.
        pool_candidates = [i for i in range(args.dict_width) if i not in sel_set]
        pool = rng.sample(pool_candidates, args.n_pool)
        flat[layer_str] = sorted(pool)

    with open(args.out_file, "w") as f:
        json.dump(flat, f)
    n_ctrl = sum(len(v) for v in flat.values())
    with open(args.meta_file, "w") as f:
        json.dump({
            "seed": args.seed, "n_pool": args.n_pool, "dict_width": args.dict_width,
            "selected_file": args.selected_file, "n_controls": n_ctrl,
        }, f)
    print(f"Wrote {args.out_file}: {len(flat)} layers, {n_ctrl} control latents "
          f"({args.n_pool}/layer)")
    print(f"Wrote {args.meta_file}: seed={args.seed}, n_pool={args.n_pool}")


if __name__ == "__main__":
    main()
