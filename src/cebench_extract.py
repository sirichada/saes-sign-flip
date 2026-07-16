"""
Step 6 (Track C) — GPU extraction pass for the CE-Bench per-feature cross-check
(docs/methodology.md, Track C, Step 6).

Applies CE-Bench's contrastive methodology (Gulko, Peng & Kumar, arXiv:2509.00691;
code: github.com/Yusen-Peng/CE-Bench, ce_bench/CE_Bench_v2.py) to this project's
4 spot-check candidates. This pass only extracts and caches raw mean-activation
vectors — NO steering anywhere, and no scoring (that's src/analyze_cebench_candidates.py,
CPU-only, so the scoring math can be iterated without re-paying the forward passes).

Per CE_Bench_v2.py (their marked-token filter is FIXME'd to average over ALL tokens):
  V1 = mean over story1 tokens of sae.encode(resid_post_layer)   (16384-dim)
  V2 = same for story2
  I1 (needed by the independence score) = token-weighted mean over BOTH stories'
  tokens = (n1*V1 + n2*V2) / (n1+n2), so we store V1, V2, n1, n2 and reconstruct
  I1 exactly downstream. Note this matches their CODE, not the paper's "I1 = V1+V2".

Amortization: one forward pass per story with capture hooks at all requested layers
simultaneously (CE-Bench runs its whole pipeline once per SAE; here 3 SAEs share
each forward). Dataset: GulkoA/contrastive-stories-v4 (5000 pairs, 1000 subjects).

Caches (per layer, resumable): data/cebench/V1_layer{L}.npy, V2_layer{L}.npy
(float32 memmaps, shape (n_pairs, d_sae)) + a shared progress/meta JSON.
Run on the GPU box; a --limit 20 smoke run first is recommended.
"""
import argparse
import json
import os
from datetime import date

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import get_gemma2_sae

CACHE_DIR = "data/cebench"
DATASET = "GulkoA/contrastive-stories-v4"
MODEL_NAME = "google/gemma-2-2b"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layers", type=str, default="2,7,14",
                   help="comma-separated residual layers (default: the candidates' layers)")
    p.add_argument("--limit", type=int, default=None,
                   help="only process the first N pairs (smoke test)")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--save_every", type=int, default=50, help="flush progress every N pairs")
    return p.parse_args()


def main():
    args = parse_args()
    layers = [int(x) for x in args.layers.split(",")]
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(CACHE_DIR, exist_ok=True)

    dataset = load_dataset(DATASET, split="train")
    n_pairs = len(dataset) if args.limit is None else min(args.limit, len(dataset))

    saes = {}
    saes = {lyr: get_gemma2_sae("2b", False, lyr, saes).to(device) for lyr in layers}
    d_sae = {lyr: saes[lyr].cfg.d_sae for lyr in layers}

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32).to(device).eval()

    # capture hooks: residual stream after each block (= gemma-scope res hook point,
    # same hook site as AmlifySAEHook in src/sae_utils.py)
    captured = {}

    def make_hook(lyr):
        def hook(module, inputs, output):
            captured[lyr] = output[0]
        return hook

    handles = [model.model.layers[lyr].register_forward_hook(make_hook(lyr)) for lyr in layers]

    # resumable memmap caches + progress file
    prog_path = os.path.join(CACHE_DIR, "progress.json")
    if os.path.exists(prog_path):
        with open(prog_path) as f:
            prog = json.load(f)
        assert prog["layers"] == layers and prog["n_pairs"] == n_pairs, \
            "existing cache was created with different --layers/--limit; move or delete data/cebench first"
    else:
        prog = {"next": 0, "layers": layers, "n_pairs": n_pairs, "n1": [], "n2": [],
                "subjects": [],
                "meta": {"dataset": DATASET, "model": MODEL_NAME, "dtype": "float32",
                         "sae_release": "gemma-scope-2b-pt-res-canonical",
                         "sae_ids": {str(l): f"layer_{l}/width_16k/canonical" for l in layers},
                         "token_average": "all tokens (per CE_Bench_v2.py FIXME'd marked-token filter)",
                         "date_started": str(date.today())}}

    mmaps = {}
    for lyr in layers:
        for name in ("V1", "V2"):
            path = os.path.join(CACHE_DIR, f"{name}_layer{lyr}.npy")
            mode = "r+" if os.path.exists(path) else "w+"
            mmaps[(name, lyr)] = np.lib.format.open_memmap(
                path, mode=mode, dtype=np.float32, shape=(n_pairs, d_sae[lyr]))

    def flush():
        for m in mmaps.values():
            m.flush()
        with open(prog_path, "w") as f:
            json.dump(prog, f)

    def story_vector(text):
        """One forward pass; returns {layer: mean-over-tokens encoded vector} and n_tokens."""
        ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            model(ids)
            out = {lyr: saes[lyr].encode(captured[lyr]).mean(dim=1).squeeze(0).float().cpu().numpy()
                   for lyr in layers}
        return out, ids.shape[1]

    from tqdm import tqdm
    for i in tqdm(range(prog["next"], n_pairs)):
        row = dataset[i]
        v1, n1 = story_vector(row["story1"])
        v2, n2 = story_vector(row["story2"])
        for lyr in layers:
            mmaps[("V1", lyr)][i] = v1[lyr]
            mmaps[("V2", lyr)][i] = v2[lyr]
        prog["n1"].append(n1)
        prog["n2"].append(n2)
        prog["subjects"].append(row["subject_title"])
        prog["next"] = i + 1
        if (i + 1) % args.save_every == 0:
            flush()

    prog["meta"]["date_finished"] = str(date.today())
    flush()
    print(f"done: {prog['next']}/{n_pairs} pairs cached in {CACHE_DIR} for layers {layers}")


if __name__ == "__main__":
    main()
