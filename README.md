
# Sign-Flip: Does SAE Steering Suppress Symmetrically?

BlackboxNLP 2026 Reproducibility Challenge (Ablation track) submission. Builds on Gemma-2-2B with Gemma Scope SAEs, extending Arad, Mueller & Belinkov's [*"SAEs Are Good for Steering — If You Select the Right Features"*](https://arxiv.org/abs/2505.20063) ([original project page](https://technion-cs-nlp.github.io/saes-are-good-for-steering/)) — the paper and repo this project reproduces and extends.

The original paper's output score measures whether *amplifying* an SAE feature (a positive steering factor `s` in `ã_i = a_i + s·a_max`, applied on a neutral prompt) shifts the model's output distribution toward that feature's logit-lens tokens. This project runs the sign-flip ablation: replacing amplification with *suppression* (negative `s`) to test whether suppression behaves asymmetrically relative to amplification, motivated by an analogous asymmetry reported in maze-transformers (Spies et al., [arXiv:2412.11867](https://arxiv.org/abs/2412.11867)).

**Finding:** the apparent suppression asymmetry is best explained as an off-distribution decoder artifact, not a genuine sign-specific effect. A "dead salmon" control shows random SAE features collapse off-manifold under steering just as much as the paper's selected max-activating features (0/26 layers survive FDR correction), and a coherence spot-check of generated text shows fluency tracking steering magnitude `|s|`, not its sign — degenerate output appears at large magnitudes in *both* directions. A follow-up robustness check (blind degeneracy metrics + a steering-free interpretability cross-check) corroborates this and narrows it further: one candidate feature (`14_6669`) is the best-supported single case of an artifact-driven reversal, while another (`7_6944`) is genuinely interpretable yet still shows the same extreme-magnitude score fragility — illustrating that interpretability and steering-effectiveness are separate questions. See the Method & Findings section below for links to the full evidence.

## Requirements
```
  pip install -r requirements
  pip install accelerate
  pip install sae-lens
```

## Code
Output Scores

```
python ./src/output_score.py --model_type=<model_type> --features_file=<features_json> --cache_path=<filename_to_load_and_save>
```

This project adds three flags to `output_score.py` for the sign-flip ablation: `--amp_factor` (sets `s`, negative for suppression), `--intervention {amplify,zero}` (adds a zero-ablation anchor, `ã_i=0`, distinct from any negative `s`), and `--log_recon_error` (logs per-feature reconstruction-error/off-manifold diagnostics alongside the score).

Input Scores

First, download feature data from [Neuronpedia](https://www.neuronpedia.org/api-doc#tag/features/POST/api/activation/new). 
Then run:

```
python ./src/input_score.py --model_type=<model_type> --features_file=<features_json> --cache_path=<filename_to_load_and_save> --feature_data_path=<path>
```

## Method & Findings

Each step below is a script + cache/report pair; open the linked report for the actual numbers rather than trusting this summary.

- **Signed re-analysis** — per-layer reversal fractions and cross-magnitude rank stability on the raw output-score caches. `src/analyze_suppression_asymmetry.py` → [`reports/signed_reanalysis.md`](reports/signed_reanalysis.md).
- **Dead-salmon control** — tests whether the off-manifold collapse under steering is targeted at selected features or generic to the SAE dictionary, using a random per-layer pool of features steered identically. `src/analyze_salmon_control.py` → [`reports/dead_salmon.md`](reports/dead_salmon.md).
- **Reconstruction-error / off-manifold logging** — per-layer amplify-vs-suppress comparison of decoder reconstruction fidelity, a magnitude sweep, and a zero-ablation anchor to separate "removal" from "pushed past zero." `src/analyze_reconstruction_error.py` → [`reports/reconstruction_error.md`](reports/reconstruction_error.md).
- **Coherence spot-check** — qualitative read of generated text for the strongest candidate reversal features, checking whether high output scores under extreme steering reflect genuine bidirectional control or degenerate/repetitive generation. → [`reports/step4_coherence_spotcheck.md`](reports/step4_coherence_spotcheck.md).
- **Blind degeneracy metrics** — replaces the coherence spot-check's manual read with deterministic repetition-rate and reference-LM perplexity metrics, to rule out reader bias in the coherence verdict. `src/analyze_generation_degeneracy.py` → [`reports/degeneracy_metrics.md`](reports/degeneracy_metrics.md).
- **CE-Bench interpretability cross-check** — per-feature contrastive activation-divergence check on unsteered text (no steering involved at all), testing whether the spot-check candidates track any real semantic contrast outside the steering context. `src/cebench_extract.py` + `src/analyze_cebench_candidates.py` → [`reports/cebench_crosscheck.md`](reports/cebench_crosscheck.md).

## Data
We provide the exact features used in our experiments, the generated texts for each steering factor and feature, the best steering factor for each feature.
For Concept500 features we additionally provide the 10 sampled instructions per feature and the concept, instruct and fluency scores computed by an external LLM.

This project's own cached runs live alongside the original data: per-feature amplify/suppress/zero-ablation score caches are in `data/output_scores/` (e.g. `gemma2_2b_{amplify,suppress,suppress_amp2,suppress_amp20,zero_ablation,unsteered}.json`), and the corresponding generated text used for the coherence spot-check is in `data/generated_texts/gemma_2b/`. Blind degeneracy metrics are in `data/output_scores/degeneracy_metrics.json`; CE-Bench per-feature activation caches are in `data/cebench/`, with the resulting per-candidate scores in `data/output_scores/cebench_metrics.json`.

## AI Assistance

Claude Code was used throughout this project's extension work: implementing the sign-flip and diagnostic code paths in `src/`, running the analysis scripts, and drafting the written analysis and initial read of results in `reports/`. This README itself was also drafted by Claude Code. See [`AI_ASSISTANCE.md`](AI_ASSISTANCE.md) for the full disclosure.
