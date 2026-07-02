import gc
import json
import torch

from collections import namedtuple
from sae_lens import SAE

TopKResult = namedtuple("TopKResult", ["values", "indices"])


def get_gemma2_sae(model_size, instruct, layer, saes, width="16k"):
    if layer in saes:
        sae = saes[layer]
    else:
        release = f"gemma-scope-{model_size}-{'it' if instruct else 'pt'}-res-canonical"
        sae, cfg_dict, sparsity = SAE.from_pretrained(
            release=release,
            sae_id=f"layer_{layer}/width_{width}/canonical",
        )
        print(sae, cfg_dict, sparsity)
        saes[layer] = sae
    return sae


def get_sae(model_type, layer, saes):
    if model_type == "gemma2_2b":
        sae = get_gemma2_sae("2b", False, layer, saes)
    elif model_type == "gemma2_9b":
        sae = get_gemma2_sae("9b", False, layer, saes)
    elif model_type == "gemma2_it":
        sae = get_gemma2_sae("2b", True, layer, saes)
    elif model_type == "gemma2_9b_it":
        sae = get_gemma2_sae("9b", True, layer, saes)
    elif model_type == "gemma2_9b_it_131":
        sae = get_gemma2_sae("9b", True, layer, saes, width="131k")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return sae


def get_features_by_layers(features_file):
    with open(features_file, "r") as f:
        features_by_layer = json.load(f)
        features_by_layer = {int(key): [int(v) for v in values] for key, values in features_by_layer.items()}
        return features_by_layer


def cache_logit_lens(layer, saes, model_type, final_layer_norm, lm_head, k, chunk_size=512):
    sae = get_sae(model_type, layer, saes)
    final_layer_norm = final_layer_norm.cpu()
    lm_head = lm_head.cpu()

    decoder_weights = sae.W_dec.cpu()
    print(decoder_weights.shape)
    decoder_weights = final_layer_norm(decoder_weights)

    # Chunked over features to avoid materializing the full [n_features, vocab_size]
    # logits/softmax matrix (~16.7GB for gemma-2-2b's 16k-width SAEs) at once.
    # Row-wise softmax/topk are independent per feature, so this is numerically
    # identical to the unchunked computation.
    confidence_chunks = []
    topk_values_chunks = []
    topk_indices_chunks = []
    for start in range(0, decoder_weights.shape[0], chunk_size):
        chunk_logits = lm_head(decoder_weights[start:start + chunk_size])
        chunk_confidence = torch.softmax(chunk_logits, dim=1).detach().cpu()
        del chunk_logits
        chunk_topk = torch.topk(chunk_confidence, dim=1, k=k)
        confidence_chunks.append(chunk_confidence)
        topk_values_chunks.append(chunk_topk.values)
        topk_indices_chunks.append(chunk_topk.indices)
        gc.collect()

    confidence = torch.cat(confidence_chunks, dim=0)
    topk = TopKResult(
        values=torch.cat(topk_values_chunks, dim=0),
        indices=torch.cat(topk_indices_chunks, dim=0),
    )

    final_layer_norm = final_layer_norm.cpu()
    lm_head = lm_head.cpu()
    return topk, confidence, None
