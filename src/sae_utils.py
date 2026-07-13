import torch
from torch import nn
from contextlib import contextmanager
from accelerate.hooks import ModelHook, add_hook_to_module
from sae_lens import SAE

@contextmanager
def _disable_hooks(sae: SAE):
    """
    Temporarily disable hooks for the SAE. Swaps out all the hooks with a fake modules that does nothing.
    """
    try:
        for hook_name in sae.hook_dict:
            _blank_hook = nn.Identity()
            setattr(sae, hook_name, _blank_hook)
        yield
    finally:
        for hook_name, hook in sae.hook_dict.items():
            setattr(sae, hook_name, hook)


class AmlifySAEHook(ModelHook):
    def __init__(self, layer, sae, features, amp_factor, device, mode="amplify", log_diag=False) -> None:
        super().__init__()
        self.amp_factor = amp_factor
        self.sae = sae
        self.device = device
        self.layer = layer
        self.features = features
        # "amplify": additive steering ã_i = a_i + s·a_max (amp_factor=0 → unsteered baseline).
        # "zero": zero-ablation ã_i = 0 (remove the feature's contribution; distinct from amp_factor=0).
        self.mode = mode
        self.log_diag = log_diag
        self.diag = None

    def __call__(self, module, args, output):
        output_tensor = output[0]
        _, n_tokens, _ = output_tensor.shape

        # encode with SAE
        feature_acts = self.sae.encode(output_tensor).to(self.device)

        with torch.no_grad():
            with _disable_hooks(self.sae):
                feature_acts_clean = self.sae.encode(output_tensor)
                x_reconstruct_clean = self.sae.decode(feature_acts_clean)
            sae_error = self.sae.hook_sae_error(output_tensor.to(torch.float64) - x_reconstruct_clean.to(torch.float64))

        max_act_value = torch.max(feature_acts[:, -1, :]).item()
        for feature in self.features:
            if self.mode == "zero":
                feature_acts[:, -1, feature] = 0.0
            else:
                feature_acts[:, -1, feature] += max_act_value * self.amp_factor

        sae_out = self.sae.decode(feature_acts)
        sae_out = sae_out + sae_error
        sae_out = sae_out.to(torch.float32)

        if self.log_diag:
            self._log_diagnostics(
                output_tensor, x_reconstruct_clean, sae_out, feature_acts, max_act_value
            )

        return tuple([sae_out] + list(output[1:]))

    def _log_diagnostics(self, output_tensor, x_reconstruct_clean, sae_out, feature_acts, max_act_value):
        """Record reconstruction-error / activation-distance scalars at the last token (float64)."""
        with torch.no_grad():
            a_i = output_tensor[:, -1, :].to(torch.float64)
            recon_clean = x_reconstruct_clean[:, -1, :].to(torch.float64)
            steered = sae_out[:, -1, :].to(torch.float64)

            # How far the steered activation lands off the SAE manifold: re-encode/decode it.
            with _disable_hooks(self.sae):
                re_encoded = self.sae.encode(sae_out)
                re_reconstruct = self.sae.decode(re_encoded)
            offmanifold = (sae_out[:, -1, :].to(torch.float64) - re_reconstruct[:, -1, :].to(torch.float64))

            self.diag = {
                "clean_recon_err": torch.linalg.vector_norm(a_i - recon_clean).item(),
                "steer_dist": torch.linalg.vector_norm(steered - a_i).item(),
                "offmanifold_err": torch.linalg.vector_norm(offmanifold).item(),
                "steered_latent": feature_acts[:, -1, self.features[0]].item(),
                "max_act_value": float(max_act_value),
            }


def init_hook(pipeline, sae, layer, feature, device, args):
    sae_hook = AmlifySAEHook(layer, sae, [feature], args.amp_factor, device)
    model_block_to_hook = pipeline.model.model.layers[layer]
    handle = model_block_to_hook.register_forward_hook(sae_hook, always_call=True)
    return handle
