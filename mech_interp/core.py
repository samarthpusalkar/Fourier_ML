"""
mech_interp/core.py — Hooks, analysis, and folding measures
=============================================================
Minimal self-contained module. No heavy frameworks needed beyond torch/numpy.
"""
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict, defaultdict


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------

class ActivationExtractor:
    def __init__(self, capture_input=False, detach=True):
        self.capture_input = capture_input
        self.detach = detach
        self._handles = []
        self._activations = OrderedDict()

    def _make_hook(self, name):
        def hook(module, input, output):
            entry = {}
            if isinstance(output, tuple):
                entry["output"] = output[0].detach() if self.detach else output[0]
                entry["output_full"] = output if not self.detach else None
            else:
                entry["output"] = output.detach() if self.detach else output
            if self.capture_input and input:
                inp = input[0]
                entry["input"] = inp.detach() if self.detach else inp
            self._activations[name] = entry
        return hook

    def register_hooks(self, model, layer_names=None):
        self._handles = []
        self._activations = OrderedDict()
        if layer_names is None:
            layer_names = []
            for nm, mod in model.named_modules():
                if nm and len(list(mod.children())) == 0:
                    layer_names.append(nm)
        registry = dict(model.named_modules())
        for nm in layer_names:
            mod = registry.get(nm)
            if mod is None:
                continue
            self._handles.append(mod.register_forward_hook(self._make_hook(nm)))
        return self

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.remove_hooks()

    def get_activations(self):
        return self._activations


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

class CoefficientTracker:
    def __init__(self):
        self.history = defaultdict(list)

    def capture(self, model, epoch_or_step=0, tag=""):
        model.eval()
        with torch.no_grad():
            idx = 0
            for name, mod in model.named_modules():
                if hasattr(mod, 'gain_real') and hasattr(mod, 'gain_imag'):
                    real = mod.gain_real.detach().cpu().numpy()
                    imag = mod.gain_imag.detach().cpu().numpy()
                    mag = np.sqrt(real**2 + imag**2)
                    self.history[f"gain_{idx}_mean"].append(float(mag.mean()))
                    self.history[f"gain_{idx}_std"].append(float(mag.std()))
                    idx += 1
            if hasattr(model, 'fourier_head'):
                fh = model.fourier_head
                if hasattr(fh, 'coeff_proj'):
                    W = fh.coeff_proj.weight.detach().cpu().numpy()
                    self.history["coeff_proj_norm"].append(float(np.linalg.norm(W)))
                if hasattr(fh, 'scale'):
                    self.history["fourier_scale"].append(float(fh.scale.item()))
        self.history["epoch_or_step"].append(epoch_or_step)
        if tag:
            self.history["tag"].append(tag)

    def get_history(self):
        return dict(self.history)


def analyze_frequency_gains(model):
    info = {}
    idx = 0
    for name, mod in model.named_modules():
        if hasattr(mod, 'gain_real') and hasattr(mod, 'gain_imag'):
            real = mod.gain_real.detach().cpu().numpy()
            imag = mod.gain_imag.detach().cpu().numpy()
            mag = np.sqrt(real**2 + imag**2)
            phase = np.arctan2(imag, real)
            info[idx] = {
                "name": name,
                "gain_magnitude": mag,
                "gain_phase": phase,
                "fft_shape": getattr(mod, 'fft_shape', mag.shape[1:]),
                "channels": getattr(mod, 'channels', mag.shape[0]),
            }
            idx += 1
    return info


def decompose_coefficients(model, sample_batch=None, input_shape=None, device="cpu"):
    if sample_batch is None:
        if hasattr(model, 'spatial_shape'):
            input_shape = model.spatial_shape
            c = getattr(model, 'latent_dim', input_shape[0] if len(input_shape) > 0 else 3)
            sample_batch = torch.zeros(4, c, *input_shape, device=device)
        else:
            # fallback: try to infer from first param
            first_param = next(model.parameters())
            sample_batch = torch.zeros(4, *first_param.shape[1:], device=device)
    else:
        sample_batch = sample_batch.to(device)
    model.eval()
    # Architecture-aware extraction
    if hasattr(model, 'get_coefficient_embedding'):
        coeffs, freqs = model.get_coefficient_embedding(sample_batch)
        coeffs = coeffs.detach().cpu().numpy()
        freqs = freqs.detach().cpu().numpy() if torch.is_tensor(freqs) else freqs
        num_modes = (coeffs.shape[1] - 1) // 2
        return {
            "coefficients": coeffs,
            "a0": coeffs[:, 0:1],
            "a_n": coeffs[:, 1:1 + num_modes],
            "b_n": coeffs[:, 1 + num_modes:],
            "frequencies": freqs,
            "num_modes": num_modes,
        }
    elif hasattr(model, 'fourier'):
        # SpectralV3 path: run forward up to fourier layer manually
        with torch.no_grad():
            x = sample_batch
            if model.input_type == "image":
                B, C, H, W = x.shape
                ph, pw = model.patch_size
                x = x.unfold(2, ph, ph).unfold(3, pw, pw)
                x = x.contiguous().view(B, C, H // ph, W // pw, ph, pw)
                x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
                x = x.view(B, -1, C * ph * pw)
                z = model.encoder(x)
                for mixer in model.mixers:
                    z = mixer(z)
                z = z.mean(dim=1)
            else:
                z = model.encoder(x)
            fvec = model.fourier(z)
        fvec = fvec.detach().cpu().numpy()
        num_modes = model.num_modes
        # fvec shape: (B, 2*(num_modes+1))
        a0 = fvec[:, 0:1]
        cos_block = fvec[:, :num_modes + 1]
        sin_block = fvec[:, num_modes + 1:]
        return {
            "coefficients": fvec,
            "a0": a0,
            "a_n": cos_block[:, 1:],
            "b_n": sin_block[:, 1:],
            "frequencies": None,
            "num_modes": num_modes,
        }
    else:
        raise ValueError("Model has neither get_coefficient_embedding nor fourier layer")


def compute_manifold_structure(activations_dict, labels_per_sample=None):
    stats = {}
    for name, tensor_list in activations_dict.items():
        mats = []
        for t in tensor_list:
            mats.append(t.reshape(t.shape[0], -1).detach().cpu().numpy())
        if not mats:
            continue
        X = np.concatenate(mats, axis=0)
        if labels_per_sample is None:
            stats[name] = {"mean_norm": float(np.linalg.norm(X, axis=1).mean()), "variance": float(X.var())}
            continue
        classes = np.unique(labels_per_sample)
        if len(classes) < 2:
            stats[name] = {"warning": "need >= 2 classes"}
            continue
        cents = {c: X[labels_per_sample == c].mean(axis=0) for c in classes}
        intra = []
        for c in classes:
            cls = X[labels_per_sample == c]
            if len(cls) == 0:
                continue
            intra.append(np.linalg.norm(cls - cents[c], axis=1).mean())
        intra_mean = np.mean(intra)
        inter = []
        for i, ci in enumerate(classes):
            for cj in classes[i + 1:]:
                inter.append(np.linalg.norm(cents[ci] - cents[cj]))
        inter_mean = np.mean(inter)
        stats[name] = {
            "inter_intra_ratio": float(inter_mean / (intra_mean + 1e-12)),
            "intra_mean": float(intra_mean),
            "inter_mean": float(inter_mean),
        }
    return stats


# ---------------------------------------------------------------------------
# Folding
# ---------------------------------------------------------------------------

def compute_fold_measure(latent_trajectory, threshold=1e-6, align_dim=16):
    from sklearn.decomposition import PCA
    results = {"per_layer": [], "cumulative": 0}
    prev = None
    for z in latent_trajectory:
        znp = z.detach().cpu().numpy() if torch.is_tensor(z) else np.asarray(z)
        if znp.size == 0:
            results["per_layer"].append(0.0)
            continue
        z2d = znp.reshape(znp.shape[0], -1)
        if z2d.shape[1] > align_dim:
            z2d = PCA(n_components=align_dim).fit_transform(z2d)
        elif z2d.shape[1] < align_dim:
            pad = np.zeros((z2d.shape[0], align_dim - z2d.shape[1]))
            z2d = np.concatenate([z2d, pad], axis=1)
        diff = z2d if prev is None else z2d - prev
        direction = z2d.mean(axis=0)
        scalar = z2d @ direction
        scalar = scalar - scalar.min()
        bins = np.where(scalar > threshold, 1, 0)
        transitions = np.sum(np.abs(np.diff(bins))) if len(bins) > 1 else 0
        results["per_layer"].append(float(transitions))
        results["cumulative"] += float(transitions)
        prev = z2d
    return results


def compute_spectral_partition_count(model):
    n_per_layer = []
    for mod in model.modules():
        if hasattr(mod, 'activation') and not isinstance(mod.activation, torch.nn.Identity):
            if hasattr(mod, 'channels'):
                n_per_layer.append(mod.channels)
    if not n_per_layer:
        return {"per_layer": [], "total": 1}
    total = 1
    for n in n_per_layer:
        total *= max(1, n)
    return {"per_layer": n_per_layer, "total": total}


def track_latent_trajectory(model, x):
    """
    Capture intermediate outputs of the model for a given input x.
    Uses forward hooks to stay architecture-agnostic.
    Returns a list of flattened output tensors: [proj, mixer0, mixer1, ..., head].
    """
    outputs = []
    handles = []

    def make_hook():
        def hook(module, inp, out):
            if isinstance(out, tuple):
                out = out[0]
            outputs.append(out.reshape(out.shape[0], -1).detach())
        return hook

    # projection layer hook
    proj = getattr(model, 'channel_proj', getattr(model, 'encoder', None))
    if proj is not None:
        handles.append(proj.register_forward_hook(make_hook()))

    # mixer hooks
    mixers = getattr(model, 'mixers', [])
    for m in mixers:
        handles.append(m.register_forward_hook(make_hook()))

    # head hook
    head = getattr(model, 'fourier_head', getattr(model, 'fourier', None))
    if head is not None:
        handles.append(head.register_forward_hook(make_hook()))

    with torch.no_grad():
        model(x)

    for h in handles:
        h.remove()
    return outputs
