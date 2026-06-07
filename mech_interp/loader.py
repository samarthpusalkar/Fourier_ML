"""
mech_interp/loader.py — Architecture-agnostic spectral model loader
======================================================================
Detects whether a checkpoint is SpectralModel (spectral_core.py) or
SpectralV3 (train_spectral_v3.py) and instantiates the correct class.
For unknown architectures, falls back to generic SpectralModel with
manual overrides and strict=False.
"""
import os, sys, math
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spectral_core import SpectralModel


def _try_import_spectralv3():
    try:
        from train_spectral_v3 import SpectralV3
        return SpectralV3
    except Exception:
        return None


def _infer_v3_config(state):
    """Infer SpectralV3 constructor args from checkpoint tensors."""
    latent_dim, patch_dim = state["encoder.weight"].shape
    num_modes = int(state["fourier.harmonic_n"].shape[0]) - 1
    num_classes = int(state["classifier.2.weight"].shape[0])
    head_hidden = int(state["classifier.0.weight"].shape[0])

    mixer_ids = set()
    for k in state.keys():
        if k.startswith("mixers."):
            mixer_ids.add(int(k.split(".")[1]))
    num_mixers = max(mixer_ids) + 1 if mixer_ids else 0

    n_rfft = int(state["mixers.0.gain_real"].shape[1])

    input_type = "flat"
    img_size = None
    patch_size = None
    in_ch = None

    if patch_dim > 1:
        for H, W in ((28, 28), (32, 32), (64, 64), (96, 96), (128, 128), (224, 224),
                     (28, 32), (32, 28), (64, 32), (32, 64)):
            for c in (1, 3, 4):
                if patch_dim % c != 0:
                    continue
                parea = patch_dim // c
                for ph in range(1, int(math.isqrt(parea)) + 1):
                    if parea % ph != 0:
                        continue
                    pw = parea // ph
                    if H % ph != 0 or W % pw != 0:
                        continue
                    num_patches = (H // ph) * (W // pw)
                    if num_patches // 2 + 1 == n_rfft:
                        input_type = "image"
                        img_size = (H, W)
                        patch_size = (ph, pw)
                        in_ch = c
                        break
                if input_type == "image":
                    break
            if input_type == "image":
                break

    if input_type == "flat":
        img_size = None
        patch_size = None
        in_ch = None

    return {
        "input_type": input_type,
        "img_size": img_size,
        "patch_size": patch_size,
        "in_ch": in_ch,
        "latent_dim": latent_dim,
        "num_modes": num_modes,
        "num_mixer_layers": num_mixers,
        "num_classes": num_classes,
        "head_hidden": head_hidden,
    }


def _infer_spectral_model_config(state):
    """Infer SpectralModel constructor args from checkpoint tensors."""
    cp_shape = state["channel_proj.op.weight"].shape
    latent_dim = cp_shape[0]
    in_channels = cp_shape[1]
    rank = len(cp_shape) - 2

    mixer0_gain = state["mixers.0.gain_real"]
    fft_shape = mixer0_gain.shape[1:]
    if rank == 1:
        spatial_shape = ((fft_shape[0] - 1) * 2,)
    elif rank == 2:
        spatial_shape = (fft_shape[0], (fft_shape[1] - 1) * 2)
    elif rank == 3:
        spatial_shape = (fft_shape[0], fft_shape[1], (fft_shape[2] - 1) * 2)
    else:
        spatial_shape = tuple((s - 1) * 2 for s in fft_shape)

    mixer_ids = set()
    for k in state.keys():
        if k.startswith("mixers."):
            mixer_ids.add(int(k.split(".")[1]))
    num_mixers = max(mixer_ids) + 1 if mixer_ids else 0

    num_classes = int(state["classifier.0.weight"].shape[0])
    head_hidden = int(state["classifier.0.weight"].shape[1])

    if "fourier_head.coeff_proj.weight" in state:
        head_type = "coefficient"
        num_modes = (int(state["fourier_head.coeff_proj.weight"].shape[0]) - 1) // 2
    elif "fourier_head.proj_weight" in state:
        head_type = "scalar"
        num_modes = int(state["fourier_head.a_n"].shape[0])
    else:
        head_type = "coefficient"
        num_modes = 64

    return {
        "spatial_shape": spatial_shape,
        "in_channels": in_channels,
        "num_classes": num_classes,
        "latent_dim": latent_dim,
        "num_modes": num_modes,
        "num_mixer_layers": num_mixers,
        "head_hidden": head_hidden,
        "head_type": head_type,
    }


def load_checkpoint(weights_path, device="cpu", strict=False, manual_overrides=None):
    """
    Load any spectral model checkpoint and return the instantiated model.
    Args:
        weights_path: path to .pt checkpoint.
        device: torch device.
        strict: passed to load_state_dict.
        manual_overrides: dict of kwargs to force when auto-detection fails.
    Returns:
        (model, metadata_dict)
    """
    if not weights_path or not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")

    raw = torch.load(weights_path, map_location=device, weights_only=False)
    state = raw.get("state_dict", raw.get("model_state_dict", raw))
    if not isinstance(state, dict):
        raise ValueError("Checkpoint does not contain a state_dict")

    keys = list(state.keys())

    # --- Manual override takes precedence ---
    if manual_overrides:
        cfg = manual_overrides
        model = SpectralModel(
            spatial_shape=cfg.get("spatial_shape", (28, 28)),
            in_channels=cfg.get("in_channels", 1),
            num_classes=cfg.get("num_classes", 10),
            latent_dim=cfg.get("latent_dim", 16),
            num_modes=cfg.get("num_modes", 64),
            num_mixer_layers=cfg.get("num_mixer_layers", 2),
            head_hidden=cfg.get("head_hidden", 128),
            head_type=cfg.get("head_type", "coefficient"),
            grid_type="nufft",
        )
        model.load_state_dict(state, strict=False)
        model.to(device)
        model.eval()
        return model, {"arch": "SpectralModel (manual fallback)", **cfg}

    # --- SpectralV3 ---
    if any(k.startswith("encoder.") for k in keys):
        SpectralV3 = _try_import_spectralv3()
        if SpectralV3 is None:
            raise RuntimeError(
                "Checkpoint appears to be SpectralV3 but train_spectral_v3.py is not importable."
            )
        cfg = _infer_v3_config(state)
        model = SpectralV3(
            input_type=cfg["input_type"],
            img_size=cfg["img_size"],
            patch_size=cfg["patch_size"],
            in_ch=cfg["in_ch"],
            latent_dim=cfg["latent_dim"],
            num_modes=cfg["num_modes"],
            num_mixer_layers=cfg["num_mixer_layers"],
            num_classes=cfg["num_classes"],
            init_period=2.0,
            head_hidden=cfg["head_hidden"],
        )
        model.load_state_dict(state, strict=strict)
        model.to(device)
        model.eval()
        return model, {"arch": "SpectralV3", **cfg}

    # --- SpectralModel ---
    if any(k.startswith("channel_proj.") for k in keys):
        cfg = _infer_spectral_model_config(state)
        model = SpectralModel(
            spatial_shape=cfg["spatial_shape"],
            in_channels=cfg["in_channels"],
            num_classes=cfg["num_classes"],
            latent_dim=cfg["latent_dim"],
            num_modes=cfg["num_modes"],
            num_mixer_layers=cfg["num_mixer_layers"],
            head_hidden=cfg["head_hidden"],
            head_type=cfg["head_type"],
            grid_type="nufft",
        )
        model.load_state_dict(state, strict=strict)
        model.to(device)
        model.eval()
        return model, {"arch": "SpectralModel", **cfg}

    raise ValueError(
        f"Unknown checkpoint architecture. Provide --spatial / --channels / --classes overrides.\n"
        f"Keys: {keys[:10]}"
    )
