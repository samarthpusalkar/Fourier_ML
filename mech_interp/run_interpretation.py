"""
mech_interp/run_interpretation.py — Standalone interpretation dashboard generator
=================================================================================
No Streamlit required. Generates an interactive HTML dashboard from any
SpectralModel or SpectralV3 checkpoint.

Usage:
    python -m mech_interp.run_interpretation --weights best_v3_mnist.pt
    python -m mech_interp.run_interpretation --weights best_cifar10_spectral.pt --output dashboard.html
"""
import os, sys, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mech_interp.loader import load_checkpoint
from mech_interp.core import (
    ActivationExtractor,
    analyze_frequency_gains,
    decompose_coefficients,
    compute_fold_measure,
    compute_spectral_partition_count,
    track_latent_trajectory,
)
from mech_interp.visualizer import (
    visualize_layer_projection,
    plot_frequency_gains,
    build_dashboard,
)

import plotly.graph_objects as go
import traceback


def analyze_raw_state_dict(state):
    """
    Generate figures from a raw state dict when model instantiation fails.
    Works for any spectral architecture variant.
    """
    figures = {}

    # 1) Parameter shape overview
    shapes = []
    names = []
    sizes = []
    for k, v in sorted(state.items()):
        if hasattr(v, 'shape'):
            names.append(k)
            shapes.append(str(tuple(v.shape)))
            sizes.append(v.numel() if hasattr(v, 'numel') else int(np.prod(v.shape)))
    fig_shapes = go.Figure(data=go.Bar(x=names, y=sizes, text=shapes, textposition="auto"))
    fig_shapes.update_layout(title="Parameter Sizes", xaxis_title="parameter", yaxis_title="elements",
                              height=600, xaxis=dict(tickangle=45))
    figures["Parameter Sizes"] = fig_shapes

    # 2) Frequency gains from raw tensors
    gain_layers = {}
    for k, v in state.items():
        if "gain_real" in k:
            layer_id = k.split(".")[1] if k.startswith("mixers.") else "0"
            gain_layers.setdefault(layer_id, {})["real"] = v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)
        elif "gain_imag" in k:
            layer_id = k.split(".")[1] if k.startswith("mixers.") else "0"
            gain_layers.setdefault(layer_id, {})["imag"] = v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)
    for lid, mats in gain_layers.items():
        if "real" in mats:
            mag = np.abs(mats["real"] + 1j * mats.get("imag", np.zeros_like(mats["real"])))
            if mag.ndim == 2:
                fig_gain = go.Figure(data=go.Heatmap(z=mag, colorscale="Cividis",
                    hovertemplate="ch %{y}\u003cbr\u003ebin %{x}\u003cbr\u003emag %{z:.3f}\u003cextra\u003e\u003c/extra\u003e"))
                fig_gain.update_layout(title=f"Frequency Gain Magnitude (Mixer {lid})",
                                       xaxis_title="frequency bin", yaxis_title="channel", height=500)
                figures[f"Frequency Gains (Mixer {lid})"] = fig_gain

    # 3) Weight distribution
    all_vals = []
    for k, v in state.items():
        if hasattr(v, 'flatten'):
            all_vals.append(v.flatten().detach().cpu().numpy())
    if all_vals:
        all_vals = np.concatenate(all_vals)
        fig_hist = go.Figure(data=go.Histogram(x=all_vals, nbinsx=200, name="all weights"))
        fig_hist.update_layout(title="Global Weight Distribution", xaxis_title="value", yaxis_title="count",
                               xaxis=dict(range=[np.percentile(all_vals, 0.1), np.percentile(all_vals, 99.9)]))
        figures["Weight Distribution"] = fig_hist

    # 4) Fourier / head info text
    info_lines = []
    for k, v in sorted(state.items()):
        if not hasattr(v, 'shape'):
            continue
        if any(tag in k for tag in ("period", "harmonic", "dc", "A", "B", "proj_dir", "frequencies", "scale")):
            info_lines.append(f"{k}: {v.shape} | mean={v.mean().item() if hasattr(v, 'mean') else 'n/a'}")
    info_text = "\n".join(info_lines) if info_lines else "No head parameters found."
    fig_info = go.Figure()
    fig_info.add_annotation(text=info_text, showarrow=False, xref="paper", yref="paper", align="left")
    fig_info.update_layout(height=400)
    figures["Head / Fourier Parameters"] = fig_info

    return figures


def get_activations(model, meta, batch_size=64):
    device = next(model.parameters()).device
    arch = meta.get("arch", "")
    if "SpectralV3" in arch and meta.get("input_type") == "image":
        shape = (batch_size, meta["in_ch"], *meta["img_size"])
    elif "SpectralModel" in arch:
        shape = (batch_size, meta["in_channels"], *meta["spatial_shape"])
    else:
        # generic fallback
        shape = (batch_size, 1, 28, 28)
    x = torch.randn(*shape, device=device)
    y = torch.randint(0, int(model.classifier[-1].out_features), (batch_size,))
    acts = {}
    with ActivationExtractor(capture_input=False, detach=True) as ex:
        names = [n for n, _ in model.named_modules() if n and not n.endswith(".")]
        ex.register_hooks(model, layer_names=names)
        with torch.no_grad():
            model(x)
        acts = ex.get_activations()
    return x, y, acts


def generate_interpretation_dashboard(weights_path, device="cpu", batch_size=64, output="spectral_dashboard.html", manual_overrides=None):
    try:
        model, meta = load_checkpoint(weights_path, device=device, manual_overrides=manual_overrides)
        print(f"Loaded {meta.get('arch', 'unknown')} from {weights_path}")
        use_model = True
    except Exception as e:
        print(f"Could not instantiate model from {weights_path}: {e}")
        print("Falling back to raw state-dict analysis.")
        raw = torch.load(weights_path, map_location=device, weights_only=False)
        state = raw.get("state_dict", raw.get("model_state_dict", raw))
        figures = analyze_raw_state_dict(state)
        build_dashboard(figures, filepath=output, title="Spectral Interpretation Dashboard (Raw State Dict)")
        print(f"Dashboard written to {os.path.abspath(output)}")
        return

    x, y, acts = get_activations(model, meta, batch_size=batch_size)

    figures = {}
    # 1) Frequency gains
    try:
        info = analyze_frequency_gains(model)
        if info:
            for li in list(info.keys())[:2]:
                figures[f"Frequency Gains (Layer {li})"] = plot_frequency_gains(info, layer_idx=li, top_k=32)
    except Exception as e:
        print("[skip frequency gains]", e)

    # 2) Coefficient decomposition
    try:
        dec = decompose_coefficients(model, sample_batch=x[:16])
        fig_coef = go.Figure()
        fig_coef.add_trace(go.Scatter(y=dec["a0"][0].ravel(), mode="lines+markers", name="a0"))
        fig_coef.add_trace(go.Scatter(y=dec["a_n"][0], mode="lines+markers", name="a_n"))
        fig_coef.add_trace(go.Scatter(y=dec["b_n"][0], mode="lines+markers", name="b_n"))
        fig_coef.update_layout(title="Coefficients sample 0", height=400)
        figures["Coefficient Decomposition"] = fig_coef
    except Exception as e:
        print("[skip coefficient decomposition]", e)

    # 3) Layer manifold (first 4 captured layers)
    try:
        chosen = [k for k in list(acts.keys())[:4] if "output" in acts[k]]
        if chosen:
            act_dict = {k: acts[k]["output"] for k in chosen}
            fig_manifold = visualize_layer_projection(act_dict, labels=y.numpy(), method="pca", dims=3)
            figures["Layer Manifold (PCA 3D)"] = fig_manifold
    except Exception as e:
        print("[skip layer manifold]", e)

    # 4) Folding complexity
    try:
        traj = track_latent_trajectory(model, x)
        fold = compute_fold_measure(traj)
        fig_fold = go.Figure(data=go.Scatter(
            x=list(range(len(fold["per_layer"]))), y=fold["per_layer"], mode="lines+markers"))
        fig_fold.update_layout(title="Folding Complexity", xaxis_title="layer", yaxis_title="fold measure", height=400)
        figures["Folding Complexity"] = fig_fold
    except Exception as e:
        print("[skip folding complexity]", e)

    # 5) Partition estimate text
    try:
        part = compute_spectral_partition_count(model)
        fig_text = go.Figure()
        fig_text.add_annotation(
            text=f"Spectral partition estimate: per_layer={part['per_layer']}, total={part['total']}",
            showarrow=False, xref="paper", yref="paper")
        fig_text.update_layout(height=200)
        figures["Partition Estimate"] = fig_text
    except Exception as e:
        print("[skip partition estimate]", e)

    build_dashboard(figures, filepath=output, title="Spectral Interpretation Dashboard")
    print(f"Dashboard written to {os.path.abspath(output)}")


def main():
    parser = argparse.ArgumentParser(description="Generate spectral model interpretation dashboard")
    parser.add_argument("--weights", type=str, default="", help="Model checkpoint path")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output", type=str, default="spectral_dashboard.html")
    parser.add_argument("--spatial", type=int, nargs="+", default=None, help="Spatial shape (e.g. 28 28)")
    parser.add_argument("--channels", type=int, default=None, help="Input channels")
    parser.add_argument("--classes", type=int, default=None, help="Num classes")
    args = parser.parse_args()

    manual = {}
    if args.spatial is not None:
        manual["spatial_shape"] = tuple(args.spatial)
    if args.channels is not None:
        manual["in_channels"] = args.channels
    if args.classes is not None:
        manual["num_classes"] = args.classes

    generate_interpretation_dashboard(
        weights_path=args.weights,
        device=args.device,
        batch_size=args.batch_size,
        output=args.output,
        manual_overrides=manual if manual else None,
    )


if __name__ == "__main__":
    main()
