"""
mech_interp/streamlit_app.py — Interactive interpretation dashboard
===================================================================
Run with:   streamlit run mech_interp/streamlit_app.py
Or CLI:     python mech_interp/streamlit_app.py --weights best_v3_mnist.pt
Requires:   plotly, scikit-learn, torch, numpy  (streamlit optional)
"""
import os, sys, json, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mech_interp.loader import load_checkpoint
from mech_interp.core import (
    ActivationExtractor,
    analyze_frequency_gains,
    decompose_coefficients,
    CoefficientTracker,
    compute_fold_measure,
    compute_spectral_partition_count,
    track_latent_trajectory,
)
from mech_interp.visualizer import (
    visualize_layer_projection,
    plot_frequency_gains,
    build_dashboard,
)

try:
    import streamlit as st
    _STREAMLIT = True
except Exception:
    _STREAMLIT = False

try:
    import plotly.graph_objects as go
    _PLOTLY = True
except Exception:
    _PLOTLY = False


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def get_activations(model, batch_size=64):
    device = next(model.parameters()).device
    # infer input shape from model metadata if available
    in_ch = getattr(model, "_interp_in_ch", 1)
    spatial = getattr(model, "_interp_spatial", (28, 28))
    x = torch.randn(batch_size, in_ch, *spatial, device=device)
    y = torch.randint(0, int(model.classifier[-1].out_features), (batch_size,))
    acts = {}
    with ActivationExtractor(capture_input=False, detach=True) as ex:
        names = [n for n, _ in model.named_modules() if n and not n.endswith(".")]
        ex.register_hooks(model, layer_names=names)
        with torch.no_grad():
            model(x)
        acts = ex.get_activations()
    return x, y, acts


# ---------------------------------------------------------------------------
# CLI dashboard builder
# ---------------------------------------------------------------------------

def build_cli_dashboard(model, x, y, acts, output_path="spectral_dashboard.html"):
    if not _PLOTLY:
        print("Plotly missing. Install: pip install plotly")
        return
    figures = {}
    info = analyze_frequency_gains(model)
    if info:
        fig_gain = plot_frequency_gains(info, layer_idx=0)
        figures["Frequency Gains (Layer 0)"] = fig_gain

    dec = decompose_coefficients(model, sample_batch=x[:16])
    fig_coef = go.Figure()
    fig_coef.add_trace(go.Scatter(y=dec["a0"][0].ravel(), mode="lines+markers", name="a0"))
    fig_coef.add_trace(go.Scatter(y=dec["a_n"][0], mode="lines+markers", name="a_n"))
    fig_coef.add_trace(go.Scatter(y=dec["b_n"][0], mode="lines+markers", name="b_n"))
    fig_coef.update_layout(title="Coefficients sample 0", height=400)
    figures["Coefficient Decomposition"] = fig_coef

    chosen = [k for k in list(acts.keys())[:4] if "output" in acts[k]]
    if chosen:
        act_dict = {k: acts[k]["output"] for k in chosen}
        fig_manifold = visualize_layer_projection(act_dict, labels=y.numpy(), method="pca", dims=3)
        figures["Layer Manifold (PCA 3D)"] = fig_manifold

    traj = track_latent_trajectory(model, x)
    fold = compute_fold_measure(traj)
    fig_fold = go.Figure(data=go.Scatter(
        x=list(range(len(fold["per_layer"]))), y=fold["per_layer"], mode="lines+markers"))
    fig_fold.update_layout(title="Folding Complexity", xaxis_title="layer", yaxis_title="fold measure", height=400)
    figures["Folding Complexity"] = fig_fold

    part = compute_spectral_partition_count(model)
    fig_text = go.Figure()
    fig_text.add_annotation(
        text=f"Spectral partition estimate: {part}", showarrow=False, xref="paper", yref="paper")
    fig_text.update_layout(height=200)
    figures["Partition Estimate"] = fig_text

    build_dashboard(figures, filepath=output_path, title="Spectral Interpretation Dashboard")
    print(f"Dashboard written to {output_path}")


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def streamlit_main():
    st.set_page_config(page_title="Spectral Interpretation Dashboard", layout="wide")
    st.title("Spectral Model Mechanical Interpretation")
    st.markdown("---")

    with st.sidebar:
        st.header("Setup")
        weights_path = st.text_input("Model checkpoint (.pt)", value="best_v3_mnist.pt")
        device = st.selectbox("Device", ["cpu", "cuda"], index=0)
        st.markdown("---")
        task = st.radio("Task", ["Overview", "Frequency Gains", "Coefficient Decomposition",
                                 "Layer Manifold", "Folding Complexity", "Attribution"])

    model, meta = load_checkpoint(weights_path, device=device)
    st.sidebar.write("Detected:", meta.get("arch", "unknown"))
    x, y, acts = get_activations(model, batch_size=64)
    y_np = y.numpy()

    if task == "Overview":
        st.subheader("Model Overview")
        c1, c2, c3 = st.columns(3)
        total = sum(p.numel() for p in model.parameters())
        c1.metric("Total Parameters", f"{total:,}")
        c2.metric("Layers", len(list(model.named_modules())))
        c3.metric("Mixer Count", len(model.mixers) if hasattr(model, "mixers") else 0)
        st.markdown("### Metadata")
        st.json(meta)
        part = compute_spectral_partition_count(model)
        st.markdown("### Spectral Partition Estimate")
        st.json(part)

    elif task == "Frequency Gains":
        st.subheader("Spectral Mixer Frequency Gains")
        info = analyze_frequency_gains(model)
        layers = list(info.keys())
        if not layers:
            st.warning("No SpectralMixer layers found.")
        else:
            chosen = st.selectbox("Mixer layer", layers, format_func=lambda i: f"Layer {i} — {info[i]['name']}")
            topk = st.slider("Top channels", 4, 128, 32)
            fig = plot_frequency_gains(info, layer_idx=chosen, top_k=topk)
            st.plotly_chart(fig, use_container_width=True)

    elif task == "Coefficient Decomposition":
        st.subheader("Coefficient Decomposition")
        batch_size = st.slider("Batch size", 4, 128, 32)
        samp = x[:batch_size]
        dec = decompose_coefficients(model, sample_batch=samp)
        st.write("Frequencies:", dec["frequencies"])
        st.write("Coefficients shape:", dec["coefficients"].shape)
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=dec["a0"][0].ravel(), mode="lines+markers", name="a0"))
        fig.add_trace(go.Scatter(y=dec["a_n"][0], mode="lines+markers", name="a_n"))
        fig.add_trace(go.Scatter(y=dec["b_n"][0], mode="lines+markers", name="b_n"))
        fig.update_layout(title="Coefficients for sample 0")
        st.plotly_chart(fig, use_container_width=True)

    elif task == "Layer Manifold":
        st.subheader("Layer-by-Layer Manifold")
        layer_names = st.multiselect("Layers", list(acts.keys()), default=list(acts.keys())[:4])
        method = st.selectbox("Reduction", ["pca", "umap", "tsne"])
        dims = st.radio("Dimensions", [2, 3], index=1)
        act_dict = {k: acts[k]["output"] for k in layer_names if k in acts}
        if st.button("Compute Projection"):
            fig = visualize_layer_projection(act_dict, labels=y_np, method=method, dims=dims)
            st.plotly_chart(fig, use_container_width=True)

    elif task == "Folding Complexity":
        st.subheader("Folding / Region Complexity")
        traj = track_latent_trajectory(model, x)
        fold = compute_fold_measure(traj)
        st.line_chart(fold["per_layer"])
        st.caption("Per-layer fold measure")
        st.json({"cumulative": fold["cumulative"]})

    elif task == "Attribution":
        st.subheader("Spectral Attribution by Class")
        st.info("Synthetic labels for demo.")
        batch_size = st.slider("Batch", 4, 128, 64)
        dec = decompose_coefficients(model, sample_batch=x[:batch_size])
        coeffs = np.abs(dec["coefficients"])
        cls_idx = st.selectbox("Class", range(int(model.classifier[-1].out_features)))
        rng = np.random.RandomState(0)
        labels_demo = rng.randint(0, int(model.classifier[-1].out_features), len(coeffs))
        mask = labels_demo == cls_idx
        if mask.any():
            mean_abs = coeffs[mask].mean(axis=0)
            topk = st.slider("Top-K coeff", 4, 64, 16)
            top_idx = np.argsort(mean_abs)[-topk:][::-1]
            fig = go.Figure(data=go.Bar(x=[str(i) for i in top_idx], y=mean_abs[top_idx]))
            fig.update_layout(title=f"Top-{topk} coefficients for class {cls_idx}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No samples in selected class")

    st.markdown("---")
    if st.button("Export HTML Dashboard"):
        gain_info = analyze_frequency_gains(model)
        fig_gain = plot_frequency_gains(gain_info, layer_idx=0) if gain_info else go.Figure()
        out_path = "spectral_dashboard.html"
        build_dashboard({"Frequency Gains": fig_gain}, filepath=out_path, title="Spectral Dashboard")
        st.success(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if _STREAMLIT:
        streamlit_main()
    else:
        parser = argparse.ArgumentParser(description="Spectral model interpretation dashboard (CLI)")
        parser.add_argument("--weights", type=str, default="", help="Model checkpoint")
        parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
        parser.add_argument("--output", type=str, default="spectral_dashboard.html", help="Output HTML path")
        args = parser.parse_args()
        print("Streamlit not installed — running CLI dashboard builder.")
        model, meta = load_checkpoint(args.weights, device=args.device)
        x, y, acts = get_activations(model)
        build_cli_dashboard(model, x, y, acts, output_path=args.output)
