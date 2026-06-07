"""
mech_interp/visualizer.py — Interactive visualization components
===============================================================
Generates interactive Plotly dashboards for spectral model interpretation.
Self-contained; does not require Dash/Streamlit (pure HTML/Plotly.js).
"""
import os
import json
import numpy as np
import torch

_HAVE_PLOTLY = False
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    _HAVE_PLOTLY = True
except Exception:
    pass

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_plotly():
    if not _HAVE_PLOTLY:
        raise RuntimeError("Plotly required for visualization. Install: pip install plotly")


def _tensor_to_np(t):
    return t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)


# ---------------------------------------------------------------------------
# Visual core
# ---------------------------------------------------------------------------

def visualize_layer_projection(layer_outputs, labels=None, method="umap", dims=3, title="Layer Latent Space"):
    """
    Reduce high-dim activations to 2D / 3D and return interactive Plotly scatter.
    Args:
        layer_outputs: dict name -> np.ndarray or tensor (B, ...)
        labels: optional (B,) integer labels/colors.
        method: 'pca', 'umap', or 'tsne'.
    Returns:
        plotly.graph_objects.Figure
    """
    _ensure_plotly()
    if method == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_components=dims, random_state=0)
        except Exception:
            method = "pca"
    if method == "pca":
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=dims)
    elif method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=dims, perplexity=min(30, max(5, len(next(iter(layer_outputs.values()))) - 1)))

    fig = make_subplots(
        rows=1, cols=len(layer_outputs),
        subplot_titles=list(layer_outputs.keys()),
        specs=[[{"type": "scatter3d" if dims == 3 else "scatter"}] * len(layer_outputs)],
    )
    for col, (name, out) in enumerate(layer_outputs.items(), start=1):
        X = _tensor_to_np(out)
        X = X.reshape(X.shape[0], -1)
        if X.shape[1] < dims:
            continue
        Z = reducer.fit_transform(X)
        color = labels if labels is not None else np.zeros(len(Z))
        kwargs = dict(
            x=Z[:, 0], y=Z[:, 1],
            mode="markers",
            marker=dict(size=4, color=color, colorscale="Viridis", showscale=(col == 1)),
            name=name,
        )
        if dims == 3:
            kwargs["z"] = Z[:, 2]
            fig.add_trace(go.Scatter3d(**kwargs), row=1, col=col)
        else:
            fig.add_trace(go.Scatter(**kwargs), row=1, col=col)
    fig.update_layout(title=title, showlegend=False, height=500)
    return fig


def visualize_coefficient_evolution(history_dict, title="Coefficient Evolution"):
    """
    history_dict: key -> list of scalars over epochs/steps.
    Returns interactive line chart.
    """
    _ensure_plotly()
    fig = go.Figure()
    for key, vals in history_dict.items():
        if not vals:
            continue
        fig.add_trace(go.Scatter(
            y=vals, mode="lines+markers", name=key,
            hovertemplate="%{x}<br>%{y:.4f}<extra></extra>",
        ))
    fig.update_layout(
        title=title,
        xaxis_title="step / epoch",
        yaxis_title="value",
        hovermode="x unified",
        height=500,
    )
    return fig


def plot_frequency_gains(gain_info, layer_idx=0, top_k=32, title="Frequency Gains"):
    """
    Interactive heatmap of gain magnitude for one spectral mixer layer.
    gain_info: output of analyze_frequency_gains().
    """
    _ensure_plotly()
    info = gain_info.get(layer_idx)
    if info is None:
        raise ValueError(f"layer_idx {layer_idx} not found")
    mag = info["gain_magnitude"]
    # pick top-k strongest channels sorted by max gain
    ch_scores = mag.reshape(mag.shape[0], -1).max(axis=1)
    top_ch = np.argsort(ch_scores)[-top_k:]
    sub = mag[top_ch]

    fig = go.Figure(data=go.Heatmap(
        z=sub,
        colorscale="Cividis",
        hovertemplate="ch %{y}<br>bin %{x}<br>mag %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"{title} — mixer {layer_idx} (channels {sub.shape[0]})",
        xaxis_title="frequency bin",
        yaxis_title="channel index",
        height=500,
    )
    return fig


def animate_layer_cloud(layer_outputs, labels=None, method="pca", title="Layer Cloud Evolution"):
    """
    Returns a Plotly figure with slider frames showing layer-by-layer deformation.
    layer_outputs: dict name -> np.ndarray (B, ...).
    """
    _ensure_plotly()
    names = list(layer_outputs.keys())
    # project with shared PCA fit
    all_X = np.concatenate([_tensor_to_np(v).reshape(v.shape[0], -1) for v in layer_outputs.values()], axis=0)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    pca.fit(all_X)

    frames = []
    for name in names:
        X = _tensor_to_np(layer_outputs[name]).reshape(layer_outputs[name].shape[0], -1)
        Z = pca.transform(X)
        c = labels if labels is not None else np.zeros(len(Z))
        scatter = go.Scatter3d(
            x=Z[:, 0], y=Z[:, 1], z=Z[:, 2],
            mode="markers",
            marker=dict(size=3, color=c, colorscale="Viridis", showscale=False),
        )
        frames.append(go.Frame(data=[scatter], name=name))

    fig = go.Figure(data=frames[0].data, frames=frames)
    fig.update_layout(
        title=title,
        scene=dict(xaxis=dict(range=[all_X[:, 0].min(), all_X[:, 0].max()]),
                   yaxis=dict(range=[all_X[:, 1].min(), all_X[:, 1].max()]),
                   zaxis=dict(range=[all_X[:, 2].min(), all_X[:, 2].max()])),
        updatemenus=[dict(
            type="buttons", showactive=True,
            buttons=[
                dict(label="Play", method="animate", args=[None, dict(frame=dict(duration=600), transition=dict(duration=300), fromcurrent=True)]),
                dict(label="Pause", method="animate", args=[[None], dict(mode="immediate")]),
            ],
        )],
        sliders=[dict(
            active=0,
            steps=[dict(method="animate", args=[[f.name], dict(mode="immediate", frame=dict(duration=0))]) for f in frames],
        )],
        height=600,
    )
    return fig


# ---------------------------------------------------------------------------
# Dashboard builder
# ---------------------------------------------------------------------------

def build_dashboard(figures, filepath="spectral_dashboard.html", title="Spectral Interpretation Dashboard"):
    """
    Assemble multiple figures into one scrollable HTML page.
    figures: dict section_name -> plotly Figure.
    """
    _ensure_plotly()
    os.makedirs(os.path.dirname(os.path.abspath(filepath)) or ".", exist_ok=True)
    html_parts = [
        "<html><head><meta charset='utf-8'><title>" + title + "</title>",
        "<style>body{font-family:sans-serif;background:#f8f9fa;margin:20px;}"
        "section{background:#fff;border-radius:8px;padding:15px;margin-bottom:30px;box-shadow:0 2px 4px rgba(0,0,0,.1)}"
        "h2{color:#333}"
        "</style></head><body>",
        f"<h1>{title}</h1>",
    ]
    for section, fig in figures.items():
        inner = fig.to_html(include_plotlyjs="cdn", full_html=False)
        html_parts.append(f"<section><h2>{section}</h2>{inner}</section>")
    html_parts.append("</body></html>")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))
    return filepath


def export_summary_json(results, filepath="spectral_summary.json"):
    """
    Serialize lightweight JSON summary for downstream tools.
    """
    serializable = {}
    for k, v in results.items():
        if isinstance(v, np.ndarray):
            serializable[k] = {"shape": list(v.shape), "mean": float(v.mean()), "std": float(v.std())}
        elif isinstance(v, dict):
            serializable[k] = {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv) for kk, vv in v.items()}
        else:
            serializable[k] = v
    os.makedirs(os.path.dirname(os.path.abspath(filepath)) or ".", exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    return filepath
