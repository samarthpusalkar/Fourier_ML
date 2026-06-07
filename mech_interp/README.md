# mech_interp — Mechanical Interpretation for Spectral Models

## Core idea
Deep models recursively fold their input space. For spectral architectures, "folding" happens in **frequency space** (via complex gain multipliers in FFT layers) and **coefficient space** (via the CoefficientFourierHead). This module captures, measures, and visualizes that process.

## Files
- `core.py` — hooks, coefficient tracking, frequency-gain analysis, manifold structure, folding measures
- `visualizer.py` — Plotly figures: layer projections (PCA/UMAP/TSNE), coefficient evolution, frequency-gain heatmaps, animated cloud transitions, HTML dashboard builder
- `loader.py` — architecture-agnostic checkpoint loader with auto-detection (SpectralV3, SpectralModel, or manual fallback)
- `run_interpretation.py` — **CLI dashboard generator** (no Streamlit needed)
- `streamlit_app.py` — **Interactive Streamlit app** (degrades to CLI if streamlit missing)

## Quick start

### Generate static HTML dashboard (CLI)
```bash
python -m mech_interp.run_interpretation \
  --weights best_v3_mnist.pt \
  --output dashboard.html
```

If the checkpoint uses a new/unknown architecture, the tool falls back to raw state-dict analysis and still produces a dashboard.

### Override architecture manually (if auto-detect fails)
```bash
python -m mech_interp.run_interpretation \
  --weights best_cifar10_spectral.pt \
  --spatial 32 32 --channels 3 --classes 10 \
  --output dashboard.html
```

### Launch interactive Streamlit app
```bash
pip install streamlit
streamlit run mech_interp/streamlit_app.py
```

### Use programmatically
```python
from mech_interp.core import (
    ActivationExtractor, analyze_frequency_gains,
    decompose_coefficients, compute_fold_measure,
    track_latent_trajectory,
)
from mech_interp.visualizer import build_dashboard, visualize_layer_projection

model = ...  # SpectralModel with coefficient head
x = torch.randn(64, 1, 28, 28)

# 1. Capture layer activations
with ActivationExtractor() as ex:
    ex.register_hooks(model)
    model(x)
    acts = ex.get_activations()

# 2. Analyze frequency gains per mixer layer
info = analyze_frequency_gains(model)

# 3. Decompose Fourier coefficients
dec = decompose_coefficients(model, sample_batch=x)
# dec["a0"], dec["a_n"], dec["b_n"], dec["frequencies"]

# 4. Measure folding complexity
traj = track_latent_trajectory(model, x)
fold = compute_fold_measure(traj)
```

## Mapping to "recursive geometric folding" thesis
| Video concept | Spectral equivalent | Tool in this module |
|---------------|----------------------|---------------------|
| Linear fold lines (ReLU) | Complex gain rotation in FFT domain | `analyze_frequency_gains` |
| Exponential region growth | Partition count from stacked activations | `compute_spectral_partition_count` |
| Piece-wise polytope tiling | Per-sample coefficient clusters in known basis | `decompose_coefficients` |
| Layer-wise manifold deformation | PCA/3D projection of latent trajectory | `visualize_layer_projection` / `animate_layer_cloud` |
| Domain substrate (spatial coords) | Frequency bins / Fourier coefficients | `decompose_coefficients` + `plot_frequency_gains` |
| Deformation operator (activation) | LearnableSquareWave / Chebyshev / FM acts | `ActivationExtractor` captures post-activation outputs |
| Latent partitioning | Inter-class vs intra-class distance ratio | `compute_manifold_structure` |

## Extending to new modalities
Swap `spatial_shape` and `input_channels`. The interpretation logic stays the same:
1. `track_latent_trajectory` captures deformation at every layer
2. `compute_fold_measure` quantifies how much the space is "folded"
3. `decompose_coefficients` gives you an interpretable fixed-basis embedding for any input
