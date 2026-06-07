"""
mech_interp — Mechanical Interpretation for Spectral Neural Networks
"""
from .loader import load_checkpoint
from .core import (
    ActivationExtractor,
    CoefficientTracker,
    analyze_frequency_gains,
    decompose_coefficients,
    compute_manifold_structure,
    compute_fold_measure,
    compute_spectral_partition_count,
    track_latent_trajectory,
)
from .visualizer import (
    visualize_layer_projection,
    visualize_coefficient_evolution,
    plot_frequency_gains,
    animate_layer_cloud,
    build_dashboard,
    export_summary_json,
)

__all__ = [
    "load_checkpoint",
    "ActivationExtractor",
    "CoefficientTracker",
    "analyze_frequency_gains",
    "decompose_coefficients",
    "compute_manifold_structure",
    "compute_fold_measure",
    "compute_spectral_partition_count",
    "track_latent_trajectory",
    "visualize_layer_projection",
    "visualize_coefficient_evolution",
    "plot_frequency_gains",
    "animate_layer_cloud",
    "build_dashboard",
    "export_summary_json",
]
