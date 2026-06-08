"""
fourier_embeddings.py — Spectral Word Embedding Layer and Operators
==================================================================
Implements:
1. FourierEmbedding: Structured representation of words as truncated Fourier series.
2. PhaseShiftOperator: Grammatical shifts as energy-preserving rotations.
3. compose_circular_convolution: Semantic composition via circular convolution.
"""

import torch
import torch.nn as nn
import numpy as np

class FourierEmbedding(nn.Module):
    """
    Structured word embedding where each word is represented by a truncated Fourier series.
    The vector representation returned is scaled such that the Euclidean inner product
    is exactly the continuous L2 inner product over the period P.
    
    Representation format:
    v = [a0, a_1 / sqrt(2), ..., a_N / sqrt(2), b_1 / sqrt(2), ..., b_N / sqrt(2)]
    """
    def __init__(self, num_embeddings, num_modes, init_scale=2.0):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.num_modes = num_modes
        
        # Learnable period scale P
        self.scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))
        
        # Fourier coefficients: a0 (DC), a_n (cosines), b_n (sines)
        self.a0 = nn.Parameter(torch.randn(num_embeddings, 1) * 0.05)
        self.a_n = nn.Parameter(torch.randn(num_embeddings, num_modes) * 0.05)
        self.b_n = nn.Parameter(torch.randn(num_embeddings, num_modes) * 0.05)
        
        # Keep track of mode numbers (1 to N)
        self.register_buffer('harmonic_n', torch.arange(1, num_modes + 1, dtype=torch.float32))

    def forward(self, word_ids):
        """
        Returns the L2-normalized/scaled embedding vector for the given word ids.
        Output shape: (B, 1 + 2*num_modes)
        """
        a0_val = self.a0[word_ids]
        an_val = self.a_n[word_ids]
        bn_val = self.b_n[word_ids]
        
        # Scale to ensure Euclidean inner product matches continuous L2 inner product
        an_scaled = an_val / np.sqrt(2.0)
        bn_scaled = bn_val / np.sqrt(2.0)
        
        return torch.cat([a0_val, an_scaled, bn_scaled], dim=-1)

    def get_raw_coefficients(self, word_ids):
        """
        Returns raw unscaled a0, a_n, b_n coefficients.
        """
        return self.a0[word_ids], self.a_n[word_ids], self.b_n[word_ids]

    def reconstruct_function(self, word_ids, t):
        """
        Reconstructs the continuous function f_w(t) at the given time/spatial points t.
        Args:
            word_ids: tensor of shape (B,) or long indices
            t: tensor of shape (T,) containing time/spatial points in [0, P]
        Returns:
            f_w(t): tensor of shape (B, T)
        """
        a0_val, an_val, bn_val = self.get_raw_coefficients(word_ids)
        # a0_val: (B, 1)
        # an_val: (B, N)
        # bn_val: (B, N)
        
        # Angular frequencies w_n = 2 * pi * n / P
        freqs = 2.0 * np.pi * self.harmonic_n / (self.scale.abs() + 1e-6) # (N,)
        
        # Projection: t.unsqueeze(0) * freqs.unsqueeze(1) -> (N, T)
        # We broadcast across batch: proj has shape (B, N, T)
        proj = freqs.unsqueeze(1) * t.unsqueeze(0) # (N, T)
        proj = proj.unsqueeze(0) # (1, N, T)
        
        # Term calculations: an_val.unsqueeze(-1) shape is (B, N, 1)
        cos_terms = an_val.unsqueeze(-1) * torch.cos(proj) # (B, N, T)
        sin_terms = bn_val.unsqueeze(-1) * torch.sin(proj) # (B, N, T)
        
        # f_w(t) = a0 + sum(cos_terms + sin_terms)
        f_t = a0_val + cos_terms.sum(dim=1) + sin_terms.sum(dim=1) # (B, T)
        return f_t


class PhaseShiftOperator(nn.Module):
    """
    Applies a learnable phase shift (rotation) to the Fourier coefficients.
    Used to model grammatical shifts (e.g. singular -> plural).
    
    If mode is 'shared', a single shift parameter theta is learned.
    If mode is 'independent', a separate shift theta_n is learned for each frequency mode.
    """
    def __init__(self, num_modes, mode='shared'):
        super().__init__()
        self.num_modes = num_modes
        self.mode = mode
        
        if mode == 'shared':
            self.theta = nn.Parameter(torch.tensor(0.0))
        elif mode == 'independent':
            self.theta = nn.Parameter(torch.zeros(num_modes))
        else:
            raise ValueError(f"Unknown mode: {mode}")
            
        self.register_buffer('harmonic_n', torch.arange(1, num_modes + 1, dtype=torch.float32))

    def forward(self, v):
        """
        Args:
            v: tensor of shape (..., 1 + 2*num_modes) representing scaled embedding vectors
        Returns:
            v_shifted: tensor of shape (..., 1 + 2*num_modes)
        """
        # Deconstruct into a0, an_scaled, bn_scaled
        a0 = v[..., 0:1]
        an_scaled = v[..., 1:1 + self.num_modes]
        bn_scaled = v[..., 1 + self.num_modes:]
        
        # Compute angles for each mode
        if self.mode == 'shared':
            angles = self.harmonic_n * self.theta  # (N,)
        else:
            angles = self.theta  # (N,)
            
        # Broadcast angles to match the dimensions of an_scaled and bn_scaled
        cos_ang = torch.cos(angles)
        sin_ang = torch.sin(angles)
        
        # Apply orthogonal rotation (time shift property: f(t - tau) <=> R(n*theta))
        an_shifted = an_scaled * cos_ang + bn_scaled * sin_ang
        bn_shifted = -an_scaled * sin_ang + bn_scaled * cos_ang
        
        # DC component a0 remains unchanged
        return torch.cat([a0, an_shifted, bn_shifted], dim=-1)


def compose_circular_convolution(v1, v2, num_modes):
    """
    Performs semantic composition via circular convolution of the wave representations.
    In the Fourier domain, circular convolution corresponds to element-wise complex multiplication.
    
    Args:
        v1: (..., 1 + 2*num_modes) scaled coefficients for word 1
        v2: (..., 1 + 2*num_modes) scaled coefficients for word 2
        num_modes: int
    Returns:
        v_composed: (..., 1 + 2*num_modes) convolved scaled coefficients
    """
    # 1. Unscale to get raw coefficients
    a0_1 = v1[..., 0:1]
    an_1 = v1[..., 1:1 + num_modes] * np.sqrt(2.0)
    bn_1 = v1[..., 1 + num_modes:] * np.sqrt(2.0)
    
    a0_2 = v2[..., 0:1]
    an_2 = v2[..., 1:1 + num_modes] * np.sqrt(2.0)
    bn_2 = v2[..., 1 + num_modes:] * np.sqrt(2.0)
    
    # 2. Pointwise multiplication of complex coefficients:
    # (an_1 + i bn_1) * (an_2 + i bn_2)
    # real part: an_1 * an_2 - bn_1 * bn_2
    # imag part: an_1 * bn_2 + bn_1 * an_2
    a0_composed = a0_1 * a0_2
    an_composed = an_1 * an_2 - bn_1 * bn_2
    bn_composed = an_1 * bn_2 + bn_1 * an_2
    
    # 3. Rescale back to vector embedding format
    an_scaled = an_composed / np.sqrt(2.0)
    bn_scaled = bn_composed / np.sqrt(2.0)
    
    return torch.cat([a0_composed, an_scaled, bn_scaled], dim=-1)
