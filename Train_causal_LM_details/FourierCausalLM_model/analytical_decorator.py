%%writefile analytical_decorator.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch

def compute_analytical_delta(x_in, error, lam=1e-3, max_norm=1.0, max_samples=None):
    """
    Computes the optimal weight update delta_W analytically.
    
    Args:
        x_in (torch.Tensor): Flattened input tensor of shape [N, D]
        error (torch.Tensor): Flattened error tensor of shape [N, Out] (target - current_output)
        lam (float): Tikhonov regularization lambda (Ridge parameter)
        max_norm (float): Maximum norm for gradient clipping
        max_samples (int): Max rows to randomly subsample (Sketching) to speed up Gram matrix
        
    Returns:
        torch.Tensor: dW of shape [Out, D]
    """
    N, D = x_in.shape
    device = x_in.device
    
    # Auto-Optimization: If the user didn't specify, but BOTH N and D are massively huge 
    # (e.g., massive LLM layers where N=4096 and D=4096), computing the exact covariance 
    # is mathematically impossible in real-time. We automatically apply sketching.
    # CRITICAL: We DO NOT sketch if D is small (like in CNNs where N=12500, D=16).
    # If D=16, X^T X is a tiny 16x16 matrix, which takes 0.001ms to invert using 100% of the data!
    if max_samples is None and min(N, D) > 1024:
        max_samples = 1024
    
    # Ensemble Optimization (Batched GPU Averaging)
    # Instead of a Python for loop, we reshape the tensor into batches and 
    # use PyTorch's native batched matrix multiplication (bmm) and batched 
    # Cholesky decomposition. This computes the entire ensemble in parallel 
    # on the GPU with zero iteration overhead!
    if max_samples is not None and N > max_samples:
        K = N // max_samples
        if K == 0: K = 1
        
        # Truncate to make perfectly divisible chunks
        N_trunc = K * max_samples
        x_c = x_in[:N_trunc].view(K, max_samples, D)
        e_c = error[:N_trunc].view(K, max_samples, -1)
        
        # We know max_samples <= min(N, D), so we use the Woodbury (N < D) form
        # M will be [K, max_samples, max_samples]
        I_batch = torch.eye(max_samples, device=device, dtype=x_in.dtype).unsqueeze(0)
        M_raw = x_c @ x_c.transpose(1, 2)
        
        # Scale-Invariant Regularization: scales lam by the variance of the input
        # This completely immunizes the Cholesky solver from crashing if weights drift
        scale = M_raw.diagonal(dim1=-2, dim2=-1).mean(dim=-1).view(K, 1, 1).clamp(min=1.0)
        M = M_raw + (lam * scale) * I_batch
        
        # Batched Cholesky
        L = torch.linalg.cholesky(M)
        
        # Batched Solve
        term = torch.cholesky_solve(e_c, L)
        
        # Batched Multiply
        dW_T_chunked = x_c.transpose(1, 2) @ term
        dW_chunked = dW_T_chunked.transpose(1, 2)
        
        # Average across chunks
        dW = dW_chunked.mean(dim=0)
        
        if max_norm is not None:
            norm = torch.norm(dW)
            if norm > max_norm:
                dW = dW * (max_norm / norm)
        return dW
    
    # Woodbury Matrix Identity for N < D (Small Batch, Large Feature Space)
    # This mathematically guarantees the same pseudo-inverse but reduces 
    # complexity from O(D^3) to O(N^3).
    # OPTIMIZATION: We use Cholesky Decomposition instead of standard LU solve. 
    # Since A @ A^T + lam * I is symmetric positive definite (SPD), Cholesky is 
    # exactly 2x faster (1/3 N^3 vs 2/3 N^3) and numerically superior.
    if N < D:
        I_N = torch.eye(N, device=device, dtype=x_in.dtype)
        M_raw = x_in @ x_in.T
        scale = M_raw.diagonal().mean().clamp(min=1.0)
        M = M_raw + (lam * scale) * I_N
        
        L = torch.linalg.cholesky(M)
        dW_T = x_in.T @ torch.cholesky_solve(error, L)
    else:
        I_D = torch.eye(D, device=device, dtype=x_in.dtype)
        M_raw = x_in.T @ x_in
        scale = M_raw.diagonal().mean().clamp(min=1.0)
        M = M_raw + (lam * scale) * I_D
        
        L = torch.linalg.cholesky(M)
        dW_T = torch.cholesky_solve(x_in.T @ error, L)
        
    dW = dW_T.T
    
    # Optional max-norm clipping to prevent destructive updates from noisy batches
    if max_norm is not None:
        norm = torch.norm(dW)
        if norm > max_norm:
            dW = dW * (max_norm / norm)
            
    return dW

def analytical_solver(lr=1.0, lr_decay=0.9, momentum=0.5, max_norm=10.0, lam=1e-3):
    """
    A generic decorator that infuses Analytical Target Propagation into any standard PyTorch model.
    It hooks into nn.Linear and nn.Conv2d to perform analytical weight updates on the backward pass,
    bypassing traditional SGD for these layers while leaving unhandled layers (like Embeddings or LayerNorm)
    free to be updated by standard optimizers.
    """
    def decorator(cls):
        original_init = cls.__init__
        
        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            
            # Attach hyperparameters
            self._analytical_lr = lr
            self._analytical_lr_decay = lr_decay
            self._analytical_momentum = momentum
            self._analytical_max_norm = max_norm
            self._analytical_lam = lam
            
            _register_analytical_hooks(self)
            
        cls.__init__ = new_init
        return cls
    return decorator

def _register_analytical_hooks(model):
    # State dictionary to track backward steps per pass
    model._analytical_state = {'backward_count': 0}
    
    # Pre-forward hook to reset the step counter
    def pre_forward_hook(module, input):
        module._analytical_state['backward_count'] = 0
    model.register_forward_pre_hook(pre_forward_hook)
    
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            
            # 1. Forward hook to save input
            def forward_hook(m, input, output):
                m._h_in = input[0].detach()
                
            module.register_forward_hook(forward_hook)
            
            # 2. Backward hook to compute Target Error and update weights
            def backward_hook(m, grad_input, grad_output):
                if not hasattr(m, '_h_in'):
                    return grad_input
                
                x_actual = m._h_in
                # Standard Autograd gives dL/dOutput. Target propagation expects Target - Output.
                # If Loss = 1/2*(Output - Target)^2, dL/dOutput = Output - Target.
                # So Error = -dL/dOutput
                error = -grad_output[0].detach().to(x_actual.dtype)
                
                # Apply depth-based learning rate decay (earlier layers get smaller LR to prevent drift)
                current_lr = model._analytical_lr * (model._analytical_lr_decay ** model._analytical_state['backward_count'])
                model._analytical_state['backward_count'] += 1
                
                lam = model._analytical_lam
                max_norm = model._analytical_max_norm
                momentum = model._analytical_momentum
                
                if isinstance(m, nn.Linear):
                    _update_linear(m, x_actual, error, current_lr, lam, max_norm, momentum)
                elif isinstance(m, nn.Conv2d):
                    _update_conv2d(m, x_actual, error, current_lr, lam, max_norm, momentum)
                
                # We return the clamped gradient to PyTorch to prevent Quadratic Error Compounding in deeper layers
                if grad_input is not None and grad_input[0] is not None:
                    clamped_grad_in = torch.clamp(grad_input[0], min=-1.0, max=1.0)
                    return (clamped_grad_in, *grad_input[1:])
                return grad_input
                
            module.register_full_backward_hook(backward_hook)
            
            # 3. Post-accumulate hook to hide these gradients from Adam / SGD
            # PyTorch 2.1+ supports this directly on the tensor
            if hasattr(module.weight, 'register_post_accumulate_grad_hook'):
                def post_acc_hook(param):
                    if param.grad is not None:
                        param.grad = None
                        
                module.weight.register_post_accumulate_grad_hook(post_acc_hook)
                if module.bias is not None:
                    module.bias.register_post_accumulate_grad_hook(post_acc_hook)

def _update_linear(layer, x_in, error, lr, lam, max_norm, momentum):
    """Analytically updates an nn.Linear layer."""
    # Handle multi-dimensional inputs (like Transformers [Batch, SeqLen, Dim])
    x_flat = x_in.reshape(-1, x_in.shape[-1])
    e_flat = error.reshape(-1, error.shape[-1])
    
    if layer.bias is not None:
        ones = torch.ones(x_flat.size(0), 1, dtype=x_flat.dtype, device=x_flat.device)
        x_aug = torch.cat([x_flat, ones], dim=1)
        dW_aug = compute_analytical_delta(x_aug, e_flat, lam=lam, max_norm=max_norm)
        
        dW = dW_aug[:, :-1]
        db = dW_aug[:, -1]
        
        if not hasattr(layer, 'dW_ema'):
            layer.dW_ema = torch.zeros_like(dW)
            layer.db_ema = torch.zeros_like(db)
            
        layer.dW_ema = momentum * layer.dW_ema + (1.0 - momentum) * dW
        layer.db_ema = momentum * layer.db_ema + (1.0 - momentum) * db
        
        layer.weight.data += lr * layer.dW_ema
        layer.bias.data += lr * layer.db_ema
    else:
        dW = compute_analytical_delta(x_flat, e_flat, lam=lam, max_norm=max_norm)
        
        if not hasattr(layer, 'dW_ema'):
            layer.dW_ema = torch.zeros_like(dW)
            
        layer.dW_ema = momentum * layer.dW_ema + (1.0 - momentum) * dW
        layer.weight.data += lr * layer.dW_ema

def _update_conv2d(layer, x_in, error, lr, lam, max_norm, momentum):
    """Analytically updates an nn.Conv2d layer."""
    N, C, H_in, W_in = x_in.shape
    
    # 1. Unfold input
    x_unfolded = F.unfold(
        x_in, kernel_size=layer.kernel_size, 
        dilation=layer.dilation, padding=layer.padding, stride=layer.stride
    )
    N, D_patch, L = x_unfolded.shape
    x_flat = x_unfolded.transpose(1, 2).reshape(N * L, D_patch)
    
    # 2. Unfold target error
    e_flat = error.view(N, layer.out_channels, -1).transpose(1, 2).reshape(N * L, layer.out_channels)
    
    if layer.bias is not None:
        ones = torch.ones(x_flat.size(0), 1, dtype=x_flat.dtype, device=x_flat.device)
        x_aug = torch.cat([x_flat, ones], dim=1)
        
        dW_aug = compute_analytical_delta(x_aug, e_flat, lam=lam, max_norm=max_norm)
        dW_flat = dW_aug[:, :-1]
        db = dW_aug[:, -1]
        
        dW = dW_flat.view(layer.weight.data.shape)
        
        if not hasattr(layer, 'dW_ema'):
            layer.dW_ema = torch.zeros_like(dW)
            layer.db_ema = torch.zeros_like(db)
            
        layer.dW_ema = momentum * layer.dW_ema + (1.0 - momentum) * dW
        layer.db_ema = momentum * layer.db_ema + (1.0 - momentum) * db
        
        layer.weight.data += lr * layer.dW_ema
        layer.bias.data += lr * layer.db_ema
    else:
        dW_flat = compute_analytical_delta(x_flat, e_flat, lam=lam, max_norm=max_norm)
        dW = dW_flat.view(layer.weight.data.shape)
        
        if not hasattr(layer, 'dW_ema'):
            layer.dW_ema = torch.zeros_like(dW)
            
        layer.dW_ema = momentum * layer.dW_ema + (1.0 - momentum) * dW
        layer.weight.data += lr * layer.dW_ema
