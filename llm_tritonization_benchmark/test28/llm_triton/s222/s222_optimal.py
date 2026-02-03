import torch
import triton
import triton.language as tl

# Optimal version: The a operations cancel out (a += b*c then a -= b*c)
# Only the e recurrence needs to be computed: e[i] = e[i-1] * e[i-1]
# This is inherently sequential - cannot be parallelized

# Version 1: Pure PyTorch (no kernel launch overhead)
def s222_triton_v1(a, b, c, e):
    """Pure PyTorch sequential - avoids kernel launch overhead."""
    n = e.shape[0]
    # e[i] = e[i-1] * e[i-1] is a sequential recurrence
    # Use a simple loop on CPU then copy back, or do it on GPU
    for i in range(1, n):
        e[i] = e[i-1] * e[i-1]

# Version 2: Minimal Triton kernel - only compute e recurrence
@triton.jit
def s222_kernel_v2(e_ptr, n_elements):
    # Single thread, minimal operations
    for i in range(1, n_elements):
        e_prev = tl.load(e_ptr + i - 1)
        e_new = e_prev * e_prev
        tl.store(e_ptr + i, e_new)

def s222_triton_v2(a, b, c, e):
    """Minimal Triton kernel - only e recurrence."""
    n_elements = e.shape[0]
    s222_kernel_v2[(1,)](e_ptr=e, n_elements=n_elements)

# Version 3: Use torch.cumprod with log trick (mathematically equivalent)
# e[i] = e[i-1]^2 means e[i] = e[0]^(2^i)
# This can be computed in parallel!
def s222_triton_v3(a, b, c, e):
    """Parallel computation using math: e[i] = e[0]^(2^i)."""
    n = e.shape[0]
    # e[i] = e[0]^(2^i)
    # Using log: log(e[i]) = 2^i * log(e[0])
    e0 = e[0].clone()
    powers = torch.pow(2.0, torch.arange(n, device=e.device, dtype=e.dtype))
    e[:] = torch.pow(e0, powers)

# The wrapper that uses the best version
def s222_triton(a, b, c, e):
    """Optimal: parallel power computation."""
    s222_triton_v3(a, b, c, e)
