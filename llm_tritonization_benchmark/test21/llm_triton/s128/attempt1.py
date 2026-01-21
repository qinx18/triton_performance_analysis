import triton
import triton.language as tl
import torch

@triton.jit
def s128_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_half, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_half
    
    # Calculate k indices: k = 2*i, 2*i+1 for each i
    k_indices = 2 * offsets
    k_indices_plus1 = k_indices + 1
    
    # Load values
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    b_k_vals = tl.load(b_ptr + k_indices, mask=mask)
    c_k_vals = tl.load(c_ptr + k_indices, mask=mask)
    
    # Compute a[i] = b[k] - d[i] where k = 2*i
    a_vals = b_k_vals - d_vals
    
    # Store a[i]
    tl.store(a_ptr + offsets, a_vals, mask=mask)
    
    # Compute b[k] = a[i] + c[k] where k = 2*i
    b_new_vals = a_vals + c_k_vals
    
    # Store b[k]
    tl.store(b_ptr + k_indices, b_new_vals, mask=mask)

def s128_triton(a, b, c, d):
    n = a.shape[0]
    n_half = n // 2
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_half, BLOCK_SIZE),)
    
    s128_kernel[grid](a, b, c, d, n_half, BLOCK_SIZE=BLOCK_SIZE)