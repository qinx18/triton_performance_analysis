import triton
import triton.language as tl
import torch

@triton.jit
def s128_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    # Load data
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    
    # Compute k values: k = 2*i for each i
    k_offsets = 2 * offsets
    k_mask = k_offsets < (2 * n)
    
    # Load b[k] and c[k] values
    b_k_vals = tl.load(b_ptr + k_offsets, mask=k_mask)
    c_k_vals = tl.load(c_ptr + k_offsets, mask=k_mask)
    
    # Compute a[i] = b[k] - d[i]
    a_vals = b_k_vals - d_vals
    
    # Store a[i]
    tl.store(a_ptr + offsets, a_vals, mask=mask)
    
    # Compute b[k] = a[i] + c[k]
    b_new_vals = a_vals + c_k_vals
    
    # Store b[k]
    tl.store(b_ptr + k_offsets, b_new_vals, mask=k_mask)

def s128_triton(a, b, c, d):
    N = a.shape[0] // 2
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s128_kernel[grid](a, b, c, d, N, BLOCK_SIZE=BLOCK_SIZE)