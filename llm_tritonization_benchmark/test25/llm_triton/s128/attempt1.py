import triton
import triton.language as tl
import torch

@triton.jit
def s128_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_half, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets
    mask = i_offsets < n_half
    
    # Calculate k indices: k = 2*i (since j starts at -1 and increments by 2)
    k_offsets = 2 * i_offsets
    k_mask = k_offsets < (2 * n_half)
    
    # Load arrays
    d_vals = tl.load(d_ptr + i_offsets, mask=mask)
    b_vals = tl.load(b_ptr + k_offsets, mask=mask & k_mask)
    c_vals = tl.load(c_ptr + k_offsets, mask=mask & k_mask)
    
    # Compute a[i] = b[k] - d[i]
    a_vals = b_vals - d_vals
    
    # Store a[i]
    tl.store(a_ptr + i_offsets, a_vals, mask=mask)
    
    # Compute b[k] = a[i] + c[k]
    b_new_vals = a_vals + c_vals
    
    # Store b[k]
    tl.store(b_ptr + k_offsets, b_new_vals, mask=mask & k_mask)

def s128_triton(a, b, c, d):
    n_half = a.shape[0] // 2
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_half, BLOCK_SIZE),)
    
    s128_kernel[grid](a, b, c, d, n_half, BLOCK_SIZE)