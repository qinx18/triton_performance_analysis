import torch
import triton
import triton.language as tl

@triton.jit
def s1244_kernel(a_ptr, a_copy_ptr, b_ptr, c_ptr, d_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    mask = idx < n
    
    # Load data
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    
    # Compute a[i] = b[i] + c[i] * c[i] + b[i]*b[i] + c[i]
    a_vals = b_vals + c_vals * c_vals + b_vals * b_vals + c_vals
    
    # Store a[i]
    tl.store(a_ptr + idx, a_vals, mask=mask)
    
    # Load a[i+1] from the read-only copy
    idx_plus1 = idx + 1
    mask_plus1 = idx_plus1 < (n + 1)  # Allow reading one element beyond n for a[i+1]
    a_plus1_vals = tl.load(a_copy_ptr + idx_plus1, mask=mask_plus1)
    
    # Compute d[i] = a[i] + a[i+1]
    d_vals = a_vals + a_plus1_vals
    
    # Store d[i]
    tl.store(d_ptr + idx, d_vals, mask=mask)

def s1244_triton(a, b, c, d):
    n = a.shape[0] - 1  # Loop runs from 0 to LEN_1D-1
    
    # Create read-only copy for WAR dependency handling
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s1244_kernel[grid](
        a, a_copy, b, c, d, n, BLOCK_SIZE
    )