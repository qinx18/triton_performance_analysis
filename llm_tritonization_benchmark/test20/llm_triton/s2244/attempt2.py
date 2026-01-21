import triton
import triton.language as tl
import torch

@triton.jit
def s2244_kernel(a_ptr, b_ptr, c_ptr, e_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate offsets for this block
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    # Mask for valid indices (i < n-1)
    mask = indices < (n - 1)
    
    # Load data
    b_vals = tl.load(b_ptr + indices, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + indices, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + indices, mask=mask, other=0.0)
    
    # First compute: a[i] = b[i] + c[i] (this happens for all valid i)
    val2 = b_vals + c_vals
    tl.store(a_ptr + indices, val2, mask=mask)
    
    # Then compute: a[i+1] = b[i] + e[i] (this overwrites a[i+1] except for last iteration)
    val1 = b_vals + e_vals
    tl.store(a_ptr + indices + 1, val1, mask=mask)

def s2244_triton(a, b, c, e):
    n = a.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n - 1, BLOCK_SIZE),)
    
    s2244_kernel[grid](a, b, c, e, n, BLOCK_SIZE=BLOCK_SIZE)