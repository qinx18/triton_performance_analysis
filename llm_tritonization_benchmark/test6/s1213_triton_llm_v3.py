import triton
import triton.language as tl
import torch

@triton.jit
def s1213_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID and calculate offsets
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements (1 <= i <= n_elements-2)
    mask = (offsets >= 1) & (offsets < n_elements - 1)
    
    # Read-only copy of 'a' to handle WAR dependency
    a_old = tl.load(a_ptr + offsets + 1, mask=(offsets + 1 < n_elements))
    
    # Load required values
    b_prev = tl.load(b_ptr + offsets - 1, mask=(offsets >= 1))
    c_curr = tl.load(c_ptr + offsets, mask=mask)
    d_curr = tl.load(d_ptr + offsets, mask=mask)
    
    # Compute: a[i] = b[i-1] + c[i]
    a_new = b_prev + c_curr
    
    # Store a[i]
    tl.store(a_ptr + offsets, a_new, mask=mask)
    
    # Compute: b[i] = a[i+1] * d[i] (using read-only copy)
    b_new = a_old * d_curr
    
    # Store b[i]
    tl.store(b_ptr + offsets, b_new, mask=mask)

def s1213_triton(a, b, c, d):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s1213_kernel[grid](
        a, b, c, d, n_elements, BLOCK_SIZE
    )