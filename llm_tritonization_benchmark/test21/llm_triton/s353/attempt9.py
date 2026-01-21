import triton
import triton.language as tl
import torch

@triton.jit
def s353_kernel(a_ptr, b_ptr, c_ptr, ip_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load alpha from c[0]
    alpha = tl.load(c_ptr)
    
    # Load indices with mask to prevent out-of-bounds
    indices = tl.load(ip_ptr + offsets, mask=mask, other=0)
    
    # Load a values
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    
    # Load b values using gathered indices
    # For masked elements, use index 0 as safe fallback
    safe_indices = tl.where(mask, indices, 0)
    b_vals = tl.load(b_ptr + safe_indices)
    
    # Compute a[i] += alpha * b[ip[i]]
    result = a_vals + alpha * b_vals
    
    # Store result only for valid elements
    tl.store(a_ptr + offsets, result, mask=mask)

def s353_triton(a, b, c, ip):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s353_kernel[grid](
        a, b, c, ip,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )