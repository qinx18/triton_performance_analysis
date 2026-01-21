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
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    
    # Create mask for valid indices to prevent out-of-bounds b access
    valid_indices_mask = mask & (indices >= 0) & (indices < n_elements)
    
    # Load b values using gathered indices, with safe indexing
    b_vals = tl.load(b_ptr + indices, mask=valid_indices_mask, other=0.0)
    
    # Apply the valid mask to b_vals for computation
    b_vals = tl.where(valid_indices_mask, b_vals, 0.0)
    
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