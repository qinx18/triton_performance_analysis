import torch
import triton
import triton.language as tl

@triton.jit
def s353_kernel(a_ptr, b_ptr, c_ptr, ip_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    # Load alpha (scalar)
    alpha = tl.load(c_ptr)
    
    # Load indices for gather
    ip_indices = tl.load(ip_ptr + offsets, mask=mask, other=0)
    
    # Create mask for valid indices (within bounds and original mask)
    ip_mask = (ip_indices >= 0) & (ip_indices < n) & mask
    
    # Gather from b array using ip indices
    b_vals = tl.load(b_ptr + ip_indices, mask=ip_mask, other=0.0)
    
    # Load current a values
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    
    # Compute saxpy: a += alpha * b[ip[i]] (only where mask is valid)
    result = tl.where(ip_mask, a_vals + alpha * b_vals, a_vals)
    
    # Store back to a
    tl.store(a_ptr + offsets, result, mask=mask)

def s353_triton(a, b, c, ip):
    n = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s353_kernel[grid](
        a, b, c, ip, n,
        BLOCK_SIZE=BLOCK_SIZE
    )