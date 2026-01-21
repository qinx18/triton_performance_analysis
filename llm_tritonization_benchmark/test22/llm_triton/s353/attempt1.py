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
    ip_indices = tl.load(ip_ptr + offsets, mask=mask)
    ip_mask = ip_indices < n
    gather_mask = mask & ip_mask
    
    # Gather from b array
    b_vals = tl.load(b_ptr + ip_indices, mask=gather_mask)
    
    # Load current a values
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    
    # Compute saxpy: a += alpha * b[ip[i]]
    result = a_vals + alpha * b_vals
    
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