import torch
import triton
import triton.language as tl

@triton.jit
def s353_kernel(a_ptr, b_ptr, c_ptr, ip_ptr, n_elements, alpha, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load indices for gather operation
    ip_indices = tl.load(ip_ptr + indices, mask=mask, other=0)
    
    # Load a values
    a_vals = tl.load(a_ptr + indices, mask=mask)
    
    # Gather b values using ip indices
    b_vals = tl.load(b_ptr + ip_indices, mask=mask)
    
    # Compute: a[i] += alpha * b[ip[i]]
    result = a_vals + alpha * b_vals
    
    # Store result
    tl.store(a_ptr + indices, result, mask=mask)

def s353_triton(a, b, c, ip):
    n_elements = a.shape[0]
    alpha = c[0].item()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s353_kernel[grid](
        a, b, c, ip,
        n_elements, alpha,
        BLOCK_SIZE=BLOCK_SIZE
    )