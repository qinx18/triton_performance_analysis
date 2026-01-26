import triton
import triton.language as tl
import torch

@triton.jit
def s353_kernel(a_ptr, b_ptr, ip_ptr, alpha, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load indices and mask for valid indices
    ip_vals = tl.load(ip_ptr + idx, mask=mask)
    ip_mask = (ip_vals >= 0) & (ip_vals < n_elements) & mask
    
    # Load a values
    a_vals = tl.load(a_ptr + idx, mask=mask)
    
    # Gather b values using indices from ip
    b_vals = tl.load(b_ptr + ip_vals, mask=ip_mask)
    
    # Compute: a[i] += alpha * b[ip[i]]
    result = a_vals + alpha * b_vals
    
    # Store result back to a
    tl.store(a_ptr + idx, result, mask=mask)

def s353_triton(a, b, ip, alpha):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s353_kernel[grid](
        a, b, ip, alpha, n_elements, BLOCK_SIZE
    )