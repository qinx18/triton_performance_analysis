import triton
import triton.language as tl
import torch

@triton.jit
def s353_kernel(a_ptr, b_ptr, ip_ptr, alpha, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load indices for gather operation
    ip_vals = tl.load(ip_ptr + indices, mask=mask)
    
    # Gather from b array using indices
    b_vals = tl.load(b_ptr + ip_vals, mask=mask)
    
    # Load current a values
    a_vals = tl.load(a_ptr + indices, mask=mask)
    
    # Compute: a[i] += alpha * b[ip[i]]
    result = a_vals + alpha * b_vals
    
    # Store result
    tl.store(a_ptr + indices, result, mask=mask)

def s353_triton(a, b, ip, alpha):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s353_kernel[grid](
        a, b, ip, alpha, n_elements, BLOCK_SIZE
    )
    
    return a