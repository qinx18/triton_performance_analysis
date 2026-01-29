import triton
import triton.language as tl
import torch

@triton.jit
def s353_kernel(a_ptr, b_ptr, ip_ptr, alpha, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load indices and apply bounds checking
    indices = tl.load(ip_ptr + offsets, mask=mask, other=0)
    
    # Load current values of a
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    
    # Load values from b using gathered indices
    b_vals = tl.load(b_ptr + indices, mask=mask)
    
    # Compute alpha * b[ip[i]] and add to a[i]
    result = a_vals + alpha * b_vals
    
    # Store result
    tl.store(a_ptr + offsets, result, mask=mask)

def s353_triton(a, b, ip):
    n_elements = a.shape[0]
    alpha = 1.0  # Default value, can be adjusted if c array is provided
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s353_kernel[grid](
        a, b, ip, alpha, n_elements, BLOCK_SIZE
    )