import triton
import triton.language as tl
import torch

@triton.jit
def s353_kernel(a_ptr, b_ptr, ip_ptr, alpha, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load indices - ensure they're within bounds
    indices = tl.load(ip_ptr + offsets, mask=mask, other=0)
    
    # Clamp indices to valid range to prevent out-of-bounds access
    indices = tl.where((indices >= 0) & (indices < n_elements), indices, 0)
    
    # Load a values
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    
    # Load b values using gathered indices
    b_vals = tl.load(b_ptr + indices, mask=mask, other=0.0)
    
    # Compute: a[i] += alpha * b[ip[i]]
    result = a_vals + alpha * b_vals
    
    # Store result
    tl.store(a_ptr + offsets, result, mask=mask)

def s353_triton(a, b, c, ip):
    n_elements = a.shape[0]
    alpha = c[0].item()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s353_kernel[grid](
        a, b, ip, alpha, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )