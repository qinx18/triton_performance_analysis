import triton
import triton.language as tl
import torch

@triton.jit
def s353_kernel(a_ptr, b_ptr, ip_ptr, alpha, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load indices from ip array
    indices = tl.load(ip_ptr + offsets, mask=mask)
    
    # Gather values from b array using indices
    b_vals = tl.load(b_ptr + indices, mask=mask)
    
    # Load current values from a array
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    
    # Compute: a[i] += alpha * b[ip[i]]
    result = a_vals + alpha * b_vals
    
    # Store result back to a array
    tl.store(a_ptr + offsets, result, mask=mask)

def s353_triton(a, b, ip, alpha):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s353_kernel[grid](
        a, b, ip, alpha, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )