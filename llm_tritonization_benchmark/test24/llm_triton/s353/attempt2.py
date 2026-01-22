import triton
import triton.language as tl
import torch

@triton.jit
def s353_kernel(a_ptr, b_ptr, c_ptr, ip_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load alpha
    alpha = tl.load(c_ptr)
    
    # Load indices and check bounds
    ip_vals = tl.load(ip_ptr + offsets, mask=mask, other=0)
    
    # Load a values
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    
    # Load b values using gather
    b_vals = tl.load(b_ptr + ip_vals, mask=mask)
    
    # Compute: a[i] += alpha * b[ip[i]]
    result = a_vals + alpha * b_vals
    
    # Store result
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