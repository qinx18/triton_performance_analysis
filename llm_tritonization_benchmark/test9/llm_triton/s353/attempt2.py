import torch
import triton
import triton.language as tl

@triton.jit
def s353_kernel(a_ptr, b_ptr, c_ptr, ip_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load alpha
    alpha = tl.load(c_ptr)
    
    # Load current a values
    a_vals = tl.load(a_ptr + indices, mask=mask)
    
    # Load ip indices and gather from b
    ip_vals = tl.load(ip_ptr + indices, mask=mask)
    b_vals = tl.load(b_ptr + ip_vals, mask=mask)
    
    # Compute saxpy: a[i] += alpha * b[ip[i]]
    result = a_vals + alpha * b_vals
    
    # Store back to a
    tl.store(a_ptr + indices, result, mask=mask)

def s353_triton(a, b, c, ip):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256  # Power of 2
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s353_kernel[grid](
        a, b, c, ip,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )