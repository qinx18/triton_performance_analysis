import torch
import triton
import triton.language as tl

@triton.jit
def vas_kernel(a_ptr, b_ptr, ip_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load values from b
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    # Load indices from ip
    indices = tl.load(ip_ptr + offsets, mask=mask)
    
    # Scatter to a using indices from ip
    tl.store(a_ptr + indices, b_vals, mask=mask)

def vas_triton(a, b, ip):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    vas_kernel[grid](
        a, b, ip,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )