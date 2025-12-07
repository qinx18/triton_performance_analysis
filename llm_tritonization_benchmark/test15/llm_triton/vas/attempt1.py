import torch
import triton
import triton.language as tl

@triton.jit
def vas_kernel(
    a_ptr,
    b_ptr,
    ip_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    
    offsets = tl.arange(0, BLOCK_SIZE)
    block_start = pid * BLOCK_SIZE
    current_offsets = block_start + offsets
    
    mask = current_offsets < n_elements
    
    # Load b values and indices
    b_vals = tl.load(b_ptr + current_offsets, mask=mask)
    indices = tl.load(ip_ptr + current_offsets, mask=mask)
    
    # Scatter: a[ip[i]] = b[i]
    tl.store(a_ptr + indices, b_vals, mask=mask)

def vas_triton(a, b, ip):
    n_elements = b.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    vas_kernel[grid](
        a,
        b, 
        ip,
        n_elements,
        BLOCK_SIZE,
    )