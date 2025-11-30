import torch
import triton
import triton.language as tl

@triton.jit
def vas_kernel(
    a_ptr, b_ptr, ip_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load b values and indices
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    indices = tl.load(ip_ptr + offsets, mask=mask)
    
    # Scatter: store b[i] to a[ip[i]]
    tl.store(a_ptr + indices, b_vals, mask=mask)

def vas_triton(a, b, ip):
    n_elements = b.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    vas_kernel[grid](
        a, b, ip,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )