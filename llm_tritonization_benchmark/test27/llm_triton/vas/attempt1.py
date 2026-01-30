import triton
import triton.language as tl
import torch

@triton.jit
def vas_kernel(a_ptr, b_ptr, ip_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    
    mask = current_offsets < n_elements
    
    # Load b values and ip indices
    b_vals = tl.load(b_ptr + current_offsets, mask=mask)
    ip_vals = tl.load(ip_ptr + current_offsets, mask=mask)
    
    # Store b values at scattered positions in a
    tl.store(a_ptr + ip_vals, b_vals, mask=mask)

def vas_triton(a, b, ip):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    vas_kernel[grid](
        a, b, ip,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )