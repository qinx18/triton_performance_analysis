import torch
import triton
import triton.language as tl

@triton.jit
def vag_kernel(a_ptr, b_ptr, ip_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    mask = (block_start + offsets) < n_elements
    
    # Load indices
    ip_offsets = block_start + offsets
    indices = tl.load(ip_ptr + ip_offsets, mask=mask, other=0)
    
    # Gather from b using indices
    b_vals = tl.load(b_ptr + indices, mask=mask, other=0.0)
    
    # Store to a
    tl.store(a_ptr + ip_offsets, b_vals, mask=mask)

def vag_triton(a, b, ip):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    vag_kernel[grid](
        a, b, ip, n_elements, BLOCK_SIZE
    )