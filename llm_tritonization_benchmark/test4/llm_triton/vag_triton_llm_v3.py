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
    ip_offsets = tl.load(ip_ptr + block_start + offsets, mask=mask)
    
    # Gather from b using the indices
    b_values = tl.load(b_ptr + ip_offsets, mask=mask)
    
    # Store to a
    tl.store(a_ptr + block_start + offsets, b_values, mask=mask)

def vag_triton(a, b, ip):
    n_elements = a.numel()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    vag_kernel[grid](a, b, ip, n_elements, BLOCK_SIZE)
    
    return a