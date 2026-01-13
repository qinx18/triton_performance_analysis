import torch
import triton
import triton.language as tl

@triton.jit
def vas_kernel(a_ptr, b_ptr, ip_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load indices from ip array
    scatter_indices = tl.load(ip_ptr + indices, mask=mask)
    
    # Load values from b array
    b_vals = tl.load(b_ptr + indices, mask=mask)
    
    # Scatter store to a array using the indirect indices
    tl.store(a_ptr + scatter_indices, b_vals, mask=mask)

def vas_triton(a, b, ip):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    vas_kernel[grid](a, b, ip, n_elements, BLOCK_SIZE)