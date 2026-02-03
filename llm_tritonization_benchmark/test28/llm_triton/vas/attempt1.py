import triton
import triton.language as tl
import torch

@triton.jit
def vas_kernel(a_ptr, b_ptr, ip_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load values from b
    b_vals = tl.load(b_ptr + idx, mask=mask)
    
    # Load indices from ip
    indices = tl.load(ip_ptr + idx, mask=mask)
    
    # Scatter store to a using indices
    tl.store(a_ptr + indices, b_vals, mask=mask)

def vas_triton(a, b, ip):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    vas_kernel[grid](a, b, ip, n_elements, BLOCK_SIZE)