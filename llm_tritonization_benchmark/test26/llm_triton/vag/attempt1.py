import triton
import triton.language as tl
import torch

@triton.jit
def vag_kernel(a_ptr, b_ptr, ip_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n_elements
    
    # Load indices
    indices = tl.load(ip_ptr + block_start + offsets, mask=mask)
    
    # Load values from b using gathered indices
    values = tl.load(b_ptr + indices, mask=mask)
    
    # Store to a
    tl.store(a_ptr + block_start + offsets, values, mask=mask)

def vag_triton(a, b, ip):
    n_elements = a.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    vag_kernel[grid](
        a, b, ip,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )