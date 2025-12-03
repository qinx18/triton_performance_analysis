import torch
import triton
import triton.language as tl

@triton.jit
def vag_kernel(a_ptr, b_ptr, ip_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    mask = idx < n_elements
    
    # Load indices
    indices = tl.load(ip_ptr + idx, mask=mask)
    
    # Load values using gathered indices
    values = tl.load(b_ptr + indices, mask=mask)
    
    # Store results
    tl.store(a_ptr + idx, values, mask=mask)

def vag_triton(a, b, ip):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    vag_kernel[grid](
        a, b, ip, 
        n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )