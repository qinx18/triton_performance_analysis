import torch
import triton
import triton.language as tl

@triton.jit
def s131_kernel(a_ptr, a_copy_ptr, b_ptr, m, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Read from copy (immutable), write to original
    a_vals = tl.load(a_copy_ptr + indices + m, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    
    result = a_vals + b_vals
    
    tl.store(a_ptr + indices, result, mask=mask)

def s131_triton(a, b, m):
    n_elements = a.shape[0] - 1  # Loop goes to LEN_1D - 1
    
    # Create read-only copy to handle WAR dependency
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s131_kernel[grid](
        a, a_copy, b, m, n_elements, BLOCK_SIZE
    )