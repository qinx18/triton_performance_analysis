import triton
import triton.language as tl
import torch

@triton.jit
def s431_kernel(a_ptr, a_copy_ptr, b_ptr, k, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n_elements
    
    # Load from read-only copy and b array
    a_copy_vals = tl.load(a_copy_ptr + indices + k, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    
    # Compute result
    result = a_copy_vals + b_vals
    
    # Store to original array
    tl.store(a_ptr + indices, result, mask=mask)

def s431_triton(a, b, k):
    N = a.shape[0]
    
    # Create read-only copy to handle WAR dependencies
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s431_kernel[grid](
        a, a_copy, b, k, N, BLOCK_SIZE
    )