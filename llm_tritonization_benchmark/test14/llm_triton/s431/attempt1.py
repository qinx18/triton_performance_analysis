import triton
import triton.language as tl
import torch

@triton.jit
def s431_kernel(a_ptr, a_copy_ptr, b_ptr, k, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n_elements
    
    # Load from read-only copy with offset k
    read_indices = indices + k
    read_mask = read_indices < n_elements
    a_vals = tl.load(a_copy_ptr + read_indices, mask=read_mask, other=0.0)
    
    # Load b values
    b_vals = tl.load(b_ptr + indices, mask=mask, other=0.0)
    
    # Compute result
    result = a_vals + b_vals
    
    # Store to original array
    tl.store(a_ptr + indices, result, mask=mask)

def s431_triton(a, b, k):
    n_elements = a.shape[0]
    
    # Create read-only copy to avoid WAR race condition
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s431_kernel[grid](
        a, a_copy, b, k, n_elements, BLOCK_SIZE
    )