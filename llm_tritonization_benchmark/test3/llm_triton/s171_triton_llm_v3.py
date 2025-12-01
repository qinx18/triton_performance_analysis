import triton
import triton.language as tl
import torch

@triton.jit
def s171_kernel(a_ptr, b_ptr, inc, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load b values
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    # Calculate a indices with stride
    a_indices = offsets * inc
    a_mask = mask & (a_indices < n_elements * inc)
    
    # Load a values
    a_vals = tl.load(a_ptr + a_indices, mask=a_mask)
    
    # Compute and store
    result = a_vals + b_vals
    tl.store(a_ptr + a_indices, result, mask=a_mask)

def s171_triton(a, b, inc):
    n_elements = b.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s171_kernel[grid](a, b, inc, n_elements, BLOCK_SIZE=BLOCK_SIZE)