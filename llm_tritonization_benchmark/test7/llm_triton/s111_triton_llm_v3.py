import torch
import triton
import triton.language as tl

@triton.jit
def s111_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    # Only process odd indices (1, 3, 5, ...)
    odd_indices = 2 * indices + 1
    mask = odd_indices < n_elements
    
    # Load a[i-1] (even indices: 0, 2, 4, ...)
    prev_indices = odd_indices - 1
    a_prev = tl.load(a_ptr + prev_indices, mask=mask)
    
    # Load b[i] (odd indices)
    b_vals = tl.load(b_ptr + odd_indices, mask=mask)
    
    # Compute a[i] = a[i-1] + b[i]
    result = a_prev + b_vals
    
    # Store back to a[i]
    tl.store(a_ptr + odd_indices, result, mask=mask)

def s111_triton(a, b):
    n_elements = a.shape[0]
    # Number of odd indices to process: (n_elements - 1) // 2
    n_odd = (n_elements - 1) // 2
    
    BLOCK_SIZE = 128
    grid = (triton.cdiv(n_odd, BLOCK_SIZE),)
    
    s111_kernel[grid](a, b, n_elements, BLOCK_SIZE)