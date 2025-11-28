import triton
import triton.language as tl
import torch

@triton.jit
def s111_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Convert to actual indices (1, 3, 5, 7, ...)
    indices = 2 * offsets + 1
    mask = indices < n_elements
    
    # Load a[i-1] and b[i]
    a_prev_ptrs = a_ptr + indices - 1
    b_ptrs = b_ptr + indices
    
    a_prev = tl.load(a_prev_ptrs, mask=mask)
    b_vals = tl.load(b_ptrs, mask=mask)
    
    # Compute a[i] = a[i-1] + b[i]
    result = a_prev + b_vals
    
    # Store result
    a_ptrs = a_ptr + indices
    tl.store(a_ptrs, result, mask=mask)

def s111_triton(a, b):
    n_elements = a.shape[0]
    # Number of odd indices to process: (1, 3, 5, ..., n_elements-1 if odd)
    num_odd_indices = (n_elements - 1 + 1) // 2
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(num_odd_indices, BLOCK_SIZE),)
    
    s111_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )