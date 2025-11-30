import triton
import triton.language as tl
import torch

@triton.jit
def s111_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Convert to actual indices (1, 3, 5, ...)
    actual_indices = 2 * offsets + 1
    mask = actual_indices < n_elements
    
    # Load a[i-1] and b[i]
    prev_indices = actual_indices - 1
    prev_mask = (prev_indices >= 0) & mask
    
    a_prev = tl.load(a_ptr + prev_indices, mask=prev_mask, other=0.0)
    b_vals = tl.load(b_ptr + actual_indices, mask=mask, other=0.0)
    
    # Compute a[i] = a[i-1] + b[i]
    result = a_prev + b_vals
    
    # Store result
    tl.store(a_ptr + actual_indices, result, mask=mask)

def s111_triton(a, b):
    n_elements = a.shape[0]
    # Number of odd indices from 1 to n_elements-1
    num_iterations = (n_elements - 1 + 1) // 2
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(num_iterations, BLOCK_SIZE),)
    
    s111_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a