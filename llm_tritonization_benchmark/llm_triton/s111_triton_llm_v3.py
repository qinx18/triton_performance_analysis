import triton
import triton.language as tl
import torch

@triton.jit
def s111_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Convert to actual indices: i = 1, 3, 5, 7, ... (start at 1, step by 2)
    indices = 1 + offsets * 2
    mask = indices < n_elements
    
    # Load a[i-1] and b[i]
    a_prev_indices = indices - 1
    a_prev_mask = (a_prev_indices >= 0) & mask
    b_mask = mask
    
    a_prev = tl.load(a_ptr + a_prev_indices, mask=a_prev_mask, other=0.0)
    b_vals = tl.load(b_ptr + indices, mask=b_mask, other=0.0)
    
    # Compute a[i] = a[i-1] + b[i]
    result = a_prev + b_vals
    
    # Store result
    tl.store(a_ptr + indices, result, mask=mask)

def s111_triton(a, b):
    n_elements = a.shape[0]
    
    # Calculate number of iterations: i goes from 1 to n_elements-1 with step 2
    # So we have indices: 1, 3, 5, 7, ..., up to n_elements-1
    num_iterations = (n_elements - 1 + 1) // 2  # +1 to include the boundary, then divide by step
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(num_iterations, BLOCK_SIZE),)
    
    s111_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )