import torch
import triton
import triton.language as tl

@triton.jit
def s1221_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Start from index 4
    current_offsets = block_start + offsets + 4
    mask = current_offsets < n_elements
    
    # Load a[i]
    a_vals = tl.load(a_ptr + current_offsets, mask=mask)
    
    # Load b[i-4]
    b_prev_offsets = current_offsets - 4
    b_prev_vals = tl.load(b_ptr + b_prev_offsets, mask=mask)
    
    # Compute b[i] = b[i-4] + a[i]
    result = b_prev_vals + a_vals
    
    # Store result
    tl.store(b_ptr + current_offsets, result, mask=mask)

def s1221_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Calculate number of elements to process (from index 4 to n_elements-1)
    n_work = n_elements - 4
    grid = (triton.cdiv(n_work, BLOCK_SIZE),)
    
    s1221_kernel[grid](a, b, n_elements, BLOCK_SIZE)
    
    return b