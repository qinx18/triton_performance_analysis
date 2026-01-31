import triton
import triton.language as tl
import torch

@triton.jit
def s151_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get block starting position
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Create offset vector once
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    
    # Create mask for valid elements (i < n_elements - 1)
    mask = current_offsets < (n_elements - 1)
    
    # Load a[i+1] and b[i]
    a_shifted = tl.load(a_ptr + current_offsets + 1, mask=mask)
    b_vals = tl.load(b_ptr + current_offsets, mask=mask)
    
    # Compute a[i] = a[i+1] + b[i]
    result = a_shifted + b_vals
    
    # Store result back to a[i]
    tl.store(a_ptr + current_offsets, result, mask=mask)

def s151_triton(a, b):
    n_elements = a.shape[0]
    
    # We process n_elements - 1 elements (loop bound is LEN_1D-1)
    elements_to_process = n_elements - 1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(elements_to_process, BLOCK_SIZE),)
    
    s151_kernel[grid](
        a, b, n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )