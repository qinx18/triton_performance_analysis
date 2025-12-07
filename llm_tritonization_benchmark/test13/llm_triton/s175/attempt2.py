import torch
import triton
import triton.language as tl

@triton.jit
def s175_kernel(a_ptr, b_ptr, inc, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n - 1, inc * BLOCK_SIZE):
        current_offsets = block_start + offsets * inc
        mask = current_offsets < n - 1
        
        # Load b[i] values
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        
        # Load a[i + inc] values
        a_read_offsets = current_offsets + inc
        a_read_mask = mask & (a_read_offsets < n)
        a_vals = tl.load(a_ptr + a_read_offsets, mask=a_read_mask)
        
        # Compute a[i] = a[i + inc] + b[i]
        result = a_vals + b_vals
        
        # Store result to a[i]
        tl.store(a_ptr + current_offsets, result, mask=mask)

def s175_triton(a, b, inc):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    # Calculate number of elements to process
    num_elements = n - 1
    
    # Calculate grid size based on strided access pattern
    grid = (triton.cdiv(triton.cdiv(num_elements, inc), BLOCK_SIZE),)
    
    s175_kernel[grid](a, b, inc, n, BLOCK_SIZE)