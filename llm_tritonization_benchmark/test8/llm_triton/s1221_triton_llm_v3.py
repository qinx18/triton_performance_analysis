import torch
import triton
import triton.language as tl

@triton.jit
def s1221_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(4, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load a[i] and b[i-4]
        a_vals = tl.load(a_ptr + current_offsets, mask=mask)
        b_prev_vals = tl.load(b_ptr + current_offsets - 4, mask=mask)
        
        # Compute b[i] = b[i-4] + a[i]
        result = b_prev_vals + a_vals
        
        # Store result
        tl.store(b_ptr + current_offsets, result, mask=mask)

def s1221_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n_elements - 4, BLOCK_SIZE),)
    
    s1221_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )