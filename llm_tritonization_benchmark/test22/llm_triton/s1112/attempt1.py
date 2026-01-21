import triton
import triton.language as tl
import torch

@triton.jit
def s1112_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Reverse the indices to process from high to low
        reversed_offsets = n_elements - 1 - current_offsets
        reversed_mask = reversed_offsets >= 0
        final_mask = mask & reversed_mask
        
        # Load b values at reversed positions
        b_vals = tl.load(b_ptr + reversed_offsets, mask=final_mask)
        
        # Add 1.0 to b values
        result = b_vals + 1.0
        
        # Store to a at reversed positions
        tl.store(a_ptr + reversed_offsets, result, mask=final_mask)

def s1112_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    s1112_kernel[grid](a, b, n_elements, BLOCK_SIZE=BLOCK_SIZE)