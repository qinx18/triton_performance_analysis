import torch
import triton
import triton.language as tl

@triton.jit
def s291_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Handle first element (i=0)
    if tl.program_id(0) == 0:
        mask_first = offsets == 0
        b_first = tl.load(b_ptr + offsets, mask=mask_first)
        b_last = tl.load(b_ptr + (n_elements - 1), mask=mask_first)
        result_first = (b_first + b_last) * 0.5
        tl.store(a_ptr + offsets, result_first, mask=mask_first)
    
    # Handle remaining elements
    for block_start in range(1, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # For i > 0, im1 = i-1
        prev_offsets = current_offsets - 1
        b_current = tl.load(b_ptr + current_offsets, mask=mask)
        b_prev = tl.load(b_ptr + prev_offsets, mask=mask)
        
        result = (b_current + b_prev) * 0.5
        tl.store(a_ptr + current_offsets, result, mask=mask)

def s291_triton(a, b):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    
    grid = (1,)
    s291_kernel[grid](a, b, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return a