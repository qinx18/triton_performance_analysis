import torch
import triton
import triton.language as tl

@triton.jit
def s254_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Get the last element of b for initial x value
    last_offset = n_elements - 1
    x = tl.load(b_ptr + last_offset)
    
    # Process elements sequentially in blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load current block of b values
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        
        # For each element in the block, compute a[i] = (b[i] + x) * 0.5
        # Then update x = b[i]
        for i in range(BLOCK_SIZE):
            elem_offset = block_start + i
            if elem_offset < n_elements:
                b_val = tl.load(b_ptr + elem_offset)
                a_val = (b_val + x) * 0.5
                tl.store(a_ptr + elem_offset, a_val)
                x = b_val

def s254_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s254_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )