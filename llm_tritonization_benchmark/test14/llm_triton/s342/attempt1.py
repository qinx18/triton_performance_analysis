import torch
import triton
import triton.language as tl

@triton.jit
def s342_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel handles the packing operation sequentially
    # Each thread block processes the entire array to maintain dependency order
    
    pid = tl.program_id(0)
    if pid > 0:  # Only use first program
        return
    
    offsets = tl.arange(0, BLOCK_SIZE)
    j = -1
    
    # Process array in blocks to handle large arrays
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load current block of a
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        
        # Process each element in the block sequentially
        for i in range(BLOCK_SIZE):
            if block_start + i >= n_elements:
                break
                
            # Load single element to check condition
            elem_offset = block_start + i
            if elem_offset < n_elements:
                a_val = tl.load(a_ptr + elem_offset)
                
                if a_val > 0.0:
                    j += 1
                    # Load from b[j] and store to a[i]
                    if j < n_elements:  # Safety check
                        b_val = tl.load(b_ptr + j)
                        tl.store(a_ptr + elem_offset, b_val)

def s342_triton(a, b):
    n_elements = a.numel()
    BLOCK_SIZE = 1024
    
    # Launch single program to maintain sequential dependency
    grid = (1,)
    
    s342_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a