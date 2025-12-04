import torch
import triton
import triton.language as tl

@triton.jit
def s321_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This is a first-order linear recurrence: a[i] += a[i-1] * b[i]
    # Cannot be parallelized due to dependency, must compute sequentially
    
    # Process all elements sequentially in blocks
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(1, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load current values
        a_vals = tl.load(a_ptr + current_offsets, mask=mask)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        
        # For recurrence, we need to handle dependencies within the block
        # Load the previous value
        prev_offsets = current_offsets - 1
        prev_mask = (prev_offsets >= 0) & mask
        a_prev_vals = tl.load(a_ptr + prev_offsets, mask=prev_mask)
        
        # Compute recurrence step by step within block
        for i in range(BLOCK_SIZE):
            if block_start + i < n_elements:
                # For first element in block, use the loaded previous value
                if i == 0:
                    if block_start > 1:
                        prev_val = tl.load(a_ptr + block_start - 1)
                    else:
                        prev_val = tl.load(a_ptr + 0)  # a[0]
                else:
                    # Use the updated value from previous iteration
                    prev_val = tl.load(a_ptr + block_start + i - 1)
                
                curr_b = tl.load(b_ptr + block_start + i)
                curr_a = tl.load(a_ptr + block_start + i)
                
                new_val = curr_a + prev_val * curr_b
                tl.store(a_ptr + block_start + i, new_val)

def s321_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Launch with single program since this is a sequential recurrence
    grid = (1,)
    
    s321_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )