import triton
import triton.language as tl
import torch

@triton.jit
def s323_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel handles the coupled recurrence sequentially
    # Each block processes a contiguous range of elements
    block_start = tl.program_id(0) * BLOCK_SIZE
    
    # Load initial values for this block
    if block_start == 0:
        # First block starts from index 1
        start_idx = 1
    else:
        start_idx = block_start
    
    # Process elements sequentially within this block
    for i in range(start_idx, min(block_start + BLOCK_SIZE, n_elements)):
        if i < n_elements:
            # Load values
            b_prev = tl.load(b_ptr + i - 1)
            c_val = tl.load(c_ptr + i)
            d_val = tl.load(d_ptr + i)
            e_val = tl.load(e_ptr + i)
            
            # Compute coupled recurrence
            a_val = b_prev + c_val * d_val
            b_val = a_val + c_val * e_val
            
            # Store results
            tl.store(a_ptr + i, a_val)
            tl.store(b_ptr + i, b_val)

def s323_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    
    # Use a single thread block to maintain sequential dependency
    BLOCK_SIZE = n_elements - 1  # Process from index 1 to n_elements-1
    grid = (1,)
    
    s323_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a, b