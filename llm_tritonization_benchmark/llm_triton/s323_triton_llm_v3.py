import triton
import triton.language as tl
import torch

@triton.jit
def s323_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel handles the coupled recurrence sequentially
    # Each block processes a contiguous chunk of elements
    block_start = tl.program_id(0) * BLOCK_SIZE
    
    # Process elements sequentially within each block
    for i in range(BLOCK_SIZE):
        idx = block_start + i + 1  # Start from 1, not 0
        
        if idx < n_elements:
            # Load required values
            b_prev = tl.load(b_ptr + idx - 1)
            c_val = tl.load(c_ptr + idx)
            d_val = tl.load(d_ptr + idx)
            e_val = tl.load(e_ptr + idx)
            
            # Compute a[i] = b[i-1] + c[i] * d[i]
            a_val = b_prev + c_val * d_val
            tl.store(a_ptr + idx, a_val)
            
            # Compute b[i] = a[i] + c[i] * e[i]
            b_val = a_val + c_val * e_val
            tl.store(b_ptr + idx, b_val)

def s323_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    
    # Use small block size to minimize overhead while maintaining sequential processing
    BLOCK_SIZE = 128
    grid = ((n_elements - 1 + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    s323_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a, b