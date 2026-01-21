import triton
import triton.language as tl
import torch

@triton.jit
def s323_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel processes the coupled recurrence sequentially
    # Each block handles a contiguous segment of the arrays
    
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    
    # Define offsets once
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process elements in this block sequentially
    for local_i in range(BLOCK_SIZE):
        global_i = block_start + local_i + 1  # Start from index 1
        
        if global_i < n_elements:
            # Load required values
            b_prev = tl.load(b_ptr + global_i - 1)
            c_val = tl.load(c_ptr + global_i)
            d_val = tl.load(d_ptr + global_i)
            e_val = tl.load(e_ptr + global_i)
            
            # Compute a[i] = b[i-1] + c[i] * d[i]
            a_val = b_prev + c_val * d_val
            tl.store(a_ptr + global_i, a_val)
            
            # Compute b[i] = a[i] + c[i] * e[i]
            b_val = a_val + c_val * e_val
            tl.store(b_ptr + global_i, b_val)

def s323_triton(a, b, c, d, e):
    N = a.shape[0]
    
    # Use a single thread to maintain sequential ordering
    BLOCK_SIZE = 1
    grid = (triton.cdiv(N - 1, BLOCK_SIZE),)
    
    s323_kernel[grid](
        a, b, c, d, e,
        N,
        BLOCK_SIZE=BLOCK_SIZE
    )