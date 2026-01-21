import triton
import triton.language as tl
import torch

@triton.jit
def s323_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Process sequentially with one element per block
    block_id = tl.program_id(0)
    i = block_id + 1  # Start from index 1
    
    if i < n_elements:
        # Load required values
        b_prev = tl.load(b_ptr + i - 1)
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        e_val = tl.load(e_ptr + i)
        
        # Compute a[i] = b[i-1] + c[i] * d[i]
        a_val = b_prev + c_val * d_val
        tl.store(a_ptr + i, a_val)
        
        # Compute b[i] = a[i] + c[i] * e[i]
        b_val = a_val + c_val * e_val
        tl.store(b_ptr + i, b_val)

def s323_triton(a, b, c, d, e):
    N = a.shape[0]
    
    # Launch N-1 blocks, each processing one element
    BLOCK_SIZE = 1
    grid = (N - 1,)
    
    s323_kernel[grid](
        a, b, c, d, e,
        N,
        BLOCK_SIZE=BLOCK_SIZE
    )