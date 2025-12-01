import triton
import triton.language as tl
import torch

@triton.jit
def s221_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Sequential computation - cannot parallelize due to dependency b[i] = b[i-1] + ...
    # Process one element at a time from i=1 to n_elements-1
    
    for i in range(1, n_elements):
        # Load current elements
        a_val = tl.load(a_ptr + i)
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        b_prev = tl.load(b_ptr + i - 1)
        
        # First statement: a[i] += c[i] * d[i]
        new_a = a_val + c_val * d_val
        tl.store(a_ptr + i, new_a)
        
        # Second statement: b[i] = b[i-1] + a[i] + d[i]
        new_b = b_prev + new_a + d_val
        tl.store(b_ptr + i, new_b)

def s221_triton(a, b, c, d):
    n_elements = a.shape[0]
    
    # Use single thread since computation is sequential
    BLOCK_SIZE = 1
    grid = (1,)
    
    s221_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )