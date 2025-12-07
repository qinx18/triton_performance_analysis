import triton
import triton.language as tl
import torch

@triton.jit
def s221_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel must process sequentially due to strict RAW dependency
    # b[i] depends on b[i-1], so no parallelization possible
    
    # Use single thread to process all elements sequentially
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    # Process elements sequentially from 1 to n_elements-1
    for i in range(1, n_elements):
        # Load elements for position i
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        a_val = tl.load(a_ptr + i)
        
        # Compute a[i] += c[i] * d[i]
        new_a = a_val + c_val * d_val
        tl.store(a_ptr + i, new_a)
        
        # Load b[i-1] and compute b[i] = b[i-1] + a[i] + d[i]
        b_prev = tl.load(b_ptr + i - 1)
        new_b = b_prev + new_a + d_val
        tl.store(b_ptr + i, new_b)

def s221_triton(a, b, c, d):
    n_elements = a.shape[0]
    BLOCK_SIZE = 1024
    
    # Launch with single thread since computation must be sequential
    grid = (1,)
    
    s221_kernel[grid](
        a, b, c, d, n_elements, BLOCK_SIZE
    )