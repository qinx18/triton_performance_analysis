import triton
import triton.language as tl
import torch

@triton.jit
def s221_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel must process sequentially due to b[i] = b[i-1] + ... dependency
    # Use a single thread to process all elements sequentially
    
    pid = tl.program_id(axis=0)
    if pid != 0:
        return
    
    # Process all elements sequentially from i=1 to n_elements-1
    for i in range(1, n_elements):
        # Load scalar values
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        a_val = tl.load(a_ptr + i)
        b_prev_val = tl.load(b_ptr + i - 1)
        
        # Compute: a[i] += c[i] * d[i]
        a_new = a_val + c_val * d_val
        tl.store(a_ptr + i, a_new)
        
        # Compute: b[i] = b[i - 1] + a[i] + d[i]
        b_new = b_prev_val + a_new + d_val
        tl.store(b_ptr + i, b_new)

def s221_triton(a, b, c, d):
    n_elements = a.shape[0]
    
    # Use BLOCK_SIZE=1 since we need sequential processing
    BLOCK_SIZE = 1
    
    # Launch with single thread block
    grid = (1,)
    
    s221_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )