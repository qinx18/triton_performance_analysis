import triton
import triton.language as tl
import torch

@triton.jit
def s322_kernel(a_ptr, b_ptr, c_ptr, N, BLOCK_SIZE: tl.constexpr):
    # This kernel processes the entire computation sequentially
    # since there's a strict loop-carried dependency
    
    # Use single thread to process sequentially
    pid = tl.program_id(0)
    if pid > 0:
        return
    
    # Process elements sequentially from i=2 to N-1
    for i in range(2, N):
        # Load current values
        a_i = tl.load(a_ptr + i)
        a_i_minus_1 = tl.load(a_ptr + i - 1)
        a_i_minus_2 = tl.load(a_ptr + i - 2)
        b_i = tl.load(b_ptr + i)
        c_i = tl.load(c_ptr + i)
        
        # Compute new value
        new_val = a_i + a_i_minus_1 * b_i + a_i_minus_2 * c_i
        
        # Store result
        tl.store(a_ptr + i, new_val)

def s322_triton(a, b, c):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    # Launch with single block since computation must be sequential
    grid = (1,)
    s322_kernel[grid](a, b, c, N, BLOCK_SIZE=BLOCK_SIZE)