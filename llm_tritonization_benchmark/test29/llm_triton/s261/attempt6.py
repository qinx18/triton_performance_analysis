import triton
import triton.language as tl
import torch

@triton.jit
def s261_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process blocks sequentially due to loop-carried dependency
    for i in range(1, n):
        # Each thread handles one element
        if tl.program_id(0) == 0 and tl.get_program_id(0) == 0:
            # Load values
            a_val = tl.load(a_ptr + i)
            b_val = tl.load(b_ptr + i)
            c_prev = tl.load(c_ptr + i - 1)
            c_val = tl.load(c_ptr + i)
            d_val = tl.load(d_ptr + i)
            
            # Compute
            t = a_val + b_val
            new_a = t + c_prev
            t = c_val * d_val
            new_c = t
            
            # Store
            tl.store(a_ptr + i, new_a)
            tl.store(c_ptr + i, new_c)

def s261_triton(a, b, c, d):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    # Since we have loop-carried dependency, use single thread approach
    grid = (1,)
    
    s261_kernel[grid](
        a, b, c, d, n,
        BLOCK_SIZE=BLOCK_SIZE
    )