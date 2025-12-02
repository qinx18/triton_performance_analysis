import torch
import triton
import triton.language as tl

@triton.jit
def s261_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for i in range(1, n_elements):
        if pid == 0:
            # Load values
            a_val = tl.load(a_ptr + i)
            b_val = tl.load(b_ptr + i)
            c_prev_val = tl.load(c_ptr + i - 1)
            c_val = tl.load(c_ptr + i)
            d_val = tl.load(d_ptr + i)
            
            # Compute
            t = a_val + b_val
            new_a = t + c_prev_val
            t = c_val * d_val
            new_c = t
            
            # Store results
            tl.store(a_ptr + i, new_a)
            tl.store(c_ptr + i, new_c)

def s261_triton(a, b, c, d):
    n_elements = a.shape[0]
    BLOCK_SIZE = 128
    
    # Launch with single thread block since we need sequential execution
    grid = (1,)
    
    s261_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a, c