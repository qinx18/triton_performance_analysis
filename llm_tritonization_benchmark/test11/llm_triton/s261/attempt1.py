import torch
import triton
import triton.language as tl

@triton.jit
def s261_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel must be serial due to dependencies
    # Each thread block processes one element sequentially
    pid = tl.program_id(0)
    
    if pid == 0:  # Only one block processes the entire array
        for i in range(1, n_elements):
            # Load values
            t = tl.load(a_ptr + i) + tl.load(b_ptr + i)
            
            # Store first result
            tl.store(a_ptr + i, t + tl.load(c_ptr + (i - 1)))
            
            # Compute second part
            t = tl.load(c_ptr + i) * tl.load(d_ptr + i)
            
            # Store second result
            tl.store(c_ptr + i, t)

def s261_triton(a, b, c, d):
    n_elements = a.shape[0]
    
    # Use only one thread block since computation must be serial
    BLOCK_SIZE = 1
    grid = (1,)
    
    s261_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )