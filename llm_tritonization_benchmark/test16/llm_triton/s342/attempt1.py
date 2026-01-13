import torch
import triton
import triton.language as tl

@triton.jit
def s342_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Sequential processing required due to data dependency
    if block_id == 0:
        j = -1
        for i in range(n_elements):
            a_val = tl.load(a_ptr + i)
            if a_val > 0.0:
                j += 1
                b_val = tl.load(b_ptr + j)
                tl.store(a_ptr + i, b_val)

def s342_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (1,)  # Single block due to sequential dependency
    
    s342_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )