import torch
import triton
import triton.language as tl

@triton.jit
def s321_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    for i in range(1, n_elements):
        if i >= block_start and i < block_start + BLOCK_SIZE:
            local_idx = i - block_start
            if local_idx < BLOCK_SIZE:
                prev_val = tl.load(a_ptr + (i-1))
                curr_val = tl.load(a_ptr + i)
                b_val = tl.load(b_ptr + i)
                new_val = curr_val + prev_val * b_val
                tl.store(a_ptr + i, new_val)

def s321_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 1
    grid = (1,)
    
    s321_kernel[grid](
        a, b,
        n_elements,
        BLOCK_SIZE
    )
    
    return a