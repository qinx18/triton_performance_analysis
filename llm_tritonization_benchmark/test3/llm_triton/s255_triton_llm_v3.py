import torch
import triton
import triton.language as tl

@triton.jit
def s255_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load all b values needed for computation
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    # Initialize carry variables
    x = tl.load(b_ptr + n_elements - 1)  # b[LEN_1D-1]
    y = tl.load(b_ptr + n_elements - 2)  # b[LEN_1D-2]
    
    # Compute sequentially within each block
    for i in range(BLOCK_SIZE):
        if block_start + i < n_elements:
            curr_b = tl.load(b_ptr + block_start + i)
            result = (curr_b + x + y) * 0.333
            tl.store(a_ptr + block_start + i, result)
            y = x
            x = curr_b

def s255_triton(a, b):
    n_elements = a.shape[0]
    
    # Use small block size for sequential processing
    BLOCK_SIZE = 32
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s255_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )