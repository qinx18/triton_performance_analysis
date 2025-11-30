import triton
import triton.language as tl
import torch

@triton.jit
def s254_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load the carry-around initial value
    x = tl.load(b_ptr + n_elements - 1)
    
    # Process elements in this block sequentially
    for i in range(BLOCK_SIZE):
        idx = block_start + i
        if idx < n_elements:
            b_val = tl.load(b_ptr + idx)
            result = (b_val + x) * 0.5
            tl.store(a_ptr + idx, result)
            x = b_val

def s254_triton(a, b):
    n_elements = a.shape[0]
    
    # Use block size of 1 to maintain sequential dependency
    BLOCK_SIZE = 1
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s254_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )