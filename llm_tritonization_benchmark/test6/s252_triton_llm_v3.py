import triton
import triton.language as tl
import torch

@triton.jit
def s252_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel implements a sequential scan operation
    # Each program processes one block sequentially
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n_elements
    
    # Load the block data
    b_vals = tl.load(b_ptr + block_start + offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + block_start + offsets, mask=mask, other=0.0)
    
    # Initialize accumulator
    t = 0.0
    
    # Process each element in the block sequentially
    for i in range(BLOCK_SIZE):
        if block_start + i < n_elements:
            s = b_vals[i] * c_vals[i]
            a_val = s + t
            tl.store(a_ptr + block_start + i, a_val)
            t = s

def s252_triton(a, b, c):
    n_elements = a.shape[0]
    
    # Use a small block size to maintain some parallelism while respecting dependencies
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s252_kernel[grid](
        a, b, c,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )