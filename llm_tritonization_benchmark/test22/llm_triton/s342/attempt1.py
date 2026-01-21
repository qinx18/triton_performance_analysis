import triton
import triton.language as tl
import torch

@triton.jit
def s342_kernel(a_ptr, b_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n
    
    # Load a values for this block
    a_offsets = block_start + offsets
    a_vals = tl.load(a_ptr + a_offsets, mask=mask, other=0.0)
    
    # Process each element in the block sequentially
    j = -1
    for i in range(BLOCK_SIZE):
        if block_start + i >= n:
            break
        
        # Load single element from a
        idx = block_start + i
        a_val = tl.load(a_ptr + idx)
        
        if a_val > 0.0:
            j = j + 1
            # Load from b[j] and store to a[i]
            b_val = tl.load(b_ptr + j)
            tl.store(a_ptr + idx, b_val)

def s342_triton(a, b):
    n = a.shape[0]
    BLOCK_SIZE = 1024
    
    # Launch kernel with sequential processing per block
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    s342_kernel[grid](a, b, n, BLOCK_SIZE=BLOCK_SIZE)
    
    return a