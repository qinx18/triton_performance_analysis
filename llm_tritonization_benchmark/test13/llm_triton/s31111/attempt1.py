import torch
import triton
import triton.language as tl

@triton.jit
def s31111_kernel(a_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Single thread handles the entire computation
    if tl.program_id(0) != 0:
        return
    
    # Compute sum of test() calls on different segments of array a
    sum_val = 0.0
    
    # Each test() call sums 4 consecutive elements starting at different offsets
    offsets = [0, 4, 8, 12, 16, 20, 24, 28]
    
    for offset in offsets:
        # Load 4 elements starting at offset
        test_offsets = tl.arange(0, 4) + offset
        mask = test_offsets < n_elements
        vals = tl.load(a_ptr + test_offsets, mask=mask, other=0.0)
        test_sum = tl.sum(vals)
        sum_val += test_sum

def s31111_triton(a):
    n_elements = a.numel()
    BLOCK_SIZE = 32
    
    # Launch kernel with single thread
    grid = (1,)
    s31111_kernel[grid](
        a, n_elements, BLOCK_SIZE
    )
    
    return a