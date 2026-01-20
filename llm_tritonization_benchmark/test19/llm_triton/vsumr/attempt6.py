import torch
import triton
import triton.language as tl

@triton.jit
def vsumr_kernel(a_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    total_sum = 0.0
    
    for block_start in range(0, n, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        total_sum += tl.sum(vals, axis=0)
    
    return total_sum

def vsumr_triton(a):
    n = a.shape[0]
    BLOCK_SIZE = 1024
    
    result = vsumr_kernel(a, n, BLOCK_SIZE=BLOCK_SIZE)
    return result