import triton
import triton.language as tl
import torch

@triton.jit
def s1112_kernel(a_ptr, b_ptr, N, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Convert forward block to reverse indexing
    reverse_offsets = N - 1 - (block_start + offsets)
    
    mask = (block_start + offsets) < N
    
    b_vals = tl.load(b_ptr + reverse_offsets, mask=mask)
    result = b_vals + 1.0
    tl.store(a_ptr + reverse_offsets, result, mask=mask)

def s1112_triton(a, b):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s1112_kernel[grid](
        a, b, N,
        BLOCK_SIZE=BLOCK_SIZE
    )