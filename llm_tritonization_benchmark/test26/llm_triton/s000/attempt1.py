import triton
import triton.language as tl
import torch

@triton.jit
def s000_kernel(
    a_ptr,
    b_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    
    offsets = tl.arange(0, BLOCK_SIZE)
    block_start = pid * BLOCK_SIZE
    current_offsets = block_start + offsets
    
    mask = current_offsets < n_elements
    
    b_vals = tl.load(b_ptr + current_offsets, mask=mask)
    a_vals = b_vals + 1
    
    tl.store(a_ptr + current_offsets, a_vals, mask=mask)

def s000_triton(a, b):
    N = a.shape[0]
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s000_kernel[grid](
        a, b, N,
        BLOCK_SIZE=BLOCK_SIZE,
    )