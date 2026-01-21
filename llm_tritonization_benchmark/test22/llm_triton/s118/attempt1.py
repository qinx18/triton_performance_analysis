import torch
import triton
import triton.language as tl

@triton.jit
def s118_kernel(a_ptr, bb_ptr, i_val, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    mask = j_offsets < i_val
    
    # Load bb[j][i] values
    bb_offsets = j_offsets * N + i_val
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=mask, other=0.0)
    
    # Load a[i-j-1] values
    a_read_offsets = i_val - j_offsets - 1
    a_vals = tl.load(a_ptr + a_read_offsets, mask=mask, other=0.0)
    
    # Compute products and sum
    products = bb_vals * a_vals
    result = tl.sum(products)
    
    # Add to a[i]
    if tl.program_id(0) == 0:
        current_val = tl.load(a_ptr + i_val)
        tl.store(a_ptr + i_val, current_val + result)

def s118_triton(a, bb):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    for i in range(1, N):
        if i > 0:
            grid = (1,)
            s118_kernel[grid](a, bb, i, N=N, BLOCK_SIZE=BLOCK_SIZE)