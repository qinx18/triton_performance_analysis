import torch
import triton
import triton.language as tl

@triton.jit
def s118_kernel(a_ptr, bb_ptr, i_val, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    mask = j_offsets < i_val
    
    # Load bb[j][i] values
    bb_offsets = j_offsets * 256 + i_val  # bb is 256x256, row-major
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=mask, other=0.0)
    
    # Load a[i-j-1] values
    a_indices = i_val - j_offsets - 1
    a_vals = tl.load(a_ptr + a_indices, mask=mask, other=0.0)
    
    # Compute products and sum
    products = bb_vals * a_vals
    result = tl.sum(products)
    
    # Add to a[i]
    old_val = tl.load(a_ptr + i_val)
    tl.store(a_ptr + i_val, old_val + result)

def s118_triton(a, bb):
    N = bb.shape[0]  # LEN_2D
    BLOCK_SIZE = 256
    
    for i in range(1, N):
        if i > 0:  # Only launch kernel if there are j values to process
            grid = (1,)
            s118_kernel[grid](a, bb, i, BLOCK_SIZE=BLOCK_SIZE)