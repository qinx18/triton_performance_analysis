import triton
import triton.language as tl
import torch

@triton.jit
def s118_kernel(a_ptr, bb_ptr, i_val, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    mask = j_offsets < i_val
    
    # Load bb[j][i] values
    bb_offsets = j_offsets * 256 + i_val  # bb is 256x256, row-major
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=mask)
    
    # Load a[i-j-1] values
    a_read_offsets = i_val - j_offsets - 1
    a_vals = tl.load(a_ptr + a_read_offsets, mask=mask)
    
    # Compute products and sum
    products = bb_vals * a_vals
    reduction = tl.sum(products, axis=0)
    
    return reduction

def s118_triton(a, bb):
    N = bb.shape[0]  # LEN_2D dimension
    BLOCK_SIZE = 256
    
    for i in range(1, N):
        if i > 0:  # Only launch kernel if there are j values to process
            grid = (1,)
            reduction = s118_kernel[grid](
                a, bb, i, BLOCK_SIZE=BLOCK_SIZE
            )
            # Add the reduction to a[i]
            a[i] += reduction