import torch
import triton
import triton.language as tl

@triton.jit
def s118_kernel(a_ptr, bb_ptr, i_val, LEN_2D, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # j ranges from 0 to i_val-1
    mask = offsets < i_val
    
    # Load bb[j][i] values - bb is row-major: bb[j][i] = bb_ptr[j * LEN_2D + i]
    bb_offsets = offsets * LEN_2D + i_val
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=mask, other=0.0)
    
    # Load a[i-j-1] values
    a_read_offsets = i_val - offsets - 1
    a_vals = tl.load(a_ptr + a_read_offsets, mask=mask, other=0.0)
    
    # Compute bb[j][i] * a[i-j-1]
    products = bb_vals * a_vals
    
    # Sum all products in this block
    partial_sum = tl.sum(products)
    
    # Use atomic add to accumulate the result to a[i]
    tl.atomic_add(a_ptr + i_val, partial_sum)

def s118_triton(a, bb):
    LEN_2D = bb.shape[0]
    BLOCK_SIZE = 256
    
    for i in range(1, LEN_2D):
        num_j = i  # j ranges from 0 to i-1, so there are i values
        grid = (triton.cdiv(num_j, BLOCK_SIZE),)
        s118_kernel[grid](a, bb, i, LEN_2D, BLOCK_SIZE)