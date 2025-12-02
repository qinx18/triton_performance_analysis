import torch
import triton
import triton.language as tl

@triton.jit
def s118_kernel(a_ptr, bb_ptr, len_2d, i_val, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # j ranges from 0 to i_val-1
    j_mask = j_offsets < i_val
    
    # Load bb[j, i_val] values
    bb_indices = j_offsets * len_2d + i_val
    bb_vals = tl.load(bb_ptr + bb_indices, mask=j_mask, other=0.0)
    
    # Load a[i_val - j - 1] values
    a_read_indices = i_val - j_offsets - 1
    a_vals = tl.load(a_ptr + a_read_indices, mask=j_mask, other=0.0)
    
    # Compute products and sum
    products = bb_vals * a_vals
    result = tl.sum(products, axis=0)
    
    # Atomically add to a[i_val]
    tl.atomic_add(a_ptr + i_val, result)

def s118_triton(a, bb):
    len_2d = bb.shape[0]
    BLOCK_SIZE = 256
    
    for i in range(1, len_2d):
        if i > 0:  # Only launch kernel if there are j values to process
            grid = (triton.cdiv(i, BLOCK_SIZE),)
            s118_kernel[grid](a, bb, len_2d, i, BLOCK_SIZE=BLOCK_SIZE)