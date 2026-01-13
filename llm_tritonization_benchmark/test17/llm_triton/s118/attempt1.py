import torch
import triton
import triton.language as tl

@triton.jit
def s118_kernel(a_ptr, bb_ptr, i_val, len_2d: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < i_val
    
    # Load bb[j][i_val] for all valid j
    bb_offsets = j_offsets * len_2d + i_val
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask, other=0.0)
    
    # Load a[i_val-j-1] for all valid j
    a_indices = i_val - j_offsets - 1
    a_vals = tl.load(a_ptr + a_indices, mask=j_mask, other=0.0)
    
    # Compute products and sum
    products = bb_vals * a_vals
    result = tl.sum(products)
    
    # Add to a[i_val]
    if tl.program_id(0) == 0:
        current_val = tl.load(a_ptr + i_val)
        tl.store(a_ptr + i_val, current_val + result)

def s118_triton(a, bb):
    len_2d = bb.shape[0]
    BLOCK_SIZE = triton.next_power_of_2(len_2d)
    
    # Sequential loop over i, parallel processing of j values
    for i in range(1, len_2d):
        s118_kernel[(1,)](
            a, bb,
            i, len_2d,
            BLOCK_SIZE=BLOCK_SIZE
        )