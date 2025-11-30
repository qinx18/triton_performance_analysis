import torch
import triton
import triton.language as tl

@triton.jit
def s118_kernel(
    a_ptr,
    bb_ptr,
    i_val,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    j_start = pid * BLOCK_SIZE
    j_offsets = j_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid j values: j <= i - 1
    j_mask = j_offsets < i_val
    
    # Load bb[j][i] values
    bb_offsets = j_offsets * LEN_2D + i_val
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask, other=0.0)
    
    # Load a[i-j-1] values
    a_indices = i_val - j_offsets - 1
    a_vals = tl.load(a_ptr + a_indices, mask=j_mask, other=0.0)
    
    # Compute products
    products = bb_vals * a_vals
    
    # Sum all products in this block
    block_sum = tl.sum(products)
    
    # Store the partial sum (use atomic add to accumulate across blocks)
    if pid == 0:
        # Initialize with 0 for the first block
        current_val = tl.load(a_ptr + i_val)
        tl.store(a_ptr + i_val, current_val + block_sum)
    else:
        # Atomic add for subsequent blocks
        tl.atomic_add(a_ptr + i_val, block_sum)

def s118_triton(a, bb):
    LEN_2D = bb.shape[0]
    BLOCK_SIZE = 256
    
    for i_val in range(1, LEN_2D):
        num_j = i_val  # j ranges from 0 to i-1
        if num_j == 0:
            continue
            
        grid = (triton.cdiv(num_j, BLOCK_SIZE),)
        s118_kernel[grid](
            a,
            bb,
            i_val,
            LEN_2D,
            BLOCK_SIZE,
        )