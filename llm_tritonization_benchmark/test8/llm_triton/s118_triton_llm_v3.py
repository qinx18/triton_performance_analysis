import torch
import triton
import triton.language as tl

@triton.jit
def s118_kernel(a_ptr, bb_ptr, i_val, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Define offsets once at the start
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process j values in blocks
    block_start = pid * BLOCK_SIZE
    j_offsets = block_start + offsets
    
    # Mask for valid j values: j <= i_val - 1
    j_mask = j_offsets <= (i_val - 1)
    
    # Load bb[j][i_val] values
    bb_indices = j_offsets * LEN_2D + i_val
    bb_vals = tl.load(bb_ptr + bb_indices, mask=j_mask, other=0.0)
    
    # Load a[i_val - j - 1] values
    a_indices = i_val - j_offsets - 1
    a_valid_mask = j_mask & (a_indices >= 0) & (a_indices < LEN_2D)
    a_vals = tl.load(a_ptr + a_indices, mask=a_valid_mask, other=0.0)
    
    # Compute products
    products = bb_vals * a_vals
    
    # Sum all products in this block
    block_sum = tl.sum(products)
    
    # Store the partial sum (will be accumulated across blocks)
    if pid == 0:
        # First block: initialize with current sum
        tl.atomic_add(a_ptr + i_val, block_sum)
    else:
        # Other blocks: add to accumulator
        tl.atomic_add(a_ptr + i_val, block_sum)

def s118_triton(a, bb):
    LEN_2D = bb.shape[0]
    BLOCK_SIZE = 256
    
    for i_val in range(1, LEN_2D):
        # Number of j values: 0 to i_val-1, so i_val values total
        num_j = i_val
        grid = (triton.cdiv(num_j, BLOCK_SIZE),)
        
        s118_kernel[grid](
            a, bb, i_val, LEN_2D, BLOCK_SIZE
        )