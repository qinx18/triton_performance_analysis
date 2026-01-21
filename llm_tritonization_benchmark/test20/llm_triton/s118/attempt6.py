import triton
import triton.language as tl
import torch

@triton.jit
def s118_kernel(a_ptr, bb_ptr, i_val, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get j indices for this block
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid j values (j <= i-1)
    j_mask = j_offsets <= (i_val - 1)
    
    # Load bb[j][i] values
    bb_offsets = j_offsets * LEN_2D + i_val
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask, other=0.0)
    
    # Load a[i-j-1] values
    a_read_indices = i_val - j_offsets - 1
    a_read_mask = j_mask & (a_read_indices >= 0)
    a_vals = tl.load(a_ptr + a_read_indices, mask=a_read_mask, other=0.0)
    
    # Compute products
    products = bb_vals * a_vals
    
    # Sum the products
    result = tl.sum(tl.where(a_read_mask, products, 0.0))
    
    # Add to a[i]
    if tl.program_id(0) == 0:  # Only one block should update a[i]
        old_val = tl.load(a_ptr + i_val)
        tl.store(a_ptr + i_val, old_val + result)

def s118_triton(a, bb):
    LEN_2D = bb.shape[0]
    BLOCK_SIZE = triton.next_power_of_2(LEN_2D)
    
    # Sequential loop over i
    for i in range(1, LEN_2D):
        grid = (1,)  # Single block since we're processing one i at a time
        s118_kernel[grid](
            a, bb, i, LEN_2D=LEN_2D, BLOCK_SIZE=BLOCK_SIZE
        )