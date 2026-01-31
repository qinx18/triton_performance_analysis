import triton
import triton.language as tl
import torch

@triton.jit
def s118_kernel(a_ptr, bb_ptr, i_val, len_2d, BLOCK_SIZE: tl.constexpr):
    # Get program ID and compute offsets
    pid = tl.program_id(0)
    j_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid j values: j <= i-1
    j_mask = j_offsets < i_val
    
    # Load bb[j][i] values
    bb_offsets = j_offsets * len_2d + i_val
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask, other=0.0)
    
    # Load a[i-j-1] values
    a_read_indices = i_val - j_offsets - 1
    a_read_mask = j_mask & (a_read_indices >= 0)
    a_vals = tl.load(a_ptr + a_read_indices, mask=a_read_mask, other=0.0)
    
    # Compute products
    products = bb_vals * a_vals
    
    # Apply final mask and sum
    final_mask = j_mask & a_read_mask
    masked_products = tl.where(final_mask, products, 0.0)
    result = tl.sum(masked_products)
    
    # Store result back to a[i] (atomic add for safety)
    if pid == 0:
        tl.atomic_add(a_ptr + i_val, result)

def s118_triton(a, bb, len_2d):
    BLOCK_SIZE = 128
    
    # Sequential loop over i
    for i in range(1, len_2d):
        num_j = i  # j goes from 0 to i-1, so i total values
        grid = (triton.cdiv(num_j, BLOCK_SIZE),)
        
        s118_kernel[grid](
            a, bb, i, len_2d, BLOCK_SIZE
        )