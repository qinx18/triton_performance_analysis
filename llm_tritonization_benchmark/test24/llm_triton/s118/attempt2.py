import triton
import triton.language as tl
import torch

@triton.jit
def s118_kernel(a_ptr, bb_ptr, i_val, n_dim, BLOCK_SIZE: tl.constexpr):
    # Get current block of j indices
    block_start = tl.program_id(0) * BLOCK_SIZE
    j_offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid j values (0 <= j <= i-1)
    mask = (j_offsets >= 0) & (j_offsets < i_val)
    
    # Load bb[j][i] values - bb is stored in row-major order
    bb_offsets = j_offsets * n_dim + i_val
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=mask, other=0.0)
    
    # Load a[i-j-1] values
    a_indices = i_val - j_offsets - 1
    a_mask = mask & (a_indices >= 0) & (a_indices < n_dim)
    a_vals = tl.load(a_ptr + a_indices, mask=a_mask, other=0.0)
    
    # Compute products and sum
    products = bb_vals * a_vals
    # Apply mask to zero out invalid contributions
    products = tl.where(a_mask, products, 0.0)
    
    # Sum all products in this block
    block_sum = tl.sum(products)
    
    # Use atomic add to accumulate results
    tl.atomic_add(a_ptr + i_val, block_sum)

def s118_triton(a, bb):
    N = bb.shape[0]  # LEN_2D
    
    BLOCK_SIZE = 256
    
    # Process each i sequentially to maintain dependencies
    for i in range(1, N):
        num_j = i  # j ranges from 0 to i-1
        if num_j > 0:
            grid = (triton.cdiv(num_j, BLOCK_SIZE),)
            s118_kernel[grid](a, bb, i, N, BLOCK_SIZE=BLOCK_SIZE)