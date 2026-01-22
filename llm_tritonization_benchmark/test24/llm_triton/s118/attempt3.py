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
    products = tl.where(a_mask, bb_vals * a_vals, 0.0)
    
    # Sum all products in this block
    block_sum = tl.sum(products)
    
    # Store the sum to a temporary location per block
    temp_ptr = a_ptr + n_dim + tl.program_id(0)
    tl.store(temp_ptr, block_sum)

@triton.jit
def s118_reduce_kernel(a_ptr, temp_ptr, i_val, n_blocks, BLOCK_SIZE: tl.constexpr):
    # Sum all partial sums from different blocks
    block_offsets = tl.arange(0, BLOCK_SIZE)
    mask = block_offsets < n_blocks
    partial_sums = tl.load(temp_ptr + block_offsets, mask=mask, other=0.0)
    total_sum = tl.sum(partial_sums)
    
    # Add to a[i]
    if tl.program_id(0) == 0:
        old_val = tl.load(a_ptr + i_val)
        tl.store(a_ptr + i_val, old_val + total_sum)

def s118_triton(a, bb):
    N = bb.shape[0]  # LEN_2D
    
    BLOCK_SIZE = 256
    
    # Allocate temporary storage for partial sums
    max_blocks = triton.cdiv(N, BLOCK_SIZE)
    temp = torch.zeros(max_blocks, dtype=a.dtype, device=a.device)
    
    # Process each i sequentially to maintain dependencies
    for i in range(1, N):
        num_j = i  # j ranges from 0 to i-1
        if num_j > 0:
            num_blocks = triton.cdiv(num_j, BLOCK_SIZE)
            
            # Compute partial sums
            grid = (num_blocks,)
            s118_kernel[grid](a, bb, i, N, BLOCK_SIZE=BLOCK_SIZE)
            
            # Reduce partial sums
            grid = (1,)
            s118_reduce_kernel[grid](a, a.data_ptr() + N * a.element_size(), i, num_blocks, BLOCK_SIZE=BLOCK_SIZE)