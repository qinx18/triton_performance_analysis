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
    
    # Compute products
    products = bb_vals * a_vals
    
    # Zero out products where mask is false
    products = tl.where(a_mask, products, 0.0)
    
    # Sum all products in this block
    block_sum = tl.sum(products)
    
    # Store the sum at a[i] atomically
    if tl.program_id(0) == 0:
        if block_sum != 0.0:
            old_val = tl.load(a_ptr + i_val)
            tl.store(a_ptr + i_val, old_val + block_sum)

@triton.jit
def s118_final_reduce_kernel(a_ptr, temp_ptr, i_val, n_blocks, BLOCK_SIZE: tl.constexpr):
    # Sum partial results from all blocks except the first one
    block_offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_offsets >= 1) & (block_offsets < n_blocks)
    partial_sums = tl.load(temp_ptr + block_offsets, mask=mask, other=0.0)
    total_sum = tl.sum(partial_sums)
    
    # Add to a[i]
    if total_sum != 0.0:
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
            
            if num_blocks == 1:
                # Single block case
                grid = (1,)
                s118_kernel[grid](a, bb, i, N, BLOCK_SIZE=BLOCK_SIZE)
            else:
                # Multiple blocks case - need reduction
                # First kernel stores partial sums to temp array
                @triton.jit
                def s118_multi_kernel(a_ptr, bb_ptr, temp_ptr, i_val, n_dim, BLOCK_SIZE: tl.constexpr):
                    block_id = tl.program_id(0)
                    block_start = block_id * BLOCK_SIZE
                    j_offsets = block_start + tl.arange(0, BLOCK_SIZE)
                    
                    # Mask for valid j values (0 <= j <= i-1)
                    mask = (j_offsets >= 0) & (j_offsets < i_val)
                    
                    # Load bb[j][i] values
                    bb_offsets = j_offsets * n_dim + i_val
                    bb_vals = tl.load(bb_ptr + bb_offsets, mask=mask, other=0.0)
                    
                    # Load a[i-j-1] values
                    a_indices = i_val - j_offsets - 1
                    a_mask = mask & (a_indices >= 0) & (a_indices < n_dim)
                    a_vals = tl.load(a_ptr + a_indices, mask=a_mask, other=0.0)
                    
                    # Compute products
                    products = bb_vals * a_vals
                    products = tl.where(a_mask, products, 0.0)
                    
                    # Sum and store partial result
                    block_sum = tl.sum(products)
                    tl.store(temp_ptr + block_id, block_sum)
                
                grid = (num_blocks,)
                s118_multi_kernel[grid](a, bb, temp, i, N, BLOCK_SIZE=BLOCK_SIZE)
                
                # Reduce all partial sums
                grid = (1,)
                s118_final_reduce_kernel[grid](a, temp, i, num_blocks, BLOCK_SIZE=BLOCK_SIZE)