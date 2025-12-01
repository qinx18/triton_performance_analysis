import triton
import triton.language as tl
import torch

@triton.jit
def s118_kernel(a_ptr, bb_ptr, i_val, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get program ID and create offset vector once
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    j_offsets = pid * BLOCK_SIZE + offsets
    
    # Mask for valid j values: j <= i_val - 1
    mask = j_offsets <= i_val - 1
    
    # Load bb[j][i_val] values
    bb_indices = j_offsets * LEN_2D + i_val
    bb_vals = tl.load(bb_ptr + bb_indices, mask=mask, other=0.0)
    
    # Load a[i_val - j - 1] values
    a_read_indices = i_val - j_offsets - 1
    a_vals = tl.load(a_ptr + a_read_indices, mask=mask, other=0.0)
    
    # Compute products
    products = bb_vals * a_vals
    
    # Mask out invalid elements
    products = tl.where(mask, products, 0.0)
    
    # Sum across the block
    partial_sum = tl.sum(products)
    
    # Use atomic add to accumulate to a[i_val]
    if pid == 0:  # Only first block does the atomic add to avoid duplicates
        # Sum all valid products in this block and add to a[i_val]
        tl.atomic_add(a_ptr + i_val, partial_sum)
    else:
        # For other blocks, we need to store partial results and sum them
        # This is handled by launching the kernel appropriately
        pass

@triton.jit  
def s118_reduction_kernel(a_ptr, bb_ptr, partial_sums_ptr, i_val, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    j_offsets = pid * BLOCK_SIZE + offsets
    
    mask = j_offsets <= i_val - 1
    
    bb_indices = j_offsets * LEN_2D + i_val
    bb_vals = tl.load(bb_ptr + bb_indices, mask=mask, other=0.0)
    
    a_read_indices = i_val - j_offsets - 1
    a_vals = tl.load(a_ptr + a_read_indices, mask=mask, other=0.0)
    
    products = bb_vals * a_vals
    products = tl.where(mask, products, 0.0)
    
    partial_sum = tl.sum(products)
    
    # Store partial sum for this block
    tl.store(partial_sums_ptr + pid, partial_sum)

def s118_triton(a, bb):
    LEN_2D = bb.shape[0]
    BLOCK_SIZE = 256
    
    # Sequential loop over i
    for i_val in range(1, LEN_2D):
        max_j = i_val  # j ranges from 0 to i-1, so i elements total
        num_blocks = triton.cdiv(max_j, BLOCK_SIZE)
        
        if num_blocks == 1:
            # Simple case - single block
            grid = (1,)
            s118_kernel[grid](
                a, bb, i_val, LEN_2D, BLOCK_SIZE
            )
        else:
            # Multiple blocks - need reduction
            partial_sums = torch.zeros(num_blocks, device=a.device, dtype=a.dtype)
            
            grid = (num_blocks,)
            s118_reduction_kernel[grid](
                a, bb, partial_sums, i_val, LEN_2D, BLOCK_SIZE
            )
            
            # Sum partial results and add to a[i_val]
            total_sum = partial_sums.sum()
            a[i_val] += total_sum