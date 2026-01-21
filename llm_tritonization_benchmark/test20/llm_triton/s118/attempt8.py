import triton
import triton.language as tl
import torch

@triton.jit
def s118_kernel(a_ptr, bb_ptr, i_val, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get j indices for this block
    j_base = tl.program_id(0) * BLOCK_SIZE
    j_offsets = j_base + tl.arange(0, BLOCK_SIZE)
    
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
    
    # Sum the products for this block
    partial_sum = tl.sum(tl.where(a_read_mask, products, 0.0))
    
    # Store partial sum to temporary location
    temp_offset = tl.program_id(0)
    tl.store(a_ptr + LEN_2D + temp_offset, partial_sum)

@triton.jit  
def s118_reduce_kernel(a_ptr, i_val, LEN_2D: tl.constexpr, num_blocks: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Sum all partial results and add to a[i]
    block_offsets = tl.arange(0, BLOCK_SIZE)
    mask = block_offsets < num_blocks
    
    partial_sums = tl.load(a_ptr + LEN_2D + block_offsets, mask=mask, other=0.0)
    total_sum = tl.sum(partial_sums)
    
    # Add to a[i]
    if tl.program_id(0) == 0:
        old_val = tl.load(a_ptr + i_val)
        tl.store(a_ptr + i_val, old_val + total_sum)

def s118_triton(a, bb):
    LEN_2D = bb.shape[0]
    BLOCK_SIZE = 256
    
    # Extend a to have space for temporary results
    max_blocks = triton.cdiv(LEN_2D, BLOCK_SIZE)
    extended_a = torch.cat([a, torch.zeros(max_blocks, dtype=a.dtype, device=a.device)])
    
    # Sequential loop over i
    for i in range(1, LEN_2D):
        # Number of j values to process
        num_j = i
        num_blocks = triton.cdiv(num_j, BLOCK_SIZE)
        
        if num_blocks > 0:
            # Launch kernels to compute partial sums
            grid = (num_blocks,)
            s118_kernel[grid](
                extended_a, bb, i, LEN_2D=LEN_2D, BLOCK_SIZE=BLOCK_SIZE
            )
            
            # Reduce partial sums
            reduce_block_size = triton.next_power_of_2(num_blocks)
            grid = (1,)
            s118_reduce_kernel[grid](
                extended_a, i, LEN_2D=LEN_2D, num_blocks=num_blocks, BLOCK_SIZE=reduce_block_size
            )
    
    # Copy result back to original a
    a[:] = extended_a[:LEN_2D]