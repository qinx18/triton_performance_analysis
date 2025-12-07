import torch
import triton
import triton.language as tl

@triton.jit
def s3110_kernel(aa_ptr, max_out_ptr, xindex_out_ptr, yindex_out_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < LEN_2D
    
    # Initialize with aa[0][0]
    max_val = tl.load(aa_ptr)
    max_i = 0
    max_j = 0
    
    # Sequential loop over i
    for i in range(LEN_2D):
        # Vectorized load for all j values in this block
        row_ptr = aa_ptr + i * LEN_2D + j_idx
        values = tl.load(row_ptr, mask=j_mask, other=float('-inf'))
        
        # Find maximum value in this block for row i
        block_max = tl.max(values, axis=0)
        
        # Check if this block max is greater than current global max
        if block_max > max_val:
            max_val = block_max
            max_i = i
            # Find which j had the maximum value (get the first occurrence)
            max_locations = values == block_max
            max_locations = max_locations & j_mask
            j_candidates = tl.where(max_locations, j_idx, LEN_2D)
            max_j = tl.min(j_candidates)
    
    # Store results (only one thread per block writes)
    lane_id = tl.program_id(1) if tl.num_programs(1) > 1 else 0
    if lane_id == 0:
        tl.store(max_out_ptr + pid, max_val)
        tl.store(xindex_out_ptr + pid, max_i)
        tl.store(yindex_out_ptr + pid, max_j)

@triton.jit
def s3110_reduction_kernel(max_vals_ptr, xindex_vals_ptr, yindex_vals_ptr, 
                          final_max_ptr, final_xindex_ptr, final_yindex_ptr, 
                          num_blocks: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_blocks
    
    max_vals = tl.load(max_vals_ptr + offsets, mask=mask, other=float('-inf'))
    xindex_vals = tl.load(xindex_vals_ptr + offsets, mask=mask, other=0)
    yindex_vals = tl.load(yindex_vals_ptr + offsets, mask=mask, other=0)
    
    # Find global maximum
    global_max = tl.max(max_vals)
    
    # Find which block had the global maximum (first occurrence)
    max_locations = max_vals == global_max
    max_locations = max_locations & mask
    block_candidates = tl.where(max_locations, offsets, num_blocks)
    winning_block = tl.min(block_candidates)
    
    # Get indices from winning block
    final_xindex = tl.load(xindex_vals_ptr + winning_block)
    final_yindex = tl.load(yindex_vals_ptr + winning_block)
    
    tl.store(final_max_ptr, global_max)
    tl.store(final_xindex_ptr, final_xindex)
    tl.store(final_yindex_ptr, final_yindex)

def s3110_triton(aa):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    num_blocks = triton.cdiv(LEN_2D, BLOCK_SIZE)
    
    # Flatten aa for easier access
    aa_flat = aa.contiguous().view(-1)
    
    # Create temporary tensors for partial results
    max_vals = torch.zeros(num_blocks, dtype=aa.dtype, device=aa.device)
    xindex_vals = torch.zeros(num_blocks, dtype=torch.int32, device=aa.device)
    yindex_vals = torch.zeros(num_blocks, dtype=torch.int32, device=aa.device)
    
    # Launch first kernel to find partial maxima
    grid = (num_blocks,)
    s3110_kernel[grid](
        aa_flat, max_vals, xindex_vals, yindex_vals,
        LEN_2D, BLOCK_SIZE
    )
    
    # Create final result tensors
    final_max = torch.zeros(1, dtype=aa.dtype, device=aa.device)
    final_xindex = torch.zeros(1, dtype=torch.int32, device=aa.device)
    final_yindex = torch.zeros(1, dtype=torch.int32, device=aa.device)
    
    # Launch reduction kernel to find global maximum
    reduction_block_size = min(128, triton.next_power_of_2(num_blocks))
    grid = (1,)
    s3110_reduction_kernel[grid](
        max_vals, xindex_vals, yindex_vals,
        final_max, final_xindex, final_yindex,
        num_blocks, reduction_block_size
    )
    
    max_val = final_max.item()
    xindex = final_xindex.item()
    yindex = final_yindex.item()
    
    return max_val + xindex + 1 + yindex + 1