import torch
import triton
import triton.language as tl

@triton.jit
def s3110_kernel(aa_ptr, max_out_ptr, xindex_out_ptr, yindex_out_ptr, 
                 LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < LEN_2D
    
    # Initialize max, xindex, yindex for this block
    max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)
    best_xindex = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    best_yindex = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    
    # Initialize with aa[0][j] values
    init_ptrs = aa_ptr + j_idx
    init_vals = tl.load(init_ptrs, mask=j_mask, other=float('-inf'))
    max_val = tl.where(j_mask, init_vals, max_val)
    best_yindex = tl.where(j_mask, j_idx, best_yindex)
    
    # Sequential loop over i dimension
    for i in range(1, LEN_2D):
        row_ptrs = aa_ptr + i * LEN_2D + j_idx
        vals = tl.load(row_ptrs, mask=j_mask, other=float('-inf'))
        
        # Update max values and indices
        is_greater = vals > max_val
        is_greater = is_greater & j_mask
        
        max_val = tl.where(is_greater, vals, max_val)
        best_xindex = tl.where(is_greater, i, best_xindex)
        best_yindex = tl.where(is_greater, j_idx, best_yindex)
    
    # Find global maximum across this block
    # First, mask out invalid entries
    max_val = tl.where(j_mask, max_val, float('-inf'))
    
    # Find the index of maximum value
    global_max = tl.max(max_val)
    is_global_max = (max_val == global_max) & j_mask
    
    # Get the first occurrence of global max
    lane_id = tl.arange(0, BLOCK_SIZE)
    max_lane = tl.argmax(tl.where(is_global_max, -lane_id, tl.full([BLOCK_SIZE], 1000000, dtype=tl.int32)), 0)
    max_lane = -max_lane
    
    final_max = tl.max(max_val)
    final_xindex = tl.sum(tl.where(lane_id == max_lane, best_xindex, 0))
    final_yindex = tl.sum(tl.where(lane_id == max_lane, best_yindex, 0))
    
    # Store results (only first thread in first block writes)
    if pid == 0:
        if tl.program_id(0) == 0:
            tl.store(max_out_ptr, final_max)
            tl.store(xindex_out_ptr, final_xindex)
            tl.store(yindex_out_ptr, final_yindex)

def s3110_triton(aa):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Output tensors
    max_out = torch.zeros(1, dtype=torch.float32, device=aa.device)
    xindex_out = torch.zeros(1, dtype=torch.int32, device=aa.device)
    yindex_out = torch.zeros(1, dtype=torch.int32, device=aa.device)
    
    # Launch single kernel to find global maximum
    grid = (1,)
    s3110_kernel[grid](
        aa, max_out, xindex_out, yindex_out,
        LEN_2D, BLOCK_SIZE
    )
    
    # Now reduce across all blocks to find true global maximum
    # Since we're using single block, the result is already global
    max_val = max_out.item()
    xindex = xindex_out.item()
    yindex = yindex_out.item()
    
    chksum = max_val + float(xindex) + float(yindex)
    
    return max_val, xindex, yindex, chksum