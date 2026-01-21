import torch
import triton
import triton.language as tl

@triton.jit
def s3110_kernel(aa_ptr, result_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < N
    
    # Initialize block values
    block_max = tl.full((BLOCK_SIZE,), float('-inf'), dtype=tl.float32)
    block_xindex = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    block_yindex = j_idx
    
    # Sequential loop over i dimension
    for i in range(N):
        # Load row i for all valid j positions
        row_offsets = i * N + j_idx
        vals = tl.load(aa_ptr + row_offsets, mask=j_mask, other=float('-inf'))
        
        # Update max values and indices where new value is greater
        update_mask = vals > block_max
        update_mask = update_mask & j_mask
        block_max = tl.where(update_mask, vals, block_max)
        block_xindex = tl.where(update_mask, i, block_xindex)
    
    # Find the maximum value in this block
    max_val = tl.max(block_max)
    
    # Find elements that have the maximum value
    is_max_mask = block_max == max_val
    is_max_mask = is_max_mask & j_mask
    
    # Among tied maximum values, find the one with smallest indices
    # Set non-max elements to large values so they don't interfere with min operations
    temp_xindex = tl.where(is_max_mask, block_xindex, N)
    temp_yindex = tl.where(is_max_mask, block_yindex, N)
    
    final_xindex = tl.min(temp_xindex)
    final_yindex = tl.min(temp_yindex)
    
    # Only thread 0 of block 0 writes the result
    if pid == 0:
        lane_id = tl.arange(0, BLOCK_SIZE)[0]
        if lane_id == 0:
            tl.store(result_ptr, max_val)
            tl.store(result_ptr + 1, final_xindex.to(tl.float32))
            tl.store(result_ptr + 2, final_yindex.to(tl.float32))

def s3110_triton(aa):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Allocate result tensor: [max_val, xindex, yindex]
    result = torch.zeros(3, dtype=torch.float32, device=aa.device)
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    s3110_kernel[grid](aa, result, N, BLOCK_SIZE=BLOCK_SIZE)
    
    max_val = result[0]
    xindex = int(result[1].item())
    yindex = int(result[2].item())
    
    return max_val + (xindex + 1) + (yindex + 1)