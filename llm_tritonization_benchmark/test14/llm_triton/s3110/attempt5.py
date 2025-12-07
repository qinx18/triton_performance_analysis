import torch
import triton
import triton.language as tl

@triton.jit
def s3110_kernel(aa_ptr, result_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < LEN_2D
    
    # Initialize with aa[0][0]
    init_val = tl.load(aa_ptr)
    max_val = tl.full((BLOCK_SIZE,), init_val, dtype=tl.float32)
    max_i = tl.full((BLOCK_SIZE,), 0, dtype=tl.int32)
    max_j = tl.full((BLOCK_SIZE,), 0, dtype=tl.int32)
    
    # Loop over all rows
    for i in range(LEN_2D):
        # Load values for current row and this block's j indices
        row_ptr = aa_ptr + i * LEN_2D + j_idx
        values = tl.load(row_ptr, mask=j_mask, other=init_val - 1.0)
        
        # Update maximums where values are greater
        update_mask = (values > max_val) & j_mask
        max_val = tl.where(update_mask, values, max_val)
        max_i = tl.where(update_mask, i, max_i)
        max_j = tl.where(update_mask, j_idx, max_j)
    
    # Store results - need to reduce across the block to find global max
    tl.store(result_ptr + pid * 4, max_val)
    tl.store(result_ptr + pid * 4 + 1, max_i.to(tl.float32))
    tl.store(result_ptr + pid * 4 + 2, max_j.to(tl.float32))

@triton.jit
def s3110_reduce_kernel(temp_results_ptr, final_result_ptr, num_blocks: tl.constexpr):
    # Load first block's results
    max_val = tl.load(temp_results_ptr)
    max_i = tl.load(temp_results_ptr + 1)
    max_j = tl.load(temp_results_ptr + 2)
    
    # Check remaining blocks
    for b in range(1, num_blocks):
        val = tl.load(temp_results_ptr + b * 4)
        is_greater = val > max_val
        if is_greater:
            max_val = val
            max_i = tl.load(temp_results_ptr + b * 4 + 1)
            max_j = tl.load(temp_results_ptr + b * 4 + 2)
    
    # Store final results
    tl.store(final_result_ptr, max_val)
    tl.store(final_result_ptr + 1, max_i)
    tl.store(final_result_ptr + 2, max_j)

def s3110_triton(aa):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    num_blocks = triton.cdiv(LEN_2D, BLOCK_SIZE)
    
    # Temporary storage for partial results (max_val, max_i, max_j, padding)
    temp_results = torch.zeros(num_blocks * 4, dtype=aa.dtype, device=aa.device)
    
    # Launch first kernel
    grid = (num_blocks,)
    s3110_kernel[grid](
        aa, temp_results,
        LEN_2D, BLOCK_SIZE
    )
    
    # Final result storage
    final_results = torch.zeros(3, dtype=aa.dtype, device=aa.device)
    
    # Reduce partial results
    grid = (1,)
    s3110_reduce_kernel[grid](
        temp_results, final_results,
        num_blocks
    )
    
    max_val = final_results[0].item()
    xindex = int(final_results[1].item())
    yindex = int(final_results[2].item())
    
    return max_val + xindex + 1 + yindex + 1