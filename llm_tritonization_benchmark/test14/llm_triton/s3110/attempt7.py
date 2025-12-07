import torch
import triton
import triton.language as tl

@triton.jit
def s3110_kernel(aa_ptr, result_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr, num_blocks: tl.constexpr):
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
    
    # Store results for valid elements only
    valid_offsets = tl.arange(0, BLOCK_SIZE)
    valid_mask = (valid_offsets < LEN_2D) & j_mask
    
    result_base = result_ptr + pid * BLOCK_SIZE
    tl.store(result_base + valid_offsets, max_val, mask=valid_mask)
    tl.store(result_base + num_blocks * BLOCK_SIZE + valid_offsets, max_i.to(tl.float32), mask=valid_mask)
    tl.store(result_base + 2 * num_blocks * BLOCK_SIZE + valid_offsets, max_j.to(tl.float32), mask=valid_mask)

@triton.jit
def s3110_reduce_kernel(temp_results_ptr, final_result_ptr, num_elements: tl.constexpr, num_blocks: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Load first element
    max_val = tl.load(temp_results_ptr)
    max_i = tl.load(temp_results_ptr + num_blocks * BLOCK_SIZE)
    max_j = tl.load(temp_results_ptr + 2 * num_blocks * BLOCK_SIZE)
    
    # Check remaining elements
    for idx in range(1, num_elements):
        val = tl.load(temp_results_ptr + idx)
        val_i = tl.load(temp_results_ptr + num_blocks * BLOCK_SIZE + idx)
        val_j = tl.load(temp_results_ptr + 2 * num_blocks * BLOCK_SIZE + idx)
        
        is_greater = val > max_val
        max_val = tl.where(is_greater, val, max_val)
        max_i = tl.where(is_greater, val_i, max_i)
        max_j = tl.where(is_greater, val_j, max_j)
    
    # Store final results
    tl.store(final_result_ptr, max_val)
    tl.store(final_result_ptr + 1, max_i)
    tl.store(final_result_ptr + 2, max_j)

def s3110_triton(aa):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    num_blocks = triton.cdiv(LEN_2D, BLOCK_SIZE)
    
    # Temporary storage for partial results
    temp_results = torch.zeros(num_blocks * BLOCK_SIZE * 3, dtype=aa.dtype, device=aa.device)
    
    # Launch first kernel
    grid = (num_blocks,)
    s3110_kernel[grid](
        aa, temp_results,
        LEN_2D, BLOCK_SIZE, num_blocks
    )
    
    # Final result storage
    final_results = torch.zeros(3, dtype=aa.dtype, device=aa.device)
    
    # Reduce partial results
    grid = (1,)
    s3110_reduce_kernel[grid](
        temp_results, final_results,
        LEN_2D, num_blocks, BLOCK_SIZE
    )
    
    max_val = final_results[0].item()
    xindex = int(final_results[1].item())
    yindex = int(final_results[2].item())
    
    return max_val + xindex + 1 + yindex + 1