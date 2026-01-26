import torch
import triton
import triton.language as tl

@triton.jit
def s13110_kernel(aa_ptr, result_ptr, len_2d: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Initialize for this block
    block_max = tl.load(aa_ptr)  # aa[0][0]
    block_xindex = 0
    block_yindex = 0
    
    # Sequential loop over i (rows)
    for i in range(len_2d):
        # Parallel processing of j (columns) in this block
        j_offsets = tl.arange(0, BLOCK_SIZE)
        j_start = pid * BLOCK_SIZE
        j_indices = j_start + j_offsets
        j_mask = j_indices < len_2d
        
        # Load values for this row
        row_ptr = aa_ptr + i * len_2d + j_indices
        values = tl.load(row_ptr, mask=j_mask, other=float('-inf'))
        
        # Find max within this block for this row
        row_max = tl.max(values)
        
        # Check if this row max is greater than current global max
        if row_max > block_max:
            block_max = row_max
            block_xindex = i
            
            # Find which j position had the max value
            max_mask = values == row_max
            max_positions = tl.where(max_mask, j_indices, len_2d)
            block_yindex = tl.min(max_positions)
    
    # Store result for this block
    result_offset = pid * 3
    tl.store(result_ptr + result_offset, block_max)
    tl.store(result_ptr + result_offset + 1, block_xindex.to(tl.float32))
    tl.store(result_ptr + result_offset + 2, block_yindex.to(tl.float32))

def s13110_triton(aa, len_2d):
    BLOCK_SIZE = 64
    n_blocks = triton.cdiv(len_2d, BLOCK_SIZE)
    
    # Result tensor to store max, xindex, yindex for each block
    result = torch.empty(n_blocks * 3, dtype=torch.float32, device=aa.device)
    
    grid = (n_blocks,)
    s13110_kernel[grid](
        aa, result, len_2d, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Find global maximum across all blocks
    block_results = result.view(n_blocks, 3)
    max_values = block_results[:, 0]
    
    # Find which block had the global maximum
    global_max_block_idx = torch.argmax(max_values)
    global_max = block_results[global_max_block_idx, 0]
    global_xindex = int(block_results[global_max_block_idx, 1])
    global_yindex = int(block_results[global_max_block_idx, 2])
    
    # Calculate final result (note: +1 for indices as in C code)
    return global_max + global_xindex + 1 + global_yindex + 1