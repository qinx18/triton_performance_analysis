import torch
import triton
import triton.language as tl

@triton.jit
def s13110_kernel(aa_ptr, output_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(0)
    
    # Initialize reduction values
    max_val = tl.load(aa_ptr)  # aa[0][0]
    max_i = 0
    max_j = 0
    
    # Sequential loop over i dimension
    for i in range(LEN_2D):
        # Parallel processing of j dimension
        j_offsets = tl.arange(0, BLOCK_SIZE)
        j_start = pid * BLOCK_SIZE
        j_idx = j_start + j_offsets
        j_mask = j_idx < LEN_2D
        
        if tl.sum(j_mask.to(tl.int32)) > 0:
            # Load aa[i][j] values for this row
            row_ptr = aa_ptr + i * LEN_2D + j_idx
            values = tl.load(row_ptr, mask=j_mask, other=float('-inf'))
            
            # Find local maximum in this block
            local_max = tl.max(values)
            
            # Check if local max is greater than global max
            if local_max > max_val:
                # Find which j index has the maximum
                max_mask = values == local_max
                # Get the first occurrence
                j_positions = tl.where(max_mask, j_idx, LEN_2D)
                local_max_j = tl.min(j_positions)
                
                # Update global maximum
                max_val = local_max
                max_i = i
                max_j = local_max_j
    
    # Store results (each thread stores its own result, will be reduced later)
    thread_id = pid
    output_offset = thread_id * 3
    tl.store(output_ptr + output_offset, max_val)
    tl.store(output_ptr + output_offset + 1, max_i.to(tl.float32))
    tl.store(output_ptr + output_offset + 2, max_j.to(tl.float32))

def s13110_triton(aa):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    
    # Calculate grid size
    grid_size = triton.cdiv(LEN_2D, BLOCK_SIZE)
    
    # Output tensor to store results from each block
    output = torch.zeros((grid_size, 3), dtype=torch.float32, device=aa.device)
    
    # Launch kernel
    grid = (grid_size,)
    s13110_kernel[grid](
        aa, output, LEN_2D, BLOCK_SIZE
    )
    
    # Reduce results from all blocks to find global maximum
    max_vals = output[:, 0]
    max_indices = output[:, 1:]
    
    # Find the block with the maximum value
    global_max_idx = torch.argmax(max_vals)
    max_val = max_vals[global_max_idx]
    max_i = int(max_indices[global_max_idx, 0])
    max_j = int(max_indices[global_max_idx, 1])
    
    # Calculate chksum
    chksum = max_val + float(max_i) + float(max_j)
    
    return max_val + max_i + 1 + max_j + 1