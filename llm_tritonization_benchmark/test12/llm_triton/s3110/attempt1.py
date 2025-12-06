import torch
import triton
import triton.language as tl

@triton.jit
def s3110_kernel(aa_ptr, output_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Initialize reduction variables
    max_val = tl.load(aa_ptr)  # aa[0][0]
    max_i = 0
    max_j = 0
    
    # Sequential loop over i dimension
    for i in range(LEN_2D):
        # Parallel processing of j dimension
        j_start = pid * BLOCK_SIZE
        j_offsets = tl.arange(0, BLOCK_SIZE)
        j_indices = j_start + j_offsets
        j_mask = j_indices < LEN_2D
        
        if j_start < LEN_2D:
            # Load values for this i row
            row_ptr = aa_ptr + i * LEN_2D + j_indices
            values = tl.load(row_ptr, mask=j_mask, other=float('-inf'))
            
            # Find elements greater than current max
            greater_mask = values > max_val
            greater_mask = greater_mask & j_mask
            
            # Check if any element in this block is greater
            if tl.max(tl.where(greater_mask, 1, 0)) > 0:
                # Find the maximum value in this block that's greater than current max
                candidate_values = tl.where(greater_mask, values, float('-inf'))
                block_max = tl.max(candidate_values)
                
                # If this block's max is better than current max
                if block_max > max_val:
                    max_val = block_max
                    max_i = i
                    
                    # Find the j index of the maximum value
                    max_positions = candidate_values == block_max
                    max_j_offsets = tl.where(max_positions, j_indices, LEN_2D)
                    max_j = tl.min(max_j_offsets)
    
    # Store results
    tl.store(output_ptr + pid * 3, max_val)
    tl.store(output_ptr + pid * 3 + 1, max_i.to(tl.float32))
    tl.store(output_ptr + pid * 3 + 2, max_j.to(tl.float32))

def s3110_triton(aa):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    num_blocks = triton.cdiv(LEN_2D, BLOCK_SIZE)
    
    # Output tensor to collect results from all blocks
    output = torch.zeros((num_blocks, 3), dtype=aa.dtype, device=aa.device)
    
    # Launch kernel
    grid = (num_blocks,)
    s3110_kernel[grid](
        aa, output, 
        LEN_2D=LEN_2D, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Reduce results across blocks to find global maximum
    max_vals = output[:, 0]
    max_is = output[:, 1]
    max_js = output[:, 2]
    
    global_max_idx = torch.argmax(max_vals)
    max_val = max_vals[global_max_idx]
    xindex = int(max_is[global_max_idx])
    yindex = int(max_js[global_max_idx])
    
    chksum = max_val + float(xindex) + float(yindex)
    
    return max_val + xindex + 1 + yindex + 1