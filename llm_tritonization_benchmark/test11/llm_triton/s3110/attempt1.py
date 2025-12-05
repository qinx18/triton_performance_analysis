import torch
import triton
import triton.language as tl

@triton.jit
def s3110_kernel(
    aa_ptr,
    result_ptr,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Initialize with aa[0][0]
    max_val = tl.load(aa_ptr)
    max_i = 0
    max_j = 0
    
    # Sequential loop over i dimension
    for i in range(LEN_2D):
        # Parallel processing of j dimension
        j_offsets = tl.arange(0, BLOCK_SIZE)
        j_start = pid * BLOCK_SIZE
        j_indices = j_start + j_offsets
        j_mask = j_indices < LEN_2D
        
        if tl.sum(j_mask.to(tl.int32)) > 0:
            # Load values for current i, block of j values
            aa_offsets = i * LEN_2D + j_indices
            vals = tl.load(aa_ptr + aa_offsets, mask=j_mask, other=float('-inf'))
            
            # Find maximum in this block
            block_max = tl.max(vals)
            
            # Check if block maximum is greater than current global max
            if block_max > max_val:
                # Find the exact position of maximum in this block
                max_mask = vals == block_max
                # Get first occurrence of maximum
                max_positions = tl.where(max_mask, j_indices, LEN_2D)
                block_max_j = tl.min(max_positions)
                
                max_val = block_max
                max_i = i
                max_j = block_max_j
    
    # Store results
    tl.store(result_ptr + pid * 3, max_val)
    tl.store(result_ptr + pid * 3 + 1, max_i.to(tl.float32))
    tl.store(result_ptr + pid * 3 + 2, max_j.to(tl.float32))

def s3110_triton(aa):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = min(256, LEN_2D)
    
    # Calculate grid size
    grid_size = triton.cdiv(LEN_2D, BLOCK_SIZE)
    
    # Create result tensor to store max_val, max_i, max_j for each block
    result = torch.empty((grid_size, 3), dtype=aa.dtype, device=aa.device)
    
    # Launch kernel
    s3110_kernel[(grid_size,)](
        aa,
        result,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reduce across blocks to find global maximum
    block_max_vals = result[:, 0]
    block_max_indices = torch.argmax(block_max_vals)
    
    max_val = result[block_max_indices, 0]
    max_i = result[block_max_indices, 1].int()
    max_j = result[block_max_indices, 2].int()
    
    # Calculate checksum
    chksum = max_val + max_i.float() + max_j.float()
    
    return max_val + max_i + 1 + max_j + 1