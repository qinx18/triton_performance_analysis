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
        
        # Find which values are greater than current max
        greater_mask = values > max_val
        greater_mask = greater_mask & j_mask
        
        # If any value is greater, find the maximum among them
        if tl.sum(greater_mask.to(tl.int32)) > 0:
            # Get the maximum value among the greater ones
            masked_values = tl.where(greater_mask, values, float('-inf'))
            block_max = tl.max(masked_values)
            
            # Update global max if this block max is greater
            if block_max > max_val:
                max_val = block_max
                max_i = i
                # Find which j had the maximum value
                max_locations = (values == block_max) & greater_mask
                j_candidates = tl.where(max_locations, j_idx, LEN_2D)
                max_j = tl.min(j_candidates)  # Get the smallest j index with max value
    
    # Store results (each block writes its findings)
    if pid == 0:
        tl.store(max_out_ptr, max_val)
        tl.store(xindex_out_ptr, max_i)
        tl.store(yindex_out_ptr, max_j)

def s3110_triton(aa):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    
    # Flatten aa for easier access
    aa_flat = aa.contiguous().view(-1)
    
    # Create output tensors for results
    max_out = torch.zeros(1, dtype=aa.dtype, device=aa.device)
    xindex_out = torch.zeros(1, dtype=torch.int32, device=aa.device)
    yindex_out = torch.zeros(1, dtype=torch.int32, device=aa.device)
    
    # Since this is a global reduction, we need a different approach
    # Use a single block to process the entire array sequentially
    grid = (1,)
    
    # Custom kernel for global max finding
    s3110_global_max_kernel[grid](
        aa_flat, max_out, xindex_out, yindex_out, 
        LEN_2D, BLOCK_SIZE=min(BLOCK_SIZE, LEN_2D)
    )
    
    max_val = max_out.item()
    xindex = xindex_out.item()
    yindex = yindex_out.item()
    
    chksum = max_val + float(xindex) + float(yindex)
    
    return max_val + xindex + 1 + yindex + 1

@triton.jit
def s3110_global_max_kernel(aa_ptr, max_out_ptr, xindex_out_ptr, yindex_out_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Initialize with aa[0][0]
    max_val = tl.load(aa_ptr)
    max_i = 0
    max_j = 0
    
    # Process in blocks of BLOCK_SIZE elements
    for start_idx in range(0, LEN_2D * LEN_2D, BLOCK_SIZE):
        offsets = tl.arange(0, BLOCK_SIZE)
        indices = start_idx + offsets
        mask = indices < LEN_2D * LEN_2D
        
        values = tl.load(aa_ptr + indices, mask=mask, other=float('-inf'))
        
        # Check if any value is greater than current max
        greater_mask = values > max_val
        greater_mask = greater_mask & mask
        
        if tl.sum(greater_mask.to(tl.int32)) > 0:
            # Find the maximum among greater values
            masked_values = tl.where(greater_mask, values, float('-inf'))
            block_max = tl.max(masked_values)
            
            if block_max > max_val:
                max_val = block_max
                # Find the linear index of the maximum
                max_locations = (values == block_max) & greater_mask
                linear_candidates = tl.where(max_locations, indices, LEN_2D * LEN_2D)
                linear_idx = tl.min(linear_candidates)
                
                # Convert linear index to 2D coordinates
                max_i = linear_idx // LEN_2D
                max_j = linear_idx % LEN_2D
    
    tl.store(max_out_ptr, max_val)
    tl.store(xindex_out_ptr, max_i)
    tl.store(yindex_out_ptr, max_j)